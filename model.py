from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from xgboost import XGBRegressor

import network


def get_trained_model(X_train, y_train_values, args, problem):
    if args.estimator in ['network', 'networka']:
        estimator = get_trained_network(X_train, y_train_values, args, problem)
    elif args.estimator in ['bayridge', 'gp']:
        estimator = get_model(args)
        estimator.fit(X_train, y_train_values[:, 0])
    else:
        estimator = get_model(args)
        estimator.fit(X_train, y_train_values)

    return estimator


def predict(estimator, X_test, args, scaling_factors=None):
    """Model prediction and adjustment for roundings, formatting, etc."""
    assert scaling_factors is None or scaling_factors.shape[1] == 2

    if args.estimator in ['bayridge', 'gp']:
        mean_preds, std_preds = estimator.predict(X_test, return_std=True)
        ds_predicted = np.zeros((X_test.shape[0], 2))
        # Calculate both bounds independent of outputs / unneeded bound will be removed later
        ds_predicted[:, 0] = mean_preds - 1.96 * std_preds
        ds_predicted[:, 1] = mean_preds + 1.96 * std_preds
    else:
        ds_predicted = estimator.predict(X_test)

    if scaling_factors is not None:
        scale_diff = (scaling_factors['dom_upper'] - scaling_factors['dom_lower']).values.reshape(-1, 1)
        scale_diff = np.repeat(scale_diff, args.outputs, axis=1)
        addend = np.repeat(scaling_factors['dom_lower'].values.reshape(-1, 1), args.outputs, axis=1)
        ds_predicted = (ds_predicted * scale_diff) + addend

    # Post-process predictions to rounded integers
    if args.outputs == 1:
        if args.problem.minmax == 'min':
            ds_predicted = np.ceil(ds_predicted).astype(int)
            ds_predicted = np.vstack([np.full(ds_predicted.shape, np.nan), ds_predicted]).T
        else:
            ds_predicted = np.floor(ds_predicted).astype(int)
            ds_predicted = np.vstack([ds_predicted, np.full(ds_predicted.shape, np.nan)]).T
    else:
        ds_predicted[:, 0] = np.floor(ds_predicted[:, 0]).astype(int)
        ds_predicted[:, 1] = np.ceil(ds_predicted[:, 1]).astype(int)

    assert (ds_predicted.shape == (X_test.shape[0], 2))

    return ds_predicted


def get_trained_network(X, y, args, problem):
    if args.ensemble > 1:
        estimators = []
        histories = []

        for i in range(args.ensemble):
            est = network.get_network(X.shape[1], args.outputs, args.loss_fn, problem, args.loss_factor,
                                      scaled_output=args.scaled_prediction)
            est, hist = network.train(est, X, y, val_split=0.2, verbose=args.verbose)
            estimators.append(est)
            histories.append(hist)

        estimator = network.form_ensemble(estimators, X.shape[1], args.loss_fn, args.loss_factor,
                                          args.ensemble_mode, problem.minmax)
    else:
        if args.outputs == 2:
            underest = network.get_network(X.shape[1], 1, args.loss_fn, problem, args.loss_factor,
                                            scaled_output=args.scaled_prediction, asymmetry='under')
            overest = network.get_network(X.shape[1], 1, args.loss_fn, problem, args.loss_factor,
                                            scaled_output=args.scaled_prediction, asymmetry='over')
            underest, hist = network.train(underest, X, y[:, 0], val_split=0.2, verbose=args.verbose)
            overest, hist = network.train(overest, X, y[:, 1], val_split=0.2, verbose=args.verbose)
            estimator = [underest, overest]
        else:
            estimator = network.get_network(X.shape[1], args.outputs, args.loss_fn, problem, args.loss_factor,
                                            scaled_output=args.scaled_prediction)
            estimator, hist = network.train(estimator, X, y, val_split=0.2, verbose=args.verbose)

    wrapped_estimator = network.NetworkWrapper(estimator)

    return wrapped_estimator  # , hist


def get_model(args):
    if args.estimator == 'linear':
        model = LinearRegression()
    elif args.estimator == 'ridge':
        model = BayesianRidge()
    elif args.estimator == 'knn':
        model = KNeighborsRegressor()
    elif args.estimator == 'svm':
        model = SVR()
    elif args.estimator == 'forest':
        model = RandomForestRegressor()
    elif args.estimator == 'gp':
        model = GaussianProcessRegressor(normalize_y=True)
    elif args.estimator == 'bayridge':
        model = BayesianRidge()
    elif args.estimator == 'xgb':
        model = XGBRegressor()
    elif args.estimator == 'xgba':
        return get_xgb_asymm(args)
    else:
        raise NotImplementedError('Unsupported estimator {}'.format(args.estimator))

    if args.outputs == 1 or args.estimator in ['bayridge', 'gp', 'xgba']:
        return model
    else:
        return MultiOutputRegressor(model)


def get_xgb_asymm(args):
    if args.loss_fn == 'shiftedmse':
        loss_under = shifted_mse_underestimate_fn(args.loss_factor)
        loss_over = shifted_mse_overestimate_fn(args.loss_factor)
    elif args.loss_fn == 'linex':
        loss_under = linex_underestimate_fn(args.loss_factor)
        loss_over = linex_overestimate_fn(args.loss_factor)
    else:
        raise NotImplementedError('Unsupported loss function {} for XGBa'.format(args.loss_fn))

    if args.outputs == 2:
        underest = XGBRegressor(objective=loss_under)
        overest = XGBRegressor(objective=loss_over)
        return CoupledModel((underest, overest))
    else:
        if args.problem.minmax == 'min':
            loss_fn = loss_over
        else:
            loss_fn = loss_under

        return XGBRegressor(objective=loss_fn)


class CoupledModel(object):
    """
    Dummy class to bundle multiple trained estimators.
    It can only predict, nothing else.
    """

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        assert (len(self.models) == 1 and y.ndim == 1) or (y.shape[1] == len(self.models)), \
            "Number of models does not match dimensions of targets"

        for i, m in enumerate(self.models):
            m.fit(X, y[:, i])
            #assert np.isfinite(m.feature_importances_).all(), \
            #    "Infinite feature importance / check labels"

    def predict(self, X):
        predictions = np.array([est.predict(X) for est in self.models]).T
        assert predictions.shape == (X.shape[0], len(self.models))
        return predictions


def shifted_mse_underestimate_fn(factor):
    return lambda y_true, y_pred: shifted_mse(y_true, y_pred, factor)


def shifted_mse_overestimate_fn(factor):
    return lambda y_true, y_pred: shifted_mse(y_true, y_pred, -factor)


def shifted_mse_underestimate(y_true, y_pred):
    return shifted_mse(y_true, y_pred, 0.95)


def shifted_mse_overestimate(y_true, y_pred):
    return shifted_mse(y_true, y_pred, -0.95)


def shifted_mse(y_true, y_pred, a):
    if a == 0 or not (-1 < a < 1):
        raise ValueError

    x = y_pred - y_true
    grad = lambda x: 2 * x * np.power(a + np.sign(x), 2)
    hess = lambda x: 2 * np.power(a + np.sign(x), 2)
    return grad(x), hess(x)


def linex_underestimate_fn(factor=1):
    return lambda y_true, y_pred: linex_loss(y_true, y_pred, factor)


def linex_overestimate_fn(factor=1):
    return lambda y_true, y_pred: linex_loss(y_true, y_pred, -factor)


def linex_underestimate(y_true, y_pred):
    return linex_loss(y_true, y_pred, 1)


def linex_overestimate(y_true, y_pred):
    return linex_loss(y_true, y_pred, -1)


def linex_loss(y_true, y_pred, a):
    """
    LinEx(eps) = b * (e^(a * eps) - a * eps - 1)

    A. Zellner, “Bayesian Estimation and Prediction Using Asymmetric Loss Functions,”
    J. Am. Stat. Assoc., vol. 81, no. 394, pp. 446–451, 1986.
    """
    assert a != 0

    b = 1
    delta = y_pred - y_true

    # func = b * (np.exp(a * delta) - a * delta - 1)
    grad = a * b * (np.exp(a * delta) - 1)
    hess = a ** 2 * b * np.exp(a * delta)
    return grad, hess
