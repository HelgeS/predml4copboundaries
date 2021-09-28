import logging
import tempfile

import numpy as np
from keras import regularizers, callbacks
from keras.engine import Input, Model
from keras.layers import average, maximum, Dense
from keras.optimizers import Adam
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split

import keras_losses as losses
from keras_helper import median, minimum, topKvalue

loss_dict = {
    ('linex', 'under'): losses.linex_underestimate_fn,
    ('linex', 'over'): losses.linex_overestimate_fn,
    ('shiftedmse', 'under'): losses.shifted_mse_underestimate_fn,
    ('shiftedmse', 'over'): losses.shifted_mse_overestimate_fn,
    ('peann', 'under'): losses.pe_ann_underestimate_fn,
    ('peann', 'over'): losses.pe_ann_overestimate_fn,
    ('mse', 'under'): lambda x: 'mean_squared_error',
    ('mse', 'over'): lambda x: 'mean_squared_error'
}

LAYERS = 5
HIDDEN_NODES = 64


def get_network(features, outputs, loss_name, problem, loss_factor=1, scaled_output=False, asymmetry=None):
    estimator = create_nn_model(**{'input_dim': features,
                                   'outputs': outputs,
                                   'loss_name': loss_name,
                                   'loss_asymmetry_factor': loss_factor,
                                   'problem': problem,
                                   'scaled_output': scaled_output,
                                   'lr': 0.001 if scaled_output else 0.01,
                                   'asymmetry': asymmetry})
    return estimator


def create_nn_model(input_dim, outputs, loss_name, problem, loss_asymmetry_factor=1, hidden_nodes=HIDDEN_NODES,
                    hidden_layers=LAYERS, lr=0.01, in_ensemble=False, scaled_output=False, asymmetry=None):
    assert (1 <= outputs <= 2)

    initializer = 'glorot_uniform'
    # initializer = 'ones'
    # initializer = initializers.RandomNormal(mean=0.0, stddev=0.0001)
    # initializer = initializers.RandomUniform(minval=-0.08, maxval=0.08)
    # initializer = initializers.VarianceScaling(scale=1.0, mode='fan_avg', distribution='normal')

    regularizer = regularizers.l2(0.0001)

    if not in_ensemble:
        inp = Input(shape=(input_dim,))
    else:
        inp = in_ensemble

    x = Dense(hidden_nodes, activation='relu',
              kernel_initializer=initializer,
              kernel_regularizer=regularizer,
              use_bias=False)(inp)

    for i in range(1, hidden_layers - (outputs == 2)):
        x = Dense(hidden_nodes, activation='relu',
                  bias_initializer='ones',
                  kernel_initializer=initializer,
                  kernel_regularizer=regularizer)(x)

    if scaled_output:
        output_activation = 'sigmoid'
    else:
        output_activation = 'linear'

    if outputs == 1 or asymmetry is not None:
        out = Dense(1, activation=output_activation,
                    bias_initializer='ones')(x)
    elif outputs == 2:
        xu = Dense(hidden_nodes // 2, activation='relu', bias_initializer='ones')(x)
        out_under = Dense(1, activation=output_activation, bias_initializer='ones')(xu)

        xo = Dense(hidden_nodes // 2, activation='relu', bias_initializer='ones')(x)
        out_over = Dense(1, activation=output_activation, bias_initializer='ones')(xo)

        out = [out_under, out_over]

    if in_ensemble:
        return out

    if outputs == 1 or asymmetry is not None:
        if asymmetry is not None:
            assert asymmetry in ['over', 'under'], "Invalid asymmetry parameter: {}".format(asymmetry)
            loss_fn = loss_dict[(loss_name, asymmetry)](loss_asymmetry_factor)
        elif problem.minmax == 'min':
            loss_fn = loss_dict[(loss_name, 'over')](loss_asymmetry_factor)
        else:
            loss_fn = loss_dict[(loss_name, 'under')](loss_asymmetry_factor)

        out = [out]
    elif outputs == 2:
        loss_fn = [loss_dict[(loss_name, 'under')](loss_asymmetry_factor),
                   loss_dict[(loss_name, 'over')](loss_asymmetry_factor)]

    # Compile model
    # optimizer = RMSprop(lr=0.000001)
    optimizer = Adam(lr=lr)
    model = Model([inp], out)
    model.compile(loss=loss_fn, optimizer=optimizer, metrics=['mape'])
    return model


def create_ensemble_model(input_dim, outputs, loss_name, problem, loss_asymmetry_factor=1, hidden_nodes=HIDDEN_NODES,
                          hidden_layers=LAYERS, lr=0.01, members=5, mode='average'):
    models = []

    for i in range(members):
        m = create_nn_model(input_dim=input_dim, outputs=outputs, loss_fn=loss_name, hidden_nodes=hidden_nodes,
                            hidden_layers=hidden_layers, lr=lr, in_ensemble=True)
        models.append(m)

    full_model = form_ensemble(models, input_dim, loss_name, loss_asymmetry_factor, mode=mode, minmax=problem.minmax)
    return full_model


def form_ensemble(models, input_dim, loss_name, loss_asymmetry_factor=1, mode='extreme', minmax='min'):
    inp = Input(shape=(input_dim,))

    models = [m.model if isinstance(m, KerasRegressor) else m for m in models]

    if len(models[0].outputs) == 1:
        outputs = [m(inp) for m in models]

        if mode == 'average':
            merged = [average(outputs)]
        elif mode == 'median':
            merged = [median(outputs)]
        elif mode == 'extreme' and minmax == 'min':
            merged = [maximum(outputs)]
        elif mode == 'extreme' and minmax == 'max':
            merged = [minimum(outputs)]
        elif mode == 'leaveoneout' and minmax == 'min':
            merged = [topKvalue(outputs, k=2)]
        elif mode == 'leaveoneout' and minmax == 'max':
            merged = [topKvalue(outputs, k=len(models) - 1)]
        else:
            raise Exception('Unknown ensemble mode: %s' % mode)

        if minmax == 'min':
            loss_fn = loss_dict[(loss_name, 'over')](loss_asymmetry_factor)
        else:
            loss_fn = loss_dict[(loss_name, 'under')](loss_asymmetry_factor)
    elif len(models[0].outputs) == 2:
        outputs_under, outputs_over = map(list, zip(*[m(inp) for m in models]))

        if mode == 'average':
            merged = [average(outputs_under, name='underestimation'), average(outputs_over, name='overestimation')]
        elif mode == 'extreme':
            merged = [minimum(outputs_under, name='underestimation'), maximum(outputs_over, name='overestimation')]
        elif mode == 'median':
            merged = [median(outputs_under, name='underestimation'), median(outputs_over, name='overestimation')]
        elif mode == 'leaveoneout':
            merged = [topKvalue(outputs_under, k=len(models) - 1, name='underestimation'),
                      topKvalue(outputs_over, k=2, name='overestimation')]
        else:
            raise Exception('Unknown ensemble mode: %s' % mode)

        loss_fn = [loss_dict[(loss_name, 'under')](loss_asymmetry_factor),
                   loss_dict[(loss_name, 'over')](loss_asymmetry_factor)]
    else:
        raise Exception('Unexpected number of outputs for ensemble')

    full_model = Model([inp], merged)
    full_model.compile(loss=loss_fn, optimizer=Adam(), metrics=['mape'])
    return full_model


def train(estimator, X_all, y_all, val_split=0.2, max_iter=10, epochs=150, batch_size=32, verbose=False):
    X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=val_split)

    if y_train.ndim == 2:
        y_train = np.split(y_train, y_train.shape[1], axis=1)
        y_val = np.split(y_val, y_val.shape[1], axis=1)

    for i in range(max_iter):
        reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
        early_stop = callbacks.EarlyStopping(patience=11)

        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=False) as modelfile:
            modelfile.close()
            modelcheck = callbacks.ModelCheckpoint(modelfile.name, monitor='val_loss', verbose=verbose,
                                                   save_best_only=True, save_weights_only=True, mode='auto', period=1)

            hist = estimator.fit(X_train, y_train,
                                 batch_size=min(batch_size, X_train.shape[0]),
                                 epochs=epochs,
                                 validation_data=(X_val, y_val),
                                 callbacks=[reduce_lr, modelcheck, early_stop],
                                 verbose=2 if verbose else 0)

            has_fitted = 'loss' in hist.history and 0.0 not in hist.history['loss'] and not any(
                np.isnan(hist.history['loss']))

            if has_fitted:
                estimator.load_weights(modelfile.name)
                break
            else:
                logging.info('Fit again')
    else:
        raise Exception('Could not fit model after %d tries' % max_iter)

    return estimator, hist


def predict(estimator, features):
    if isinstance(estimator, (list, tuple)):
        predicted = [est.predict(features) for est in estimator]
    else:
        predicted = estimator.predict(features)

    # Models with two outputs return a list, but we prefer to work on numpy arrays
    if not isinstance(predicted, (np.ndarray, np.generic)):
        predicted = np.hstack(predicted)

    # Make sure each result is in one row
    if features.shape[0] == predicted.shape[1]:
        predicted = predicted.transpose()

    if predicted.shape[1] == 1:
        predicted = predicted[:, 0]

    return predicted


def grid_search(problem, outputs, X_train, y_train):
    search_parameters = {
        'hidden_nodes': [16, 32, 64, 128],
        'hidden_layers': [2, 3, 5, 10, 16],
        'lr': [0.01, 0.001, 0.0001, 0.00001]
    }

    def create_model(hidden_nodes, hidden_layers, lr):
        return create_nn_model(X_train.shape[1], outputs, loss_name='shiftedmse', loss_asymmetry_factor=0.2,
                               problem=problem, hidden_nodes=hidden_nodes, hidden_layers=hidden_layers, lr=lr)

    estimator = KerasRegressor(create_model, hidden_nodes=HIDDEN_NODES, hidden_layers=LAYERS, lr=0.01)

    def loss(y_true, y_pred, a=-0.4):
        x = y_pred - y_true
        return np.mean(np.power(x, 2) * np.power(np.sign(x) + a, 2))

    scorer = make_scorer(loss, greater_is_better=False)

    clf = GridSearchCV(estimator, search_parameters, cv=5, n_jobs=1, scoring=scorer)  # 'neg_mean_squared_error')
    clf.fit(X_train, y_train)

    print("Best parameters set found on training set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on training set:")
    print()

    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

    return clf


class NetworkWrapper(object):
    """Wrapper for the network model, only to provide an instance function predict(X)"""

    def __init__(self, estimator):
        self.estimator = estimator

    def predict(self, features):
        return predict(self.estimator, features)
