import argparse
import csv
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config
import features
import model
from solve.helper import export_configurations


def main(args):
    problem = args.problem
    print('Problem %s' % problem.name)

    file_prefix = os.path.join(config.OUTPUT_DIR, '{}_{}'.format(problem.name, args.estimator))

    if args.file_suffix:
        file_prefix += args.file_suffix

    all_stats = []

    logging.info('Start training...')

    X_all, y_all = features.cached_feature_matrix(problem, include_opt=True, include_mzn2feat=not args.smallfeat)

    rkf = RepeatedKFold(n_splits=config.KFOLD, n_repeats=args.iter)

    for i, (train_idx, test_idx) in enumerate(rkf.split(X_all), start=1):
        X_train = np.array(X_all[train_idx, :], copy=True)
        y_train = y_all.iloc[train_idx]

        X_test = np.array(X_all[test_idx, :], copy=True)
        y_test = y_all.iloc[test_idx]

        if args.scaled_prediction:
            X_train_dom_bounds = y_train[['dom_lower', 'dom_upper']]  # Also in X_train, but without labeled columns
            X_test_dom_bounds = y_test[['dom_lower', 'dom_upper']]  # Also in X_test, but without labeled columns
        else:
            X_train_dom_bounds = X_test_dom_bounds = None

        start = time.time()

        # Label Shift: Scale target values for under-/overestimation
        y_train_values = training_labels(y_train, args, X_train_dom_bounds)

        # Preprocess input features
        X_train, X_test = preprocess_features(X_train, X_test, args)

        # Build estimator and train it
        estimator = model.get_trained_model(X_train, y_train_values, args, problem)

        # Predict on test set
        ds_predicted = model.predict(estimator, X_test, args, X_test_dom_bounds)

        # setattr(estimator, 'preprocessor', preprocessors)

        traintime = time.time() - start

        # Data organization and book keeping
        results = list(zip(y_test.index, y_test['optimum'], ds_predicted[:, 0], ds_predicted[:, 1],
                           y_test['dom_lower'], y_test['dom_upper']))
        prediction = pd.DataFrame(results, columns=['dzn', 'optimum', 'underest', 'overest', 'dom_lower', 'dom_upper'])

        # Repair misestimations
        if args.outputs == 2:
            repairable = prediction['overest'] < prediction['underest']
            new_under = prediction.loc[repairable, 'overest']
            new_over = prediction.loc[repairable, 'underest']

            prediction.loc[repairable, 'underest'] = new_under
            prediction.loc[repairable, 'overest'] = new_over

            repaired_pairs = repairable.sum()
        else:
            repaired_pairs = 0

        underestimates = prediction['optimum'] - prediction['underest']
        overestimates = prediction['overest'] - prediction['optimum']
        true_underest = underestimates >= 0.0
        true_overest = overestimates >= 0.0
        pruned_lower = true_underest & (prediction['underest'] > prediction['dom_lower'])
        pruned_upper = true_overest & (prediction['overest'] < prediction['dom_upper'])
        underest_error = (
                (prediction[true_underest]['underest'] - prediction[true_underest]['optimum']) /
                prediction[true_underest][
                    'optimum']).abs().mean()
        overest_error = (
                (prediction[true_overest]['optimum'] - prediction[true_overest]['overest']) / prediction[true_overest][
            'optimum']).abs().mean()

        if args.outputs == 2:
            true_pairs = true_overest & true_underest
        elif problem.minmax == 'min':
            true_pairs = true_overest
        else:
            true_pairs = true_underest

        prediction['dom_upper_new'] = prediction['dom_upper']
        prediction.loc[pruned_upper, 'dom_upper_new'] = prediction['overest']

        prediction['dom_lower_new'] = prediction['dom_lower']
        prediction.loc[pruned_lower, 'dom_lower_new'] = prediction['underest']

        prediction['dom_size'] = prediction['dom_upper'] - prediction['dom_lower']
        prediction['dom_size_new'] = prediction['dom_upper_new'] - prediction['dom_lower_new']

        pruned_ratio = 1 - prediction[true_pairs]['dom_size_new'] / prediction[true_pairs]['dom_size']

        stats = {
            'problem': problem.name,
            'estimator': args.estimator,
            'adjustment': args.adjustment,
            'loss': args.loss_fn,
            'loss_factor': args.loss_factor,
            'outputs': args.outputs,
            'ensemble': args.ensemble,
            'ensemble_mode': args.ensemble_mode if args.ensemble > 1 else None,
            'iteration': i,
            'traintime': traintime,
            'nb_val_instances': len(prediction),
            'true_underest': true_underest.sum() / len(prediction),
            'true_overest': true_overest.sum() / len(prediction),
            'true_pairs': true_pairs.sum() / len(prediction),
            'underest_error': underest_error,
            'overest_error': overest_error,
            'pruned_lower_dom': pruned_lower.sum() / len(prediction),
            'pruned_upper_dom': pruned_upper.sum() / len(prediction),
            'pruned_domain': (true_pairs & (pruned_lower | pruned_upper)).sum() / len(prediction),
            'pruned_ratio': pruned_ratio.mean(),
            'orig_domain_size': prediction['dom_size'].mean(),
            'pruned_domain_size': prediction['dom_size_new'].mean(),
            'repaired_pairs': repaired_pairs
        }

        if args.output_stats:
            clean_stats = {k: v if not isinstance(v, np.generic) else np.asscalar(v) for k, v in stats.items()}
            writer = csv.DictWriter(open(args.output_stats, 'a'), fieldnames=sorted(list(clean_stats.keys())))
            if i == 1:
                writer.writeheader()

            writer.writerow(clean_stats)

        if args.output_predictions:
            prediction['problem'] = stats['problem']
            prediction['estimator'] = stats['estimator']
            prediction['adjustment'] = stats['adjustment']
            prediction['loss'] = stats['loss']
            prediction['outputs'] = stats['outputs']
            prediction['ensemble'] = stats['ensemble']
            prediction['ensemble_mode'] = stats['ensemble_mode']
            prediction['iteration'] = stats['iteration']
            pred_keys = sorted(prediction.columns.tolist())

            prediction.to_csv(args.output_predictions, mode='a', columns=pred_keys, header=(i == 1), index=False)

        all_stats.append(stats)

    alldf = pd.DataFrame.from_records(all_stats)

    print(alldf.loc[0, ['estimator', 'problem']])
    print('true_pairs', alldf['true_pairs'].mean())
    print('pruned_domain', alldf['pruned_domain'].mean())
    print('pruned_ratio', alldf['pruned_ratio'].mean())

    # figures.learning_curves(protocol, problem, filename=file_prefix + '_learning_curves')
    # figures.prediction_scatter_plot(prediction, problem, filename=file_prefix + '_pred_scatter')


def training_labels(y_train, args, scaling_factors=None):
    assert scaling_factors is None or scaling_factors.shape[1] == 2

    if args.adjustment == 0:
        # No label shift
        if args.outputs == 1:
            y_train_values = y_train['optimum'].values
        elif args.outputs == 2:
            y_train_values = np.array((y_train['optimum'].values, y_train['optimum'].values)).T
    else:
        # Label shift with lambda = args.adjustment
        lower_range = y_train['optimum'] - y_train['dom_lower']
        upper_range = y_train['dom_upper'] - y_train['optimum']

        if args.outputs == 1:
            if problem.minmax == 'min':
                y_train_values = (y_train['optimum'] + args.adjustment * upper_range).values.astype(int)
            else:
                y_train_values = (y_train['optimum'] - args.adjustment * lower_range).values.astype(int)
        elif args.outputs == 2:
            y_train_values = np.array((y_train['optimum'] - args.adjustment * lower_range,
                                       y_train['optimum'] + args.adjustment * upper_range), dtype=int).T

    if scaling_factors is not None:
        denominator = (scaling_factors['dom_upper'] - scaling_factors['dom_lower']).values.reshape(-1, 1)
        denominator = np.repeat(denominator, args.outputs, axis=1)
        subtrahend = np.repeat(scaling_factors['dom_lower'].values.reshape(-1, 1), args.outputs, axis=1)
        y_train_values = (y_train_values - subtrahend) / denominator

    return y_train_values


def preprocess_features(X_train, X_test, args):
    varthreshold = VarianceThreshold()  # removes all zero-variance features
    X_train = varthreshold.fit_transform(X_train)
    X_test = varthreshold.transform(X_test)

    # preprocessors = [varthreshold]
    # preprocessors = [SelectKBest(score_func=f_regression)]

    if args.estimator not in ['gtb', 'gtba', 'forest']:
        if args.estimator == 'knn':
            scaler = MinMaxScaler()
        elif args.estimator in ['network', 'networka']:
            scaler = MinMaxScaler((-1, 1))
        else:
            scaler = StandardScaler()  # Standardize features by removing the mean and scaling to unit variance

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test


def prepare_solver_validation(solver, file_prefix, prediction, problem):
    solvers = config.SOLVER.values() if solver == 'all' else [config.SOLVER[solver]]
    configurations = [[] for _ in solvers]
    mzn_path = os.path.join(problem.basedir, problem.name, problem.mzn)

    for dzn, obj_bound, _ in prediction:
        dzn_path = os.path.join(problem.basedir, problem.name, dzn)

        if problem.minmax == 'min':
            lower_bound = obj_bound
            upper_bound = None
        else:
            lower_bound = None
            upper_bound = obj_bound

        for solverid, s in enumerate(solvers):
            conf = config.ExecConfig(solver=s, problem=problem, mzn_path=mzn_path, dzn=dzn,
                                     dzn_path=dzn_path, lower_bound=lower_bound, upper_bound=upper_bound,
                                     timeout=config.TIMEOUT,
                                     dataset='full')
            configurations[solverid].append(conf)

    # Then run the solvers
    for solver, configs in zip(solvers, configurations):
        csv_filename = file_prefix + '_%s_estimation.csv' % solver.name

        if os.path.isfile(csv_filename):
            logging.warning('Skip %s as it already exists: Renaming it...', csv_filename)
            os.rename(csv_filename, csv_filename + '.bak')

        # run_configurations(configs, csv_filename)
        export_configurations(configs, csv_filename, file_prefix + '_run.sh')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('problem_name')
    parser.add_argument('-est', '--estimator',
                        choices=['network', 'networka', 'knn', 'forest', 'gp', 'bayridge', 'ridge', 'svm', 'linear',
                                 'xgb', 'xgba'],
                        default='network')
    parser.add_argument('-o', '--outputs', type=int, choices=[1, 2], default=1)
    parser.add_argument('-e', '--ensemble', type=int, default=1)
    parser.add_argument('-em', '--ensemble-mode', choices=['extreme', 'average', 'median', 'leaveoneout'],
                        default='average')
    parser.add_argument('-i', '--iter', type=int, default=1)
    parser.add_argument('-v', "--verbose", help="increase output verbosity",
                        action="store_true")
    parser.add_argument('-af', '--adjustment', type=float, default=0)
    parser.add_argument('-l', '--loss-fn', choices=['mse', 'mae', 'linex', 'shiftedmse', 'peann'], default='shiftedmse')
    parser.add_argument('-lf', '--loss-factor', type=float, default=0.8,
                        help='Asymmetry factor (only for linex/shiftedmse/peann')
    parser.add_argument('-f', '--force-new', action='store_true', default=False, help='Overwrites any existing model')
    parser.add_argument('--grid-search', action='store_true')
    parser.add_argument('--file-suffix', default=None)
    parser.add_argument('-val', '--validate', action='store_true', default=False)
    parser.add_argument('-s', '--solver', choices=['all'] + list(config.SOLVER.keys()),
                        default=None,
                        help='If given, these solvers are used for validation of estimated boundaries.')
    parser.add_argument('--validation-file', default=None,
                        help='File to store the validation commands, if a solver is to be used.')
    parser.add_argument('-of', '--output-stats', help='File to store statistics about the trained estimator.')
    parser.add_argument('-op', '--output-predictions', help='File to store predictions on validation instances.')
    parser.add_argument('--smallfeat', action='store_true', default=False)
    parser.add_argument('-scaled', '--scaled-prediction', action='store_true', default=False,
                        help="Predict the objective in relation to the instance's domain bounds and scale back to ints")
    parser.add_argument('-use_firstsol', action='store_true', default=False)

    args = parser.parse_args()

    if args.estimator == 'networka' and args.loss_fn not in ('linex', 'shiftedmse', 'peann'):
        args.loss_fn = 'shiftedmse'
    elif args.estimator == 'network' and args.loss_fn not in ('mse', 'mae'):
        args.loss_fn = 'mse'

    try:
        problem = next(p for p in config.PROBLEMS if args.problem_name == p.name)
    except StopIteration:
        logging.error('Problem %s does not exist in problem list' % args.problem_name)
        sys.exit(1)

    if args.loss_fn != parser.get_default('loss_fn') and args.estimator not in ['network', 'linear', 'xgba']:
        logging.warning('Loss function is only relevant for estimators network, xgba, and linear')

    args.problem = problem
    args.minmax = problem.minmax

    main(args)
