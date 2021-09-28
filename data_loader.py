import glob
import json
import logging
import multiprocessing
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import config
import dzn_eval

logger = logging.getLogger('combination-learning')


def flatten(lst):
    flattened = []

    if not isinstance(lst, (list, tuple, set)):
        return [lst]

    for sublist in lst:
        if isinstance(sublist, (list, tuple, set)):
            flattened.extend(flatten(sublist))
        elif isinstance(sublist, dict):
            flattened.extend(flatten(sorted(sublist.items(), key=lambda x: x[0])))
        else:
            flattened.append(sublist)

    return flattened


def dzn_input_string(dznfile, delim='#'):
    #dzn = pymzn.dzn_eval(dznfile)
    dzn = dzn_eval.dzn_eval(dznfile)
    sorted_values = []

    for (_, v) in sorted(dzn.items(), key=lambda x: x[0]):
        sorted_values.extend(flatten(v))

        if delim:
            sorted_values.append(delim)

    if delim:
        del sorted_values[-1]

    return sorted_values


def join_result_csvs(csv_dir, keyword):
    sub_dfs = []

    for f in glob.glob(os.path.join(csv_dir, keyword)):
        try:
            pdf = pd.read_csv(f, sep=';', na_values='None', dtype={'Failed': bool})

            if len(pdf) == 0:
                print(f, 'is empty')
                continue

            if 'HasBound' not in pdf.columns:
                pdf['HasBound'] = False
                pdf['ObjBound'] = np.nan

            sub_dfs.append(pdf)
        except Exception as e:
            print('File %s: %s' % (f, e))

    if len(sub_dfs) > 1:
        df = pd.concat(sub_dfs, axis=0)
    else:
        df = sub_dfs[0]

    df['Failed'] = df['Failed'].astype('bool')
    df.loc[~df.Complete, 'Duration'] = df[~df.Complete]['Duration'].astype('int')
    return df


def get_baseline(problem=None, include_failed=False):
    csv_dir = config.OUTPUT_DIR
    baseline = join_result_csvs(csv_dir, 'baseline_*.csv')

    # This is mostly for debug purposes, but might help for actual evaluations
    # We limit the considered solvers and problems
    if problem:
        problem_names = [problem.name]
    else:
        problem_names = [p.name for p in config.PROBLEMS]

    #rel_solvers = ['chuffed', 'cbc', 'gecode']
    #baseline = baseline[(baseline.Problem.isin(problem_names) & (baseline.Solver.isin(rel_solvers)))]
    baseline = baseline[(baseline.Problem.isin(problem_names))]

    if not include_failed:
        baseline = baseline[~baseline.Failed]

    return baseline


def generate_benchmark_sample(sample_size=20, max_duration=180):
    fn = lambda x: x.iloc[np.random.choice(range(0, len(x)), min(sample_size, len(x)), replace=False)]

    df = get_baseline()
    df = df[['Problem', 'DZN', 'Duration']].groupby(['Problem', 'DZN'], as_index=False).max()
    df = df[df.Duration < max_duration][['Problem', 'DZN']].drop_duplicates()
    df = df.groupby('Problem', as_index=False).apply(fn)
    return df


def missing_optima(problem_set=config.PROBLEMS):
    """
    Lists all problems and their instances where the optimum result has not been found.
    For each problem instance the previous best result (from the baseline results) is given in column `Best`.    
    :return: pd.DataFrame with columns ['Problem', 'DZN', 'Best'] 
    """
    missing_optima = []
    baseline = get_baseline()

    for problem in problem_set:
        for dzn, dzn_path in problem.get_dzns():
            opt_path = dzn_path + '.opt'
            if not os.path.isfile(opt_path):
                if problem.minmax == 'min':
                    prev_best = baseline[(baseline.Problem == problem.name) & (baseline.DZN == dzn)]['Objective'].min()
                else:
                    prev_best = baseline[(baseline.Problem == problem.name) & (baseline.DZN == dzn)]['Objective'].max()
                missing_optima.append((problem.name, dzn, prev_best))

    return pd.DataFrame(missing_optima, columns=['Problem', 'DZN', 'Best'])


def missing_results(problem_set=config.PROBLEMS):
    """ 
    Returns a DataFrame with columns ['Problem', 'DZN'], listing all problems and their instances without any result.
    Considers opt result files and the existing baseline results.    
    """
    missing_results = missing_optima(problem_set)

    return missing_results[missing_results.Best.isnull()][['Problem', 'DZN']]


def get_best_results(problem_set=config.PROBLEMS, dzn_filter=None):
    rows = []

    for problem in problem_set:
        for dzn, dzn_path in problem.get_dzns(dzn_filter=dzn_filter):
            opt, stat = get_best_result(dzn_path, include_status=True)
            rows.append((problem.name, problem.mzn, dzn, opt, stat))

    return pd.DataFrame(rows, columns=['problem', 'mzn', 'dzn', 'objective', 'status'])


def get_best_result(dzn_path, optimum_only=False, include_status=False):
    opt_path = dzn_path + '.opt'
    best_path = dzn_path + '.best'

    if os.path.isfile(opt_path):
        raw_optimum = open(opt_path, 'r').read()
        status = 'COMPLETE'
    elif not optimum_only and os.path.exists(best_path):
        raw_optimum = open(best_path, 'r').read()
        status = 'SOLFOUND'
    else:
        logging.warning('No result for %s' % dzn_path)
        raw_optimum = 0
        status = 'UNKNOWN'

    if include_status:
        return int(raw_optimum), status
    else:
        return int(raw_optimum)


def split_train_test_instances(problem, output_file, nb_datasets):
    dzns = [dzn for dzn, path in problem.get_dzns()]
    datasets = []

    kf = KFold(n_splits=nb_datasets, shuffle=True)

    for train, test in kf.split(dzns):
        ds = {
            'train': [dzns[x] for x in train],
            'test': [dzns[x] for x in test]
        }
        datasets.append(ds)

    pickle.dump(datasets, open(output_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    return datasets


def load_stats_log(files, problem_filter=None):
    if isinstance(files, str):
        files = [files]

    files2 = []

    for f in files:
        files2.extend(glob.glob(f))

    partial_dfs = []

    for f in files2:
        if open(f, 'r').read(1) == "{":
            # JSON-based format
            records = []
            for x in open(f, 'r'):
                if problem_filter and '"{}"'.format(problem_filter) not in x:
                    break

                records.append(json.loads(x))

            partial_dfs.append(pd.DataFrame.from_records(records))
        else:
            # CSV format
            try:
                partial_dfs.append(pd.read_csv(f))
            except:
                print("Failed to read file: ", f)

    return pd.concat(partial_dfs)


def load_predict_log(files, problem_filter=None, filter_fn=None):
    if isinstance(files, str):
        files = [files]

    files2 = []

    for f in files:
        files2.extend(glob.glob(f))

    dfs = []

    with multiprocessing.Pool() as p:
        for df in p.imap(load_predict_log_file, files2):
            if problem_filter:
                df = df[df.problem == problem_filter]

            if filter_fn:
                df = filter_fn(df)

            dfs.append(df)

    return pd.concat(dfs)


def load_predict_log_file(filepath):
    if open(filepath, 'r').read(1) == "{":
        records = []

        for x in open(filepath, 'r'):
            try:
                rec = pd.read_json(x.strip())
            except:
                rec = pd.Series(json.loads(x.strip()))

            records.append(rec)

        if len(records) == 0:
            return None

        df = pd.concat(records)
    else:
        df = pd.read_csv(filepath)

    return df
