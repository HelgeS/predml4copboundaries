import ast
import collections
import hashlib
import itertools
import multiprocessing
import operator
import os
import pickle
import subprocess

import numpy as np
import pandas as pd
import pymzn
import scipy.stats
from statsmodels import robust

from data_loader import get_best_result


def cached_feature_matrix(problem, dzns=None, include_opt=False, include_mzn2feat=False, include_labels=False):
    if dzns:
        list_hash = hashlib.sha256(repr(tuple(sorted(dzns))).encode('utf-8')).hexdigest()
    else:
        list_hash = 'all'

    filename = '{}_{}'.format(problem.name, list_hash)
    filename += '_mzn2feat' if include_mzn2feat else ''
    filename += '_opt' if include_opt else '_noopt'
    filename += '_labeled' if include_labels else ''
    filename += '.p'

    filepath = os.path.join('data', filename)

    if os.path.isfile(filepath):
        X, y = pickle.load(open(filepath, 'rb'))
    else:
        X, y = feature_matrix(problem, dzns=dzns, include_opt=include_opt, include_mzn2feat=include_mzn2feat,
                              include_labels=include_labels)
        pickle.dump((X, y), open(filepath, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

    return X, y


def feature_matrix(problem, dzns=None, include_opt=False, include_mzn2feat=False, include_labels=False):
    feature_vectors = []
    nb_vars = []
    y = []

    if dzns and len(dzns) > 0:
        dzn_tuples = []
        all_dzns = problem.get_dzns()

        for dn in dzns:
            dzn_tuples.append(next(t for t in all_dzns if t[0] == dn))
    else:
        dzn_tuples = problem.get_dzns()

    conf_gen = ([n, p, include_labels, include_mzn2feat, include_opt, problem] for n, p in dzn_tuples)

    with multiprocessing.Pool() as p:
        for feats, result in p.starmap(feature_result_pair, conf_gen):
            feature_vectors.append(feats)
            nb_vars.append(len(feats))
            y.append(result)

    if any(x != nb_vars[0] for x in nb_vars[1:]):
        raise Exception('Varying number of variables!')

    X = np.array(feature_vectors)

    if include_opt:
        return X, pd.DataFrame.from_records(y, index='dzn')
    else:
        return X, None


def feature_result_pair(dzn_name, dzn_path, include_labels, include_mzn2feat, include_opt, problem):
    features = feature_vector(dzn_path, include_labels=include_labels)
    result = None

    if include_mzn2feat or include_opt:
        m2f_dict = mzn2feat(problem, problem.mzn_path, dzn_path)

        if include_mzn2feat:
            vals = sorted(m2f_dict.items(), key=operator.itemgetter(0))

            if include_labels:
                m2f = np.array([[(v, k) for (k, v) in vals]])
            else:
                m2f = [v for (k, v) in vals]

            features = np.hstack((features, m2f))

        if include_opt:
            lower_bound = m2f_dict['o_dom_lower']
            upper_bound = m2f_dict['o_dom_upper']
            opt = get_best_result(dzn_path)
            result = {'problem': problem.name, 'dzn': dzn_name, 'dom_lower': lower_bound, 'dom_upper': upper_bound,
                      'optimum': opt}

    return features, result


def feature_vector(dzn_path, include_labels=False):
    vars_in = pymzn.dzn2dict(dzn_path) #, ignore_array_dimensions=True)

    features = []
    types = []

    for varname, cont in sorted(vars_in.items(), key=lambda k: k[0]):
        if isinstance(cont, (tuple, list, set, dict)):
            lfeat = list_features(list(cont))
            features.extend(lfeat)
            types.extend(['{}/{}'.format(s, varname) for s in
                          ['len', 'mean', 'median', 'std', 'iqr', 'min', 'max', 'skew', 'kurtosis']])
        elif is_number(cont):
            features.append(cont)
            types.append('number/{}'.format(varname))
        else:
            raise Exception('Incompatible data type: ', cont)

    rounded_mat = np.array(features).round(4)

    if include_labels:
        annot_mat = np.array(types)
        assert (annot_mat.shape == rounded_mat.shape)
        rounded_mat = np.dstack((rounded_mat, annot_mat))

    return rounded_mat


def instance_vector(dzn_path):
    vars_in = pymzn.dzn2dict(dzn_path)
    vector = list(flatten(([cont] for _, cont in sorted(vars_in.items(), key=lambda k: k[0]))))

    return vector


def list_features(values):
    if len(values) == 0:
        return [0] * 9  #13

    if all(is_number(x) for x in values):
        #lfeat = [len(values), np.mean(values), np.median(values), np.std(values), scipy.stats.iqr(values),
        #         np.min(values), np.max(values), np.percentile(values, q=25), np.percentile(values, q=75),
        #         np.ptp(values), robust.scale.mad(values), scipy.stats.skew(values),
        #         scipy.stats.kurtosis(values)]  # 13

        lfeat = [len(values), np.mean(values), np.median(values), np.std(values), scipy.stats.iqr(values),
                 np.min(values), np.max(values), scipy.stats.skew(values), scipy.stats.kurtosis(values)]  # 9

        return lfeat

    if isinstance(values[0], (tuple, list, set)):
        subfeatures = []

        for sublist in values:
            if all(is_number(x) for x in sublist):
                subfeatures.append(list_features(list(sublist)))
            else:
                subfeatures.append(len(sublist))

        return np.array(subfeatures).sum(axis=0)

    if isinstance(values[0], dict):
        return np.mean([len(v) for v in values])

    raise Exception('Incompatible data type: ', values)


def feature_names(problem, include_mzn2feat=True):
    dzn_path = problem.get_dzns()[0][1]
    features, _ = feature_result_pair(dzn_path, dzn_path, include_labels=True, include_mzn2feat=include_mzn2feat,
                                      include_opt=False, problem=problem)
    return features[0, :, 1]


def mzn2feat(problem, dzn_name, dzn_path):
    mzn2feat_path = '/home/helge/Sandbox/mzn2feat/bin/mzn2feat'

    cmd = [mzn2feat_path, '-i', problem.mzn_path, '-d', dzn_path, '-o', 'dict']
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e)
        print(e.output)
        raise

    output = str(output, 'utf-8').splitlines()[-1]

    output = output.replace(' -nan', ' 0')
    output = output.replace(' nan', ' 0')

    feature_dict = ast.literal_eval(output)
    # feature_dict['problem'] = problem.name
    # feature_dict['dzn'] = dzn_name

    # Exclude search-related features + global constraints information
    #feature_dict = {k: v for k, v in feature_dict.items() if not k.startswith('s_') and not k.startswith('gc_')}
    feature_dict = {k: v for k, v in feature_dict.items()}

    return feature_dict


def get_propagated_bounds_and_optimum(problem, dzns=None):
    pool = multiprocessing.Pool(4)

    results = pool.map(get_single_propagated_bounds_and_optimum, itertools.product([problem], problem.get_dzns(dzns)))

    return pd.DataFrame.from_records(results)


def get_single_propagated_bounds_and_optimum(param):
    problem, dzn_info = param
    dzn_name, dzn_path = dzn_info
    features = mzn2feat(problem, dzn_name, dzn_path)
    lower_bound = features['o_dom_lower']
    upper_bound = features['o_dom_upper']
    opt = get_best_result(dzn_path)
    return {'problem': problem.name, 'dzn': dzn_name, 'lower': lower_bound, 'upper': upper_bound,
            'optimum': opt}


def is_number(x):
    try:
        float(x)
        return True
    except:
        return False


def flatten(l):
    """Recursively flatten a list of irregular lists.

    Taken from: https://stackoverflow.com/questions/2158395/flatten-an-irregular-list-of-lists"""
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el