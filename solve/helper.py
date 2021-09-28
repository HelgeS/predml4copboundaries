import json
import os
import random
import shutil
import tempfile
from multiprocessing.pool import Pool

import pandas as pd
from tqdm import tqdm

import config
from solve.pymzn import execute_config


def export_configurations(configurations, result_filename, script_filename):
    with open(script_filename, 'w') as writer:
        # print('Solver;Problem;DZN;Duration;Objective;Complete;Failed;HasBound;ObjBound;Dataset',
        #      file=open(result_filename, 'w'))

        for conf in configurations:
            script_line = 'python solve.py {} {} -s {} -t {:d} -ds "{}" --output {}'.format(conf.problem.name,
                                                                                            conf.dzn_path,
                                                                                            conf.solver.name,
                                                                                            conf.timeout,
                                                                                            conf.dataset,
                                                                                            result_filename)

            if conf.has_lower_bound:
                script_line += ' -lb {:d}'.format(conf.lower_bound_value)

            if conf.has_upper_bound:
                script_line += ' -ub {:d}'.format(conf.upper_bound_value)

            print(script_line, file=writer)


def run_configurations(configurations, result_filename):
    with open(result_filename, 'w') as reswriter:
        print('Solver;Problem;DZN;Duration;Objective;Complete;Failed;HasBound;ObjBound;Dataset', file=reswriter)

        if config.ALLOW_PARALLEL:
            random.shuffle(configurations)  # Poor student's load distribution

        with Pool(config.PARALLEL_POOL_SIZE if config.ALLOW_PARALLEL else 1) as parallel_pool:
            for res in tqdm(parallel_pool.imap_unordered(execute_config, configurations), total=len(configurations)):
                print(
                    '{};{};{};{:.2f};{};{};{};{};{};{}'.format(res.solver.name, res.problem.name, res.dzn, res.duration,
                                                               res.objective, res.is_complete, res.has_failed,
                                                               res.has_bound, res.obj_bound, res.dataset),
                    file=reswriter)
                reswriter.flush()


def post_constraints(problem, dzn_name, ub=None, lb=None):
    if lb and ub:
        assert (lb <= ub)

        if lb == ub:
            new_constraint = "constraint {obj} = {bound:d};\n".format(obj=problem.objective_var, bound=ub)
        else:
            new_constraint = "constraint {lb:d} <= {obj} /\ {obj} <= {ub:d};\n".format(lb=lb, ub=ub,
                                                                                       obj=problem.objective_var)
    elif ub:
        new_constraint = "constraint {obj} <= {ub:d};\n".format(ub=ub, obj=problem.objective_var)
    elif lb:
        new_constraint = "constraint {lb:d} <= {obj};\n".format(lb=lb, obj=problem.objective_var)
    else:
        new_constraint = ""

    mzn_path = problem.mzn_path
    mzn_base = os.path.basename(mzn_path)
    mzn_new = tempfile.NamedTemporaryFile(delete=False, prefix='{}-{}-'.format(mzn_base, dzn_name), suffix='.mzn').name

    shutil.copy(mzn_path, mzn_new)
    open(mzn_new, 'a').write(new_constraint)

    return mzn_new


def read_stats(files, group_multiples=True):
    if not isinstance(files, list):
        files = [open(files, 'r')]

    records = []

    for f in files:
        records.extend((json.loads(x) for x in f.readlines()))

    df = pd.DataFrame.from_records(records)

    df['LowerBound'].fillna(-1, inplace=True)
    df['UpperBound'].fillna(-1, inplace=True)
    df['LowerBoundValue'].fillna(-1, inplace=True)
    df['UpperBoundValue'].fillna(-1, inplace=True)

    int_columns = ['Backtracks', 'Conflicts', 'Constraints', 'Entailments', 'LowerBoundValue', 'UpperBoundValue',
                   'Variables', 'Timeout', 'SATBackjumps', 'SATVariables', 'Propagations', 'Propagators', 'Prunings',
                   'Resumptions', 'Runtime', 'Solutions', 'Objective']

    float_columns = ['Inittime', 'LowerBound', 'UpperBound', 'search_time']

    for ic in int_columns:
        if ic in df:
            df[ic].fillna(0, inplace=True)
            df[ic] = df[ic].astype(int)

    for fc in float_columns:
        if fc in df:
            df[fc].fillna(0, inplace=True)
            df[fc] = df[fc].astype(float)

    # If multiple measures exist, than take
    if group_multiples:
        groupcols = ['Solver', 'Problem', 'Instance', 'LowerBoundValue', 'UpperBoundValue', 'Timeout']
        aggdict = {key: 'min' if key != 'Solutions' else 'max' for key in df.columns if key not in groupcols}
        df = df.groupby(groupcols, as_index=False).agg(aggdict)

    return df
