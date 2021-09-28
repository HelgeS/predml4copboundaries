import argparse
import itertools
import logging
import os
import random
import sys

import numpy as np
import pandas as pd

import config
import data_loader
from data_loader import get_best_result
from solve.helper import export_configurations

parser = argparse.ArgumentParser()
parser.add_argument('problem')
parser.add_argument('-s', '--solver', nargs='+', choices=['all'] + list(config.SOLVER.keys()),
                    default=['sicstus'])
parser.add_argument('-t', '--timeout', type=int, default=config.TIMEOUT)
parser.add_argument('-lb', '--lower_boundaries', nargs='+', default=[None, 1.0, 0.98, 0.95, 0.9, 0.8, 0.5])
parser.add_argument('-ub', '--upper_boundaries', nargs='+', default=[None, 1.0, 1.02, 1.05, 1.1, 1.2, 1.5])
parser.add_argument('-n', '--instances', type=int, default=10)
parser.add_argument('--cluster', action='store_true', default=False,
                    help='Store commands in instance format for cluster execution (abel)')
parser.add_argument('-v', "--verbose", help="increase output verbosity",
                    action="store_true")
parser.add_argument('--instance-filter', default=None)
parser.add_argument('--select-best', action="store_true", default=False,
                    help='Select instances with the best estimations, otherwise random')
parser.add_argument('--predictions', nargs='+', default=None)
args = parser.parse_args()

solvers = config.SOLVER.values() if 'all' in args.solver else [config.SOLVER[key] for key in args.solver]

try:
    problem = next(p for p in config.PROBLEMS if args.problem == p.name)
except StopIteration:
    logging.error('Problem %s does not exist in problem list' % args.problem.name)
    sys.exit(1)

result_filename = 'run_motivation_{}.sh'.format(problem.name)

if args.predictions:
    pred_cache = 'logs/{}_median_estimations.p'.format(problem.name)

    if not os.path.isfile(pred_cache):
        filter_fn = lambda df: pd.DataFrame(df.loc[(df.estimator == 'networka') & (df.loss == 'shiftedmse') & (df.adjustment == 0.1)])
        preds = data_loader.load_predict_log(args.predictions, problem.name, filter_fn)
        preds = preds.groupby(['problem', 'estimator', 'adjustment', 'loss', 'outputs', 'dzn'], as_index=False).median()
        preds.to_pickle(pred_cache)
    else:
        preds = pd.read_pickle(pred_cache)

configurations = []
mzn_path = os.path.join(problem.basedir, problem.name, problem.mzn)
avail_dzns = problem.get_dzns()

if args.instance_filter:
    feasible_instances = open(args.instance_filter).read().splitlines()
    dzns = [d for d in avail_dzns if d[1] in feasible_instances]

    if 0 < args.instances < len(dzns):
        dzns = random.sample(dzns, args.instances)
elif args.predictions:
    preds = preds[((preds['underest'] <= preds['optimum']) & (preds['optimum'] <= preds['overest']))]
    dzns = random.sample([d for d in avail_dzns if d[0] in preds.dzn.tolist()], args.instances)
elif 0 < args.instances < len(avail_dzns):
    dzns = random.sample(avail_dzns, args.instances)
else:
    dzns = avail_dzns

dzns.sort(key=lambda x: x[1])

if args.predictions:
    dzn_names = [dzn_name for dzn_name, _ in dzns]
    preds = preds[preds.dzn.isin(dzn_names)]

for i, (dzn_name, dzn_path) in enumerate(dzns):
    try:
        best_result = get_best_result(dzn_path)
    except ValueError as e:
        logging.warning('No boundary found: %s: %s: %s' % (problem.name, dzn_name, e))
        continue

    boundaries = []

    if args.predictions and not args.select_best:
        for lb, ub in preds[preds.dzn == dzn_name][['underest', 'overest']].values:
            if np.isnan(lb):
                lb = None
            else:
                lb = int(np.floor(lb))

            if np.isnan(ub):
                ub = None
            else:
                ub = int(np.ceil(ub))

            boundaries.append((lb, ub, lb, ub))
    else:
        for lb, ub in itertools.product(args.lower_boundaries, args.upper_boundaries):
            if lb == 'None':
                lb = None
            if ub == 'None':
                ub = None

            if lb:
                lbv = np.floor(lb * best_result).astype(int)
            else:
                lbv = None

            if ub:
                ubv = np.ceil(ub * best_result).astype(int)
            else:
                ubv = None

            if (lbv, ubv) in boundaries:
                continue

            boundaries.append((lb, ub, lbv, ubv))

    for lb, ub, lbv, ubv in boundaries:
        for s in solvers:
            ds = '({},{})'.format('%.2f' % lb if lb is not None else 'None',
                                  '%.2f' % ub if ub is not None else 'None')
            conf = config.ExecConfig(solver=s, problem=problem, mzn_path=mzn_path, dzn=dzn_name,
                                     dzn_path=dzn_path, lower_bound_value=lbv, upper_bound_value=ubv,
                                     lower_bound=lb, upper_bound=ub,
                                     timeout=args.timeout, dataset=ds)
            configurations.append(conf)

if args.cluster:
    with open(result_filename, 'w') as writer:
        for conf in configurations:
            if conf.has_lower_bound:
                if conf.solver.name == 'sunnycp':
                    bound_param = '-l'
                else:
                    bound_param = '-lb'

                lbarg = '{} {:d}'.format(bound_param, conf.lower_bound_value)
            else:
                lbarg = ''

            if conf.has_upper_bound:
                if conf.solver.name == 'sunnycp':
                    bound_param = '-u'
                else:
                    bound_param = '-ub'

                ubarg = '{} {:d}'.format(bound_param, conf.upper_bound_value)
            else:
                ubarg = ''

            script_line = '{solver};{mzn};{dzn};{lbarg};{ubarg};{timeout:d};{addargs};{comment}'.format(
                solver=conf.solver.name,
                mzn=conf.mzn_path,
                dzn=conf.dzn_path,
                lbarg=lbarg,
                ubarg=ubarg,
                timeout=conf.timeout,
                addargs='-var {}'.format(conf.problem.objective_var),
                comment=conf.dataset)

            print(script_line, file=writer)
else:
    csv_filename = 'motivation_{}.csv'.format(problem.name)
    # run_configurations(configurations, csv_filename)
    export_configurations(configurations, csv_filename, result_filename)
