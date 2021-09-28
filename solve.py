import argparse
import ast
import json
import logging
import os
import pymzn
import sys

import config
import solve.solvers

parser = argparse.ArgumentParser()
parser.add_argument('problem', help='Problem name or path to minizinc model')
parser.add_argument('dzn', type=argparse.FileType('r'))
parser.add_argument('-lb', '--lowerbound', type=int, default=None)
parser.add_argument('-ub', '--upperbound', type=int, default=None)
parser.add_argument('-s', '--solver', choices=['all'] + list(config.SOLVER.keys()),
                    default='chuffed')
parser.add_argument('-t', '--timeout', type=int, default=config.TIMEOUT)
parser.add_argument('-f', '--free', action='store_true', default=False, help='Free search (if supported by solver)')
parser.add_argument('--no-timeout', action='store_true', default=False)
parser.add_argument('-ds', '--dataset', default=None)
parser.add_argument('--output', default=None)
parser.add_argument('--pymzn', action='store_true', default=False)
args = parser.parse_args()

if os.path.isfile(args.problem):
    mzn_path = args.problem
    problem = config.Problem()
else:
    try:
        problem = next(p for p in config.MZNC1617_PROBLEMS if args.problem == p.name)
        mzn_path = problem.mzn_path
    except StopIteration:
        logging.error('Problem %s does not exist in problem list and is not an existing file' % args.problem)
        sys.exit(1)

sel_solvers = config.SOLVER.values() if args.solver == 'all' else [args.solver]
timeout = None if args.no_timeout or args.timeout == 0 else args.timeout

if args.pymzn or len(sel_solvers) > 1:
    pymzn.solve(args.dzn, args.lowerbound, args.upperbound, timeout, args.dataset, sel_solvers)
    sys.exit(0)
elif args.solver not in ['chuffed', 'sicstus', 'ortools']:
    logging.error('Solver %s not supported. Try parameter --pymzn' % args.solver)
    sys.exit(1)

try:
    dataset = ast.literal_eval(args.dataset)
except:
    dataset = args.dataset

if args.solver == 'chuffed':
    solver = solve.solvers.Chuffed()
elif args.solver == 'sicstus':
    solver = solve.solvers.Sicstus()
elif args.solver == 'ortools':
    solver = solve.solvers.OrTools()

stats = solver.solve(problem, dzn=args.dzn.name, timeout=timeout, lb=args.lowerbound, ub=args.upperbound,
                     dataset=dataset, free_search=args.free)

if args.output:
    stats_json = json.dumps(stats)
    open(args.output, 'a').write(stats_json + os.linesep)
else:
    print(stats)
