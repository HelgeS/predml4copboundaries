import math
import os
import re
import sys
import numpy as np

import pandas as pd

STATUS = ('COMPLETE', 'SOLFOUND', 'UNSATISFIABLE', 'UNKNOWN', 'FAILED', 'INTERMEDIATE')

re_worker_memory = re.compile('^\s+JobID\s+MaxVMSize\s+MaxRSS\s*$\n' +
                              '^.*$\n^[\w\d.+]+\s+(?P<maxvmsize>[\d.\w]+)\s+(?P<maxrss>[\d.\w]+)\s*$',
                              re.MULTILINE)

re_log = re.compile(
    '(?P<jobid>\d+)-(?P<solver>\w+)-(?P<mzn>[\w.\-_\d]+\.mzn)-(?P<dzn>[\w.\-_\d]+\.dzn)-(?P<taskid>\d+)(-(?P<boundsmode>\w+))?.log')

re_choco_sol = re.compile(
    '^% Model\[.*\], (?P<solutions>\d+) Solutions, ' +
    '(MINIMIZE|MAXIMIZE) [\w_]+ = (?P<objective>\d+), ' +
    'Resolution time (?P<restime>[\d.]+)s, (?P<nodes>\d+) Nodes \([.,\d]+ n/s\), ' +
    '(?P<backtracks>\d+) Backtracks, (?P<fails>\d+) Fails, (?P<restarts>\d+) Restarts')
re_choco_final = re.compile('^% Model\[.*\], (?P<vars>\d+) variables, (?P<constr>\d+) constraints, ' +
                            'building time: (?P<buildtime>[\d.]+)s, w/o user-defined search strategy, ' +
                            'w/o complementary search strategy')
re_chuffed_stats = re.compile('^-?\d+,\d+,\d+,\d+,\d+,\d+,\d+,\d+,[\d.]+,[\d.]+')
re_gecode_ortools_stats = re.compile('^%%\s+(?P<key>[\w\s]+):\s+(?P<value>[\w\d. \(\)]+)')
re_gecode_obj = re.compile('^(?:objective|t_end) = (?P<objective>\d+);$')

re_solution_time = re.compile('^% time elapsed: (?P<time>\d+) ms')
re_objective = re.compile('^(?:objective|makespan|t_end)\s?=\s?(?P<objective>\d+);')


def process_logfile(f, logdir, instances):
    m = re_log.match(f)

    if not m:
        print('Filename {} does not match regex'.format(f))

    solver_log_file = os.path.join(logdir, f)
    worker_log_file = os.path.join(logdir, '{}-worker.out'.format(m.group('jobid')))

    if not os.path.exists(worker_log_file):
        print('Worker log {} does not exist'.format(worker_log_file))

    worker_log = open(worker_log_file).read()

    if 'completed' not in worker_log:
        raise Exception('{}: Job seems to have failed...'.format(worker_log_file))

    worker_match = re_worker_memory.search(worker_log)

    if not worker_match:
        raise Exception('{}: Could not find memory statistics'.format(worker_log_file))

    output = open(solver_log_file).read()

    if 'aborted' in output or 'failure stack trace' in output:
        raise Exception('{}: Solver seems to have failed...'.format(solver_log_file))

    solver_lines = output.splitlines()

    solver = m.group('solver')
    taskid = int(m.group('taskid'))

    assert (solver == instances.ix[taskid - 1]['solver'])

    lbarg = instances.ix[taskid - 1]['lbarg']
    ubarg = instances.ix[taskid - 1]['ubarg']

    if isinstance(lbarg, str) and lbarg.startswith('-lb '):
        lb = int(lbarg.replace('-lb ', ''))
    else:
        lb = None

    if isinstance(ubarg, str) and ubarg.startswith('-ub '):
        ub = int(ubarg.replace('-ub ', ''))
    else:
        ub = None

    boundstype = 'hard'

    if 'boundsmode' in m.groupdict():
        if m.group('boundsmode') == 'NoBounds':
            bounds = 'No'
            ub = lb = None
            ubarg = lbarg = None
        elif m.group('boundsmode') == 'UpperBound':
            bounds = 'Upper'
            lb = None
            lbarg = None
        elif m.group('boundsmode') == 'LowerBound':
            bounds = 'Lower'
            ub = None
            ubarg = None
        elif m.group('boundsmode') == 'BothBounds':
            bounds = 'Both'
            assert lb is not None and ub is not None, "Not all bounds set, but marked as BothBounds"
        elif m.group('boundsmode') == 'HardBounds':
            boundstype = 'hard'
        elif m.group('boundsmode') == 'SoftBounds':
            boundstype = 'soft'

    if lb is None and ub is None:
        bounds = 'No'
    elif lb is None:
        bounds = 'Upper'
    elif ub is None:
        bounds = 'Lower'
    else:
        bounds = 'Both'

    timeout = instances.ix[taskid - 1]['timeout']

    stats = {
        'solver': solver,
        'mzn': m.group('mzn'),
        'dzn': m.group('dzn'),
        'lowerbound': -1 if lb is None else lb,
        'upperbound': -1 if ub is None else ub,
        'lowerboundvalue': -1 if lb is None else lb,
        'upperboundvalue': -1 if ub is None else ub,
        'timeout': timeout,
        'taskid': taskid,
        'maxvmsize': convert_memory(worker_match.group('maxvmsize')),
        'maxrss': convert_memory(worker_match.group('maxrss')),
        'bounds': bounds,
        'boundstype': boundstype
    }

    for l in solver_lines:
        if l.startswith('Time_mzn2fzn: '):
            stats['time_mzn2fzn'] = math.ceil(float(l.replace('Time_mzn2fzn: ', '')))
        elif l.startswith('Time_solver: '):
            stats['time_solver'] = math.ceil(float(l.replace('Time_solver: ', '')))

    if '###DONE###' not in solver_lines[-4:]:
        print(f, 'Solver output not complete, but necessary info might be included')
    assert 'time_mzn2fzn' in stats, 'time_mzn2fzn missing'  # Preparation time
    assert 'time_solver' in stats, 'time_solver missing'  # Runtime

    stats['time_solver'] = min(stats['time_solver'], stats['timeout'])

    solutions = parse_solutions(solver_lines, stats)

    try:
        if solver == 'choco':
            stats = parse_choco(stats, solver_lines)
        elif solver == 'chuffed':
            stats = parse_chuffed(stats, solver_lines)
        elif solver == 'gecode':
            stats = parse_gecode(stats, solver_lines)
        elif solver == 'ortools':
            stats = parse_ortools(stats, solver_lines)
        elif solver == 'sunnycp':
            stats = parse_sunnycp(stats, solver_lines)
        else:
            raise NotImplementedError('{}: Solver {} not supported'.format(f, solver))

        assert (all('solutions' in s for s in stats))
    except Exception as e:
        print(solver_log_file, e)
        raise

    return stats, solutions


def parse_solutions(lines, static_fields):
    """
    Iterate through log and parse time when solutions are found and their objective value.
    Solver-independent as the time is issued by `solns2out` and the objective is part of the flatzinc output.
    """
    solution_times = []
    solution_objectives = []
    has_optimum = False

    for l in lines:
        m_time = re_solution_time.match(l)
        m_obj = re_objective.match(l)

        if m_time:
            time_found = int(m_time.group('time')) / 1000  # Convert ms to s
            solution_times.append(time_found)
        elif m_obj:
            obj = int(m_obj.group('objective'))
            solution_objectives.append(obj)
        elif l == '========':
            # Optimal solution found
            has_optimum = True

    if len(solution_times) != len(solution_objectives):
        raise Exception("Different number of times and objectives")

    optimal_sol = [False] * len(solution_times)

    if len(optimal_sol) > 0:
        optimal_sol[-1] = has_optimum

    return pd.DataFrame({'time': solution_times, 'objective': solution_objectives, 'optimum': optimal_sol, **static_fields})


def parse_choco(stats, lines):
    solutions = []

    for l in lines:
        m1 = re_choco_sol.match(l)

        if m1:
            solutions.append(m1.groupdict())
        else:
            m2 = re_choco_final.match(l)

            if m2:  # This is the final statistics line and should only appear once
                stats.update(m2.groupdict())

        m_first_sol = re_solution_time.match(l)

        if m_first_sol and 'first_sol_time' not in stats:
            stats['first_sol_time'] = int(m_first_sol.group('time')) / 1000

        m_first_obj = re_objective.match(l)

        if m_first_obj and 'first_objective' not in stats:
            stats['first_objective'] = m_first_obj.group('objective')

    intermediates = []

    for s in solutions[:-1]:
        intermediate_stats = dict(stats)
        intermediate_stats.update(s)
        intermediate_stats['status'] = 'INTERMEDIATE'
        intermediates.append(intermediate_stats)

    if len(solutions) > 0:
        stats.update(solutions[-1])
        stats['solutions'] = int(stats['solutions'])
    else:
        stats['solutions'] = 0

    assert ((len(solutions) == 0 and stats['solutions'] == 0) or len(solutions) == stats['solutions'] + 1)
    assert (len(intermediates) == stats['solutions'])

    if '=====UNKNOWN=====' in lines:
        stats['status'] = 'UNKNOWN'
    elif '=====UNSATISFIABLE=====' in lines:
        stats['status'] = 'UNSATISFIABLE'
    elif '==========' in lines:
        stats['status'] = 'COMPLETE'
    elif stats['solutions'] > 0:
        stats['status'] = 'SOLFOUND'
    else:
        stats['status'] = 'UNKNOWN'

    return [stats] + intermediates


def parse_chuffed(stats, lines):
    fields = ['objective', 'variables', 'satvariables', 'propagators', 'conflicts', 'satbackjumps', 'propagations',
              'solutions', 'inittime', 'search_time']

    values = []

    for l in lines:
        if re_chuffed_stats.fullmatch(l):
            values = l.split(',')

        m_first_sol = re_solution_time.match(l)

        if m_first_sol and 'first_sol_time' not in stats:
            stats['first_sol_time'] = int(m_first_sol.group('time')) / 1000

        m_first_obj = re_objective.match(l)

        if m_first_obj and 'first_objective' not in stats:
            stats['first_objective'] = m_first_obj.group('objective')

    if len(values) == 0:
        stats['status'] = 'FAILED'
        return [stats]

    stats.update({f: v for f, v in zip(fields, values)})

    if stats['objective'] == '-1':
        del stats['objective']

    if '=====UNKNOWN=====' in lines:
        stats['status'] = 'UNKNOWN'
    elif '=====UNSATISFIABLE=====' in lines:
        stats['status'] = 'UNSATISFIABLE'
    elif '==========' in lines:
        stats['status'] = 'COMPLETE'
    elif int(stats['solutions']) > 0:
        stats['status'] = 'SOLFOUND'
    else:
        stats['status'] = 'UNKNOWN'

    return [stats]


def parse_gecode(stats, lines):
    stats['solutions'] = lines.count('----------')

    for l in lines:
        m = re_gecode_ortools_stats.fullmatch(l)

        if m:
            dictkey = m.group('key').strip().replace(' ', '_')
            stats[dictkey] = m.group('value').strip().replace(' ms', '')
            continue

        m = re_gecode_obj.fullmatch(l)

        if m:
            stats['objective'] = int(m.group('objective'))

        m_first_sol = re_solution_time.match(l)

        if m_first_sol and 'first_sol_time' not in stats:
            stats['first_sol_time'] = int(m_first_sol.group('time')) / 1000

        m_first_obj = re_objective.match(l)

        if m_first_obj and 'first_objective' not in stats:
            stats['first_objective'] = m_first_obj.group('objective')

    if '=====UNKNOWN=====' in lines:
        stats['status'] = 'UNKNOWN'
    elif '=====UNSATISFIABLE=====' in lines:
        stats['status'] = 'UNSATISFIABLE'
    elif '==========' in lines:
        stats['status'] = 'COMPLETE'
    elif int(stats['solutions']) > 0:
        stats['status'] = 'SOLFOUND'
    else:
        stats['status'] = 'UNKNOWN'

    return [stats]


def parse_ortools(stats, lines):
    stats['solutions'] = lines.count('----------')

    for l in lines:
        m = re_gecode_ortools_stats.fullmatch(l)

        if m:
            if m.group('key').strip() == 'max objective' or m.group('key').strip() == 'min objective':
                dictkey = 'objective'
                stats[dictkey] = int(m.group('value').strip().replace(' (proven)', ''))
            else:
                dictkey = m.group('key').strip().replace(' ', '_')
                stats[dictkey] = m.group('value').strip().replace(' ms', '')

        m_first_sol = re_solution_time.match(l)

        if m_first_sol and 'first_sol_time' not in stats:
            stats['first_sol_time'] = int(m_first_sol.group('time')) / 1000

        m_first_obj = re_objective.match(l)

        if m_first_obj and 'first_objective' not in stats:
            stats['first_objective'] = m_first_obj.group('objective')

    if '=====UNKNOWN=====' in lines:
        stats['status'] = 'UNKNOWN'
    elif '=====UNSATISFIABLE=====' in lines:
        stats['status'] = 'UNSATISFIABLE'
    elif '==========' in lines:
        stats['status'] = 'COMPLETE'
    elif int(stats['solutions']) > 0:
        stats['status'] = 'SOLFOUND'
    else:
        stats['status'] = 'UNKNOWN'

    return [stats]


def parse_sunnycp(stats, lines):
    stats['solutions'] = lines.count('----------')

    for l in lines:
        if l.startswith('% Current Best Bound: '):
            stats['objective'] = int(l.replace('% Current Best Bound: ', ''))

        m_first_sol = re_solution_time.match(l)

        if m_first_sol and 'first_sol_time' not in stats:
            stats['first_sol_time'] = int(m_first_sol.group('time')) / 1000

        m_first_obj = re_objective.match(l)

        if m_first_obj and 'first_objective' not in stats:
            stats['first_objective'] = m_first_obj.group('objective')

    if '=====UNKNOWN=====' in lines:
        stats['status'] = 'UNKNOWN'
    elif '=====UNSATISFIABLE=====' in lines:
        stats['status'] = 'UNSATISFIABLE'
    elif '==========' in lines:
        stats['status'] = 'COMPLETE'
    elif int(stats['solutions']) > 0:
        stats['status'] = 'SOLFOUND'
    else:
        stats['status'] = 'UNKNOWN'

    return [stats]


def convert_memory(memstring):
    if memstring.endswith('K'):
        return float(memstring[:-1])
    elif memstring.endswith('M'):
        return float(memstring[:-1]) * 1000
    elif memstring.endswith('G'):
        return float(memstring[:-1]) * 1000 * 1000
    else:
        return float(memstring) / 1000


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Need logdir as parameter')

    logdir = sys.argv[1]
    logname = logdir + '.csv'
    solution_file = logdir + '_solutions.csv'
    filelist = sorted(os.listdir(logdir))
    instances = pd.read_csv(os.path.join(logdir, 'instances'), sep=';',
                            names=['solver', 'mzn_path', 'dzn_path', 'lbarg', 'ubarg', 'timeout', 'addargs', 'comment'])

    dzn_filter = [os.path.basename(dzn) for dzn in instances.dzn_path.unique()]

    records = []
    all_solutions = []
    failed = []

    for f in filelist:
        if not f.endswith('.log'):
            # Not a log file, ignore
            continue

        try:
            stats, solutions = process_logfile(f, logdir, instances)

            records.extend(stats)
            all_solutions.append(solutions)
        except Exception as e:
            print(f, e)
            failed.append(f)
            continue

    solution_df = pd.concat(all_solutions)
    solution_df.to_csv(solution_file, index=False)

    df = pd.DataFrame.from_records(records)
    df.to_csv(logname, index=False)

    missing = [str(x) for x in range(1, len(instances) + 1) if x not in df.taskid.unique()]
    print('{}\t Results: {}, Intermediate: {}, Failed: {}, Missing: {}'.format(logname,
                                                                               (df['status'] != 'INTERMEDIATE').sum(),
                                                                               (df['status'] == 'INTERMEDIATE').sum(),
                                                                               len(failed),
                                                                               ",".join(missing)))
