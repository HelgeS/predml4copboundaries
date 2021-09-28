import logging
import os
import subprocess

from solve import helper


class BaseSolver(object):
    def solve(self, problem, dzn, timeout=0, ub=None, lb=None, dataset=None, free_search=False):
        dzn_name = os.path.splitext(os.path.basename(dzn))[0]
        mzn_new = helper.post_constraints(problem, dzn_name, ub=ub, lb=lb)

        cmd = self.get_cmd(mzn_new, dzn, timeout, free_search)

        try:
            output = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
            output = str(output, 'utf-8').splitlines()
            print(output)
        except Exception:
            print(cmd)
            raise

        os.unlink(mzn_new)

        stats = self.parse_stats(output, problem, lb, ub, dzn_name, dataset, timeout)
        return stats

    def get_cmd(self, mzn, dzn, timeout=0, free_search=False):
        pass

    def parse_stats(self, output, problem, lb, ub, dzn_name, dataset, timeout):
        return {}


class Choco(BaseSolver):
    def get_cmd(self, mzn, dzn, timeout=0, free_search=False):
        cmd = ['choco.sh', mzn, dzn, '-s']

        if timeout and timeout > 0:
            cmd.extend(['--fzn-flags', '--time-out {:d}'.format(timeout)])

        return cmd


class Sicstus(BaseSolver):
    def get_cmd(self, mzn, dzn, timeout=0, free_search=False):
        timeout_opt = ",timeout({:d})".format(timeout * 1000) if timeout and timeout > 0 else ""

        prolog = """
        use_module(library(clpfd)),
        use_module(library(zinc)),
        statistics(runtime,_),
        mzn_run_file('{}', [data_file('{}'),solutions(all),statistics(true){}]),
        statistics(runtime, [_,T]),
        fd_statistics,
        format(user_error, 'Runtime: ~d\n', [T]),
        halt.
        """.format(mzn, dzn, timeout_opt).replace('\n', '').replace('\r', '')

        cmd = ['sicstus', '--noinfo', '--nologo', '--goal', prolog]
        return cmd

    def parse_stats(self, output, problem, lb, ub, dzn_name, dataset, timeout):
        stats = {}
        obj_value = None

        for line in output:
            if ':' not in line and not line.startswith('objective = '):
                continue

            if line.startswith('objective = '):
                obj_value = int(line.replace('objective = ', '').replace(';', ''))

            try:
                key, value = line.split(': ')
                stats[key] = value
            except ValueError as e:
                logging.warning(e)

        stats['Solutions'] = output.count('----------')
        stats['Satisfiable'] = '=====UNSATISFIABLE=====' not in output and stats['Solutions'] > 0
        stats['Timeout'] = timeout * 1000
        stats['TimedOut'] = int(stats['Runtime']) >= timeout * 1000 or '=====UNKNOWN=====' in output
        stats['LowerBoundValue'] = lb
        stats['UpperBoundValue'] = ub
        stats['Objective'] = obj_value

        if int(stats['Solutions']) > 0:
            if stats['TimedOut']:
                status = 'Satisfied'
            else:
                status = 'Complete'
        else:
            if stats['TimedOut']:
                status = 'Unknown'
            else:
                status = 'Unsatisfiable'

        stats['Status'] = status

        if dataset and len(dataset) == 2:
            stats['LowerBound'] = dataset[0]
            stats['UpperBound'] = dataset[1]
        else:
            stats['LowerBound'] = lb
            stats['UpperBound'] = ub

        stats['Problem'] = problem.name
        stats['Instance'] = dzn_name
        stats['Solver'] = 'sicstus'

        if 'Constraints created' in stats:
            stats['Constraints'] = stats['Constraints created']
            del stats['Constraints created']

        return stats


class Chuffed(BaseSolver):
    def get_cmd(self, mzn, dzn, timeout=0, free_search=False):
        cmd = ['mzn-chuffed', mzn, dzn, '-s']

        if free_search:
            cmd.extend(['--fzn-flags', '-f'])

        if timeout and timeout > 0:
            cmd.extend(['--fzn-flags', '--time-out {:d}'.format(timeout)])

        return cmd

    def parse_stats(self, output, problem, lb, ub, dzn_name, dataset, timeout):
        fields = ['Variables', 'SATVariables', 'Propagators', 'Conflicts', 'SATBackjumps', 'Propagations',
                  'Solutions', 'Inittime', 'search_time']
        values = output[0].split(',')

        if len(values) == len(fields) + 1:
            fields = ['Objective'] + fields

        stats = {f: v for f, v in zip(fields, values)}

        stats['Satisfiable'] = '=====UNSATISFIABLE=====' not in output and int(stats['Solutions']) > 0
        stats['Timeout'] = timeout * 1000
        stats['TimedOut'] = '% Time limit exceeded!' in output
        stats['LowerBoundValue'] = lb
        stats['UpperBoundValue'] = ub
        stats['Inittime'] = float(stats['Inittime']) * 1000
        stats['search_time'] = float(stats['search_time']) * 1000

        if int(stats['Solutions']) > 0:
            if stats['TimedOut']:
                status = 'Satisfied'
            else:
                status = 'Complete'
        else:
            if stats['TimedOut']:
                status = 'Unknown'
            else:
                status = 'Unsatisfiable'

        stats['Status'] = status

        if dataset and len(dataset) == 2:
            stats['LowerBound'] = dataset[0]
            stats['UpperBound'] = dataset[1]
        else:
            stats['LowerBound'] = lb
            stats['UpperBound'] = ub

        stats['Problem'] = problem.name
        stats['Instance'] = dzn_name
        stats['Solver'] = 'chuffed'

        return stats


class OrTools(BaseSolver):
    def get_cmd(self, mzn, dzn, timeout=0, free_search=False):
        cmd = ['minizinc', '-Gortools', '-f', 'fzn-or-tools', mzn, dzn, '-fzn-flags', '-statistics', '--fzn-flags',
               '-fz_logging']

        if free_search:
            cmd.extend(['--fzn-flags', '-free_search'])

        if timeout and timeout > 0:
            cmd.extend(['--fzn-flags', '--time_limit {:d}'.format(timeout * 1000)])

        return cmd

    def parse_stats(self, output, problem, lb, ub, dzn_name, dataset, timeout):
        fields = ['Variables', 'SATVariables', 'Propagators', 'Conflicts', 'SATBackjumps', 'Propagations',
                  'Solutions', 'Inittime', 'search_time']
        values = output[0].split(',')

        if len(values) == len(fields) + 1:
            fields = ['Objective'] + fields

        stats = {f: v for f, v in zip(fields, values)}

        stats['Satisfiable'] = '=====UNSATISFIABLE=====' not in output and int(stats['Solutions']) > 0
        stats['Timeout'] = timeout * 1000
        stats['TimedOut'] = '% Time limit exceeded!' in output
        stats['LowerBoundValue'] = lb
        stats['UpperBoundValue'] = ub
        stats['Inittime'] = float(stats['Inittime']) * 1000
        stats['search_time'] = float(stats['search_time']) * 1000

        if int(stats['Solutions']) > 0:
            if stats['TimedOut']:
                status = 'Satisfied'
            else:
                status = 'Complete'
        else:
            if stats['TimedOut']:
                status = 'Unknown'
            else:
                status = 'Unsatisfiable'

        stats['Status'] = status

        if dataset and len(dataset) == 2:
            stats['LowerBound'] = dataset[0]
            stats['UpperBound'] = dataset[1]
        else:
            stats['LowerBound'] = lb
            stats['UpperBound'] = ub

        stats['Problem'] = problem.name
        stats['Instance'] = dzn_name
        stats['Solver'] = 'chuffed'

        return stats
