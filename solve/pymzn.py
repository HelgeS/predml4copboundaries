import logging
import time

import math
import pymzn

import config


def solve(problem, dzn, lb, ub, timeout, dataset, solvers, output_file=None):
    mzn_path = problem.mzn_path
    dzn_path = dzn

    for solver_key in solvers:
        solver = config.SOLVER[solver_key]

        conf = config.ExecConfig(solver, problem, mzn_path, dzn, dzn_path,
                                 lower_bound_value=lb, upper_bound_value=ub,
                                 timeout=timeout, dataset=dataset)

        res = execute_config(conf)
        output = '{};{};{};{:.2f};{};{};{};{};{};{}'.format(res.solver.name, res.problem.name, res.dzn,
                                                            res.duration,
                                                            res.objective, res.is_complete, res.has_failed,
                                                            res.has_bound, res.obj_bound, res.dataset)
        print(output)

        if output_file:
            print(output, file=open(output, 'a'))


def execute_config(conf):
    objective = None
    is_complete = False
    has_failed = False

    start = time.time()

    try:
        if conf.has_bound:
            model = pymzn.MiniZincModel(conf.mzn_path)

            if conf.has_upper_bound and conf.has_lower_bound:
                bound_constraint = '{:d} <= {} /\ {} <= {:d}'.format(conf.problem.objective_var,
                                                                     math.floor(conf.obj_bound),
                                                                     conf.problem.objective_var,
                                                                     math.ceil(conf.obj_bound))
            elif conf.has_upper_bound:
                bound_constraint = '{} <= {:d}'.format(conf.problem.objective_var, math.ceil(conf.obj_bound))
            elif conf.has_lower_bound:
                bound_constraint = '{} >= {:d}'.format(conf.problem.objective_var, math.floor(conf.obj_bound))

            model.constraint(bound_constraint)
        else:
            model = conf.mzn_path

        objective_values = pymzn.minizinc(model, *([conf.dzn_path] + conf.solver.args), solver=conf.solver.obj,
                                          timeout=conf.timeout, output_vars=[conf.problem.objective_var])

        is_complete = objective_values.complete

        if len(list(objective_values)) > 0:
            objective = objective_values[-1][conf.problem.objective_var]
    except pymzn.MiniZincUnsatisfiableError as e:
        is_complete = True  # Completed with guarantee = no solution possible
        has_failed = True
        logging.warning('%s / %s / %s failed: unsatisfiable', conf.solver.name, conf.problem.name, conf.dzn)

        time_left = int(conf.timeout - (time.time() - start))

        # If failed before timeout, run again with switched boundary (upper <=> lower)
        if time_left > 0 and conf.has_bound:
            if conf.has_upper_bound:
                new_bound_type = 'lower'
            else:
                new_bound_type = 'upper'

            new_conf = conf._replace(timeout=time_left, bound_type=new_bound_type)

            logging.info('%s / %s / %s rerun with %.2f sec remaining', conf.solver.name, conf.problem.name, conf.dzn,
                         new_conf.timeout)
            new_res = execute_config(new_conf)
            objective = new_res.objective
            is_complete = new_res.is_complete
            has_failed = new_res.has_failed
    except Exception as e:
        has_failed = True
        logging.critical('%s / %s / %s failed: %s', conf.solver.name, conf.problem.name, conf.dzn, str(e))

    end = time.time()
    duration = end - start

    return conf._replace(duration=duration, objective=objective, is_complete=is_complete, has_failed=has_failed)