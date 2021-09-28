import os
from collections import namedtuple

import pymzn


class Problem(object):
    def __init__(self, name, mzn, basedir, minmax, objective_var='objective'):
        self.name = name
        self.mzn = mzn
        self.basedir = basedir
        self.minmax = minmax
        self.objective_var = objective_var

    def get_dzns(self, dzn_filter=None):
        problem_dir = os.path.join(self.basedir, self.name)
        problem_dzns = []  # List of all dzns, (filename, fullpath)

        for (sub_dir, _, files) in os.walk(problem_dir):
            sub_dzns = []

            for f in files:
                if not f.endswith('.dzn'):
                    continue

                if dzn_filter and f not in dzn_filter:
                    continue

                sub_dzns.append((f, os.path.join(sub_dir, f)))

            problem_dzns.extend(sub_dzns)

        return problem_dzns

    @property
    def problem_dir(self):
        return os.path.join(self.basedir, self.name)

    @property
    def mzn_path(self):
        return os.path.join(self.problem_dir, self.mzn)

    def __repr__(self):
        return 'Problem(%s)' % self.name


LARGE_PROBLEMS = [
    # Problem('tsp', 'tsp.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # special case, not part of minizinc benchmarks
    # Problem('tsp-gcc', 'tsp_gcc.mzn', 'tsp-minizinc', 'min', 'objective'),  # special case, not part of minizinc benchmarks
    Problem('mrcpsp', 'mrcpsp.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 11182
    Problem('rcpsp', 'rcpsp.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 2904
    # Problem('2DBinPacking', '2DPacking.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 500
    Problem('2DBinPacking', '2DLevelPacking.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 500
    Problem('cutstock', 'cutstock.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 121
    # Problem('fast-food', 'fastfood.mzn', 'minizinc-benchmarks', 'min', 'obj'),  # 89  # fast-food uses string arrays
    # Problem('prize-collecting', 'pc.mzn', 'minizinc-benchmarks', 'max', 'objective'),  # 80
    Problem('jobshop', 'jobshop.mzn', 'minizinc-benchmarks', 'min', 't_end'),  # 74
    Problem('vrp', 'vrp.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 74
    Problem('open_stacks', 'open_stacks_01.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 50
]

SMALL_PROBLEMS = [
    # Problem('amaze', 'amaze.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 47
    # Problem('filters', 'filter.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 29
    Problem('table-layout', 'TableLayout.mzn', 'minizinc-benchmarks', 'min', 'totalheight'),  # 26
    # Problem('ship-schedule', 'ship-schedule.mip.mzn', 'minizinc-benchmarks', 'max', 'objective'),  # 24  # Good with CBC, but that's not considered
    # Problem('ship-schedule', 'ship-schedule.cp.mzn', 'minizinc-benchmarks', 'max', 'objective'),  # 24
    Problem('depot-placement', 'depot_placement.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 24
    # Problem('ghoulomb', 'ghoulomb.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 23
    # Problem('roster', 'roster_model.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 20  # objective = 0?
    # Problem('league', 'league.mzn', 'minizinc-benchmarks', 'min', 'obj'),  # 20
    # Problem('mspsp', 'mspsp.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 20
    # Problem('rcpsp-max', 'rcpsp_max.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 20 / Disabled due to high memory demand
    # Problem('tpp', 'tpp.mzn', 'minizinc-benchmarks', 'min', 'objective'),  # 20
    # Problem('carpet-cutting', 'cc_base.mzn', 'minizinc-benchmarks', 'min', 'RollLen'),  # 20
]

MZNC1617_PROBLEMS = [
    # MZNC 2017
    Problem('cargo', 'cargo_coarsePiles.mzn', 'mznc1617_probs', 'min'),
    Problem('city-position', 'city-position.mzn', 'mznc1617_probs', 'min'),
    Problem('community-detection', 'community-detection.mzn', 'mznc1617_probs', 'max'),
    Problem('crosswords', 'crossword_opt.mzn', 'mznc1617_probs', 'max'),
    Problem('gbac', 'gbac.mzn', 'mznc1617_probs', 'min'),
    Problem('groupsplitter', 'group.mzn', 'mznc1617_probs', 'max'),
    Problem('hrc', 'hrc.mzn', 'mznc1617_probs', 'min'),
    Problem('jp-encoding', 'jp-encoding.mzn', 'mznc1617_probs', 'min'),
    Problem('ma-path-finding', 'mapf.mzn', 'mznc1617_probs', 'min'),
    Problem('mario', 'mario.mzn', 'mznc1617_probs', 'max'),
    Problem('opd', 'opd.mzn', 'mznc1617_probs', 'min'),
    Problem('opt-cryptanalysis', 'mznc1617_aes_opt.mzn', 'mznc1617_probs', 'min'),
    Problem('rcpsp-wet', 'rcpsp-wet.mzn', 'mznc1617_probs', 'min'),
    Problem('rel2onto', 'rel2onto.mzn', 'mznc1617_probs', 'min'),
    Problem('road-cons', 'road_naive.mzn', 'mznc1617_probs', 'min'),
    # Problem('routing-flexible', 'TableLayout.mzn', 'mznc1617_probs', 'min'),
    Problem('steelmillslab', 'steelmillslab.mzn', 'mznc1617_probs', 'min'),
    Problem('tc-graph-color', 'tcgc2.mzn', 'mznc1617_probs', 'min'),
    Problem('tdtsp', 'tdtsp.mzn', 'mznc1617_probs', 'min'),
    Problem('traveling-tppv', 'ttppv.mzn', 'mznc1617_probs', 'min'),
    # MZNC 2016
    Problem('carpet-cutting', 'cc_base.mzn', 'mznc1617_probs', 'min'),
    Problem('celar', 'celar.mzn', 'mznc1617_probs', 'min'),
    Problem('depot-placement', 'depot_placement.mzn', 'mznc1617_probs', 'min'),
    Problem('diameterc-mst', 'dcmst.mzn', 'mznc1617_probs', 'min'),
    Problem('elitserien', 'handball.mzn', 'mznc1617_probs', 'min'),
    Problem('filters', 'filter.mzn', 'mznc1617_probs', 'min'),
    Problem('gfd-schedule', 'gfd-schedule2.mzn', 'mznc1617_probs', 'min'),
    # Problem('java-auto-gen', 'TableLayout.mzn', 'mznc1617_probs', 'min'),
    Problem('mapping', 'mapping.mzn', 'mznc1617_probs', 'min'),
    Problem('maximum-dag', 'maximum-dag.mzn', 'mznc1617_probs', 'max'),
    Problem('mrcpsp', 'mrcpsp.mzn', 'mznc1617_probs', 'min'),
    Problem('nfc', 'nfc.mzn', 'mznc1617_probs', 'min'),
    Problem('prize-collecting', 'pc.mzn', 'mznc1617_probs', 'max'),
    Problem('rcpsp-wet', 'rcpsp-wet.mzn', 'mznc1617_probs', 'min'),
    Problem('tpp', 'tpp.mzn', 'mznc1617_probs', 'min'),
    Problem('zephyrus', 'zephyrus.mzn', 'mznc1617_probs', 'min')
]

PROBLEMS = LARGE_PROBLEMS #+ SMALL_PROBLEMS

TIMEOUT = 3 * 60

TMP_DIR = 'tmp/'  # or /tmp/
MODEL_DIR = 'models/'
OUTPUT_DIR = '/home/helge/Dropbox/Simula/CpBoundEst/cpboundest_vatras/'
# OUTPUT_DIR = '/home/helge/Dropbox/Simula/cpboundest_riordian/'
# OUTPUT_DIR = 'results/'

ALLOW_PARALLEL = True
PARALLEL_POOL_SIZE = 4

KFOLD = 10

Solver = namedtuple('Solver', ['name', 'obj', 'args'])

SOLVER = {
    'chuffed': Solver('chuffed', pymzn.Chuffed('mzn-chuffed', 'fzn-chuffed', 'chuffed'), []),
    # '/usr/bin/fzn-chuffed', 'chuffed', None),  # has default timeout of 30 min
    'gecode': Solver('gecode', pymzn.Gecode('/usr/bin/mzn-gecode', '/usr/bin/fzn-gecode', 'gecode'), []),
    # Solver('gecode', '/usr/bin/fzn-gecode', 'gecode', None),
    # 'gurobi': Solver('gurobi', '/usr/bin/mzn-gurobi', 'linear', None),  # accepts also .fzn
    'cbc': Solver('cbc', pymzn.CBC('/usr/bin/mzn-cbc', 'linear'), []),
    # Solver('cbc', '/usr/bin/mzn-cbc', 'linear', None),  # accepts also .fzn
    #'sicstus': Solver('sicstus', pymzn_solvers.Sicstus('/usr/bin/minizinc', '/usr/local/bin/spfz', 'sicstus'), []),
    # 'g12mip': Solver('g12mip', pymzn.G12MIP('/usr/bin/mzn-g12mip', '/usr/bin/flatzinc', 'linear'), []),  # no timeout
    # 'g12fd': Solver('g12fd', pymzn.G12MIP('/usr/bin/mzn-g12fd', '/usr/bin/flatzinc', 'g12_fd'), []),  # no timeout
    # 'g12lazy': Solver('g12lazy', pymzn.G12MIP('/usr/bin/mzn-g12lazy', '/usr/bin/flatzinc', 'g12_lazyfd'), []),  # no timeout
    'cplex': Solver('cplex', pymzn.CBC('./mzn-cplex', 'linear'),
                    ['--stdlib-dir', '/usr/share/minizinc-ide/share/minizinc', '--workmem', '10000']),
    'ortools': Solver('ortools', None, None),
    'choco': Solver('choco', None, None),
    'sunnycp': Solver('sunnycp', None, None)
}

ALTSOLVER = {
    'cplexul': Solver('cplex', pymzn.CBC('./mzn-cplex', 'linear'),
                      ['--stdlib-dir', '/usr/share/minizinc-ide/share/minizinc', '-p', '1', '--workmem', '3000']),
}


class ExecConfig(
    namedtuple('ExecConfig',
               ['solver', 'problem', 'mzn_path', 'dzn', 'dzn_path', 'lower_bound', 'upper_bound', 'lower_bound_value',
                'upper_bound_value', 'timeout', 'dataset', 'objective', 'duration', 'is_complete', 'has_failed'])):
    def __new__(cls, solver, problem, mzn_path, dzn, dzn_path, lower_bound=None, upper_bound=None,
                lower_bound_value=None, upper_bound_value=None, timeout=TIMEOUT, dataset=None, objective=None,
                duration=None, is_complete=False, has_failed=False):
        if (lower_bound is None) ^ (lower_bound_value is None):
            lower_bound = lower_bound if lower_bound is not None else lower_bound_value
            lower_bound_value = lower_bound_value if lower_bound_value is not None else lower_bound

        if (upper_bound is None) ^ (upper_bound_value is None):
            upper_bound = upper_bound if upper_bound is not None else upper_bound_value
            upper_bound_value = upper_bound_value if upper_bound_value is not None else upper_bound

        return super().__new__(cls, solver, problem, mzn_path, dzn, dzn_path, lower_bound, upper_bound,
                               lower_bound_value, upper_bound_value, timeout, dataset, objective, duration, is_complete,
                               has_failed)

    __slots__ = ()

    @property
    def has_bound(self):
        return self.has_lower_bound or self.has_upper_bound

    @property
    def has_lower_bound(self):
        return self.lower_bound_value is not None

    @property
    def has_upper_bound(self):
        return self.upper_bound_value is not None

    @property
    def opt_path(self):
        return self.dzn_path + '.opt'

    @property
    def best_path(self):
        return self.dzn_path + '.best'


if __name__ == '__main__':
    print('This is not meant to be directly executed.')
