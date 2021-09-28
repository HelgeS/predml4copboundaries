import pymzn
from tqdm import tqdm
import os

import config

VALIDATIONFILE = '20180816-validated'
validated = open(VALIDATIONFILE, 'r').readlines() if os.path.isfile(VALIDATIONFILE) else []

for p in config.PROBLEMS[3:5]:
    for (dzn_name, dzn_path) in tqdm(p.get_dzns(), desc=p.name):
        opt_path = dzn_path + '.opt'

        if dzn_path in validated or not os.path.isfile(opt_path):
            continue

        optimum = int(open(opt_path, 'r').read())
        m = pymzn.MiniZincModel(p.mzn_path)
        m.constraint("{} < {}".format(p.objective_var, optimum))

        try:
            sol = pymzn.minizinc(m, dzn_path, solver=pymzn.or_tools, parallel=6, all_solutions=True, timeout=3600)

            if len(sol) > 0:
                print("{}: Invalid".format(opt_path))
                if sol.complete:
                    open(opt_path, 'w').write("{}".format(sol[-1][p.objective_var]))
                else:
                    os.unlink(opt_path)
        except pymzn.MiniZincUnsatisfiableError:
            pass
        except RuntimeError:
            print("{}: Valid, but strange error".format(opt_path))

        open(VALIDATIONFILE, 'a').write(dzn_path + os.linesep)
