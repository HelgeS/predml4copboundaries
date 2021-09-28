import pandas as pd
import config
import data_loader
import os


def get_problem(mzn):
    return next(p for p in config.PROBLEMS if mzn == p.mzn)


best_results = data_loader.get_best_results()
best_results = best_results[['mzn', 'dzn', 'objective']].set_index(['mzn', 'dzn'])
best_results.rename(columns={'objective': 'best'}, inplace=True)

run_results = pd.read_csv('logs/solver_evaluation_scaled_y_4h.csv')
run_results = run_results[(run_results.bounds == "No") & (run_results.mzn != 'tsp.mzn')]

df = run_results[['mzn', 'dzn', 'solver', 'first_objective']]
df = df[~df.first_objective.isna()]
df['first_objective'] = df['first_objective'].astype(int)

ndf = df.join(best_results, how='left', on=['mzn', 'dzn'])
ndf['bound'] = ndf['best'] + (ndf['first_objective'] - ndf['best']) // 2  # Set bound in middle (i.e. 50% QOF)
ndf['ubarg'] = ndf['bound'].apply(lambda x: '-ub ' + str(x))
ndf['lbarg'] = ''
ndf['timeout'] = 14400 # 1200  # int(run_results.first_sol_time.max() * 1.2)
ndf['comment'] = ndf['bound'].apply(lambda x: '(None,' + str(x) + ')')

for mzn in ndf.mzn.unique():
    p = get_problem(mzn)
    dzns = p.get_dzns()
    dzn_path = lambda d: next(dzn[1] for dzn in dzns if dzn[0] == d)
    ndf.loc[ndf.mzn == mzn, 'mzn_full'] = p.mzn_path
    ndf.loc[ndf.mzn == mzn, 'dzn_full'] = ndf.loc[ndf.mzn == mzn, 'dzn'].apply(dzn_path)
    ndf.loc[ndf.mzn == mzn, 'objective_var'] = '-var ' + p.objective_var

ndf.sort_values(['mzn_full', 'dzn_full', 'solver'], inplace=True)
ndf[['solver', 'mzn_full', 'dzn_full', 'lbarg', 'ubarg', 'timeout', 'objective_var', 'comment']].to_csv('sanity_check_instances', sep=';', index=False, header=False)
