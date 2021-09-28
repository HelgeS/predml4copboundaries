import argparse
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon
from statsmodels import robust

import config
import data_loader
import solve.helper
from data_loader import load_stats_log, load_predict_log


def figsize_column(scale, height_ratio=1.0):
    fig_width_pt = 239  # Get this from LaTeX using \the\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * height_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def figsize_text(scale, height_ratio=1.0):
    fig_width_pt = 505  # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * height_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


pgf_with_latex = {  # setup matplotlib to use latex for output
    "pgf.texsystem": "pdflatex",  # change this if using xetex or lautex
    "text.usetex": True,  # use LaTeX to write all text
    "font.family": "serif",
    "font.serif": [],  # blank entries should cause plots to inherit fonts from the document
    "font.sans-serif": [],
    "font.monospace": [],
    "axes.labelsize": 9,
    "font.size": 9,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.figsize": figsize_column(1.0),
    "text.latex.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts because your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}

sns.set_style("whitegrid", pgf_with_latex)
sns.set_context("paper")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

LABELS = {
    'mrcpsp': 'MRCPSP',
    'rcpsp': 'RCPSP',
    '2DBinPacking': 'Bin Packing',
    '2DLevelPacking': 'Bin Packing',
    'prize-collecting': 'Prize Coll.',
    'jobshop': 'Jobshop',
    'vrp': 'VRP',
    'tsp': 'TSP',
    'open_stacks': 'Open Stacks',
    'cutstock': 'Cutting Stock',
    '2DLevelPacking.mzn': 'Bin Packing',
    'cutstock.mzn': 'Cutting Stock',
    'mrcpsp.mzn': 'MRCPSP',
    'jobshop.mzn': 'Jobshop',
    'open_stacks_01.mzn': 'Open Stacks',
    'rcpsp.mzn': 'RCPSP',
    'tsp.mzn': 'TSP',
    'vrp.mzn': 'VRP',
    'chuffed': 'Chuffed',
    'choco': 'Choco',
    'ortools': 'OR-Tools',
    'gecode': 'Gecode',
    'sunnycp': 'Sunny-CP',
    'objective': 'O',
    'objective_diff': 'O',
    'time': 'T',
    'time_diff': 'T',
    'mzn': 'Model',
    'solver': 'Solver',
    'filters': 'Filters',
    'table-layout': 'Table Layout',
    'depot-placement': 'Depot Place.',
    'carpet-cutting': 'Carpet',
    'average': 'Avg.',
    'extreme': 'Max.',
    'network': 'NN$_s$',
    'networka': 'NN$_a$',
    'xgb': 'GTB$_s$',
    'xgba': 'GTB$_a$',
    'linear': 'LR',
    'svm': 'SVM',
    'pruned_domain': 'Domain Pruned (%)',
    'pruned_ratio': 'Pruning (%)',
    'estimator': 'Prediction Model',
    'adjustment': 'Target Shift',
    'problem': 'Problem',
    'count': 'I',
    'obj_diff': 'QOF',
    'time_diff': 'TTF',
    'hard': 'Boundary Constraints',
    'soft': 'Bounds-Aware Search',
    'est': 'EST',
}


def estimation_table_network(files, filename=None):
    labels = {
        'problem': 'Problem',
        'dzn': 'Inst.',
        'correct': 'Satisfiable (%)',
        'error': 'Error (%)',
        'loss_fn': 'Loss',
        'ensemble_mode': 'Ensemble Mode'
    }

    active_problems = [p.name for p in config.PROBLEMS]

    dfs = []

    for i, f in enumerate(files):
        dump = pickle.load(open(f, 'rb'))

        if dump['loss_fn'] != 'shiftedmse':
            continue

        if dump['problem'].name not in active_problems:
            continue

        pred_df = pd.DataFrame(dump['prediction'], columns=['dzn', 'predicted', 'truth'])
        pred_df['iteration'] = i
        pred_df['problem'] = dump['problem'].name
        pred_df['loss_fn'] = dump['loss_fn']
        pred_df['ensemble_mode'] = dump['ensemble_mode']

        if dump['problem'].minmax == 'min':
            pred_df['correct'] = pred_df['predicted'] >= pred_df['truth']
        else:
            pred_df['correct'] = pred_df['predicted'] <= pred_df['truth']

        corr_df = pred_df[pred_df['correct']]
        pred_df['error'] = abs(corr_df['predicted'] - corr_df['truth']) / corr_df['truth']

        dfs.append(pred_df)

    df = pd.concat(dfs)
    gdf = df.groupby(['problem', 'ensemble_mode', 'loss_fn'], as_index=False).agg({
        'dzn': 'nunique',
        'correct': lambda x: 100 * np.mean(x),
        'error': lambda x: 100 * np.mean(x)
    })

    gdf = gdf.round(2)
    gdf = gdf.rename(columns=labels)
    gdf = gdf.replace(LABELS.keys(), LABELS.values())

    out_df = gdf[[labels['problem'], labels['ensemble_mode'], labels['correct'], labels['error']]]
    out_df = out_df.pivot(index=labels['problem'], columns=labels['ensemble_mode'])
    print(out_df)

    if filename:
        output_string = out_df.to_latex(multicolumn=True, multicolumn_format='c')
        open(filename, 'w').write(output_string)


def adjustment_table(logfile, outputdir=None):
    df = load_stats_log(logfile)

    df.loc[(df.estimator == 'network') & (df.loss == 'shiftedmse'), 'estimator'] = 'networka'
    df[['pruned_domain', 'adjustment']] *= 100

    out_df = df.pivot_table(index='adjustment', columns='estimator', values='pruned_domain')

    out_df = out_df.round(1)
    out_df.index = out_df.index.astype(int)
    out_df.rename(columns=LABELS, inplace=True)
    out_df.sort_index(level=0, axis=1, inplace=True)

    if outputdir:
        formatter = [(lambda cmax: (lambda x: max_formatter(x, cmax)))(cmax) for cmax in out_df.max(axis=0).tolist()]

        output_string = out_df.to_latex(index_names=False, column_format='rrrrrrr', formatters=formatter, escape=False)
        output_string = output_string.replace('toprule\n{} &', 'toprule\nAdj. &')
        output_string = output_string.replace('\\midrule\n', '')
        output_string = output_string.replace('\n0 ', '\n\\midrule\n0 ')

        if 1 in df.outputs.unique():
            filename = 'adjustment_o1.tex'
        else:
            filename = 'adjustment_o2.tex'

        open(os.path.join(outputdir, filename), 'w').write(output_string)

    print(out_df)
    out_df.to_clipboard()


def adjustment_graph(logfile, outputdir=None, column='true_overest'):
    if outputdir:
        height_ratio = 0.4 if column in ('true_overest', 'true_pairs') else 0.65
        _, ax = plt.subplots(figsize=figsize_column(1.0, height_ratio=height_ratio))
    else:
        _, ax = plt.subplots()

    df = load_stats_log(logfile)

    df.loc[(df.estimator == 'network') & (df.loss == 'shiftedmse'), 'estimator'] = 'networka'
    df[[column]] *= 100

    df = df[(df.estimator.isin(['network', 'networka', 'xgb', 'xgba', 'linear', 'svm'])) & (df.adjustment < 1.0)]

    # df['adjustment'] = df['adjustment'].astype(int)
    df = df.replace(LABELS.keys(), LABELS.values())

    sns.pointplot(x='adjustment', y=column, hue='estimator', scale=0.5, estimator=np.median, join=True,
                  # hue_order=['GTB/a', 'GTB/s', 'NN/a', 'NN/s', 'SVM', 'Linear'],
                  markers=['d', '.', '+', 'x', '*', 'v'],
                  dodge=False, data=df, ax=ax)  # ci='sd'

    if column in ('true_overest', 'true_pairs'):
        ax.set_ylabel('Admissible (\%)')
        ax.set_xlabel('')
        ax.legend(ncol=3, loc='lower right', labelspacing=0.2, columnspacing=0.1)
        # ax.set_xticklabels([])
    else:
        ax.set_ylabel('Admissible (\%)')
        ax.set_xlabel('Adjustment Factor $\lambda$')
        # ax.legend_.remove()
        ax.legend(ncol=3, loc='lower right', labelspacing=0.2, columnspacing=0.1)

    ax.set_ylim([0, 100])

    sns.despine(ax=ax)

    if outputdir:
        if 1 in df.outputs.unique():
            filename = 'adjustment_o1_{}.pgf'.format(column)
        else:
            filename = 'adjustment_o2_{}.pgf'.format(column)

        # plt.tight_layout()
        plt.savefig(os.path.join(outputdir, filename), dpi=500, bbox_inches='tight', pad_inches=0)


def estimation_table_o2(logfile, outputdir=None):
    run_name = os.path.basename(os.path.dirname(logfile[0]))

    df = load_stats_log(os.path.join(logfile[0], "*-stats.log"))

    # Backwards compatibility; in new runs it is already named networka
    df.loc[(df.estimator == 'network') & (df.loss == 'shiftedmse'), 'estimator'] = 'networka'
    df = df[df.estimator.isin(['network', 'networka', 'xgb', 'xgba', 'linear', 'svm'])]

    # print(df[['estimator', 'traintime']].groupby(['estimator'], as_index=False).median())
    gdf = df[['estimator', 'adjustment', 'pruned_domain']].groupby(['estimator', 'adjustment'], as_index=False).median()
    bestconfigs = gdf.ix[gdf.groupby('estimator', as_index=False)['pruned_domain'].idxmax()][
        ['estimator', 'adjustment']]

    preddf = load_predict_log(os.path.join(logfile[0], "*-predict.log"))
    preddf.loc[(preddf.estimator == 'network') & (preddf.loss == 'shiftedmse'), 'estimator'] = 'networka'
    preddf = preddf[preddf.estimator.isin(['network', 'networka', 'xgb', 'xgba', 'linear', 'svm'])]

    pdf = preddf.merge(bestconfigs, how='inner', suffixes=['', '_r'], on=['estimator', 'adjustment'])

    # bestconfigs = gdf.ix[gdf.groupby('estimator', as_index=False)['pruned_domain'].idxmax()][
    #    ['estimator', 'adjustment']]
    # pdf = df.merge(bestconfigs, how='inner', suffixes=['', '_r'], on=['estimator', 'adjustment'])

    print(bestconfigs)

    labels = {'true_pair': 'Feas.',
              'size_red': 'Size',
              'gap': 'Gap',
              'pruned_domain': 'SP',
              'pruned_ratio': 'Pruned',
              'estimator': 'Model',
              'problem': 'Problem',
              'overest_error': 'OE',
              'underest_error': 'UE'}

    pdf['true_pair'] = (pdf['underest'] <= pdf['optimum']) & (pdf['optimum'] <= pdf['overest'])
    pdf['size_red'] = 0
    pdf.loc[pdf['true_pair'], 'size_red'] = 1 - (pdf['dom_size_new'] / pdf['dom_size'])
    pdf['gap'] = 0
    pdf.loc[pdf['true_pair'], 'gap'] = 1 - (
            (pdf['dom_upper_new'] - pdf['optimum']).abs() / (pdf['dom_upper'] - pdf['optimum']).abs())
    pdf[['size_red', 'gap']] *= 100
    pdf[['size_red', 'gap']]  # .astype(int, copy=False)

    # pdf[['true_pairs', 'pruned_domain', 'pruned_ratio', 'overest_error', 'underest_error']] *= 100
    # pdf['overest_error'] += 1
    # pdf['underest_error'] += 1
    pdf.rename(columns=labels, inplace=True)
    pdf.replace(LABELS.keys(), LABELS.values(), inplace=True)

    def cust_aggfunc_star(x):
        m = np.median(x)
        # m = np.mean(x)

        std = np.ceil(robust.scale.mad(x)).astype(int)
        # std = np.ceil(np.std(x)).astype(int)

        if std <= 5:
            appendix = ''
        elif std <= 10:
            appendix = '+'
        elif std <= 20:
            appendix = '*'
        elif std <= 30:
            appendix = '**'
        elif std <= 40:
            appendix = '***'
        else:
            appendix = '{:d}'.format(std)
        # appendix = '*' * (min(std // 5, 5))

        if x.name == labels['overest_error']:
            return "{:.1f}\\textsuperscript{{{}}}".format(m, appendix)
        else:
            m = np.round(m).astype(int)
            return "{:d}\\textsuperscript{{{}}}".format(m, appendix)

    def cust_aggfunc_pm(x):
        m = np.floor(np.median(x)).astype(int)
        std = np.ceil(robust.scale.mad(x)).astype(int)
        # std = np.ceil(np.std(x)).astype(int)

        if std > 5:
            return "{:d}\\textsuperscript{{$\\pm${:d}}}".format(m, std)
        else:
            return "{:d}\\textsuperscript{{}}".format(m)

    def median_int(x):
        m = np.median(x)
        m = np.floor(m).astype(int)
        return "{:d}".format(m)

    out_df = pdf.pivot_table(index=labels['problem'], columns=labels['estimator'], margins=False, margins_name='All',
                             values=[labels['size_red'], labels['gap']], aggfunc=cust_aggfunc_star)
    out_df.columns = out_df.columns.swaplevel(0, 1)
    out_df.sort_index(level=0, axis=1, inplace=True)

    if outputdir:
        output_string = out_df.to_latex(multicolumn=True, multicolumn_format='c',
                                        column_format='l' + "|".join(
                                            ['RR' for _ in range(pdf[labels['estimator']].nunique())]),
                                        index_names=False, escape=False)
        output_string = output_string.replace('\\begin{tabular}', '\\begin{tabularx}{0.97\\textwidth}')
        output_string = output_string.replace('\\end{tabular}', '\\end{tabularx}')
        output_string = output_string.replace('±', '\\(\\pm\\)')

        filename = 'estimation_o2.tex'

        open(os.path.join(outputdir, filename), 'w').write(output_string)


def estimation_table(logfile, outputdir=None):
    run_name = os.path.basename(os.path.dirname(logfile[0]))

    df = load_stats_log(logfile)
    # Backwards compatibility; in new runs it is already named networka
    df.loc[(df.estimator == 'network') & (df.loss == 'shiftedmse'), 'estimator'] = 'networka'
    df = df[df.estimator.isin(['network', 'networka', 'xgb', 'xgba', 'linear', 'svm'])]

    print(df[['estimator', 'traintime']].groupby(['estimator'], as_index=False).median())

    gdf = df[['estimator', 'adjustment', 'pruned_domain']].groupby(['estimator', 'adjustment'], as_index=False).median()

    # gdf['pruned_domain_both'] = gdf['true_pairs'] & gdf['pruned_lower_dom'] & gdf['pruned_upper_dom']

    bestconfigs = gdf.ix[gdf.groupby('estimator', as_index=False)['pruned_domain'].idxmax()][
        ['estimator', 'adjustment']]
    pdf = df.merge(bestconfigs, how='inner', suffixes=['', '_r'], on=['estimator', 'adjustment'])

    print(bestconfigs)

    if df.outputs.unique()[0] == 1:
        labels = {'true_pairs': 'Feas.',
                  'pruned_domain': 'SP',
                  'pruned_ratio': 'Pruned',
                  'estimator': 'Model',
                  'problem': 'Problem',
                  'overest_error': 'Gap',
                  'underest_error': 'Gap'}
    else:
        labels = {'true_pairs': 'Feas.',
                  'pruned_domain': 'SP',
                  'pruned_ratio': 'Pruned',
                  'estimator': 'Model',
                  'problem': 'Problem',
                  'overest_error': 'OE',
                  'underest_error': 'UE'}

    pdf[['true_pairs', 'pruned_domain', 'pruned_ratio', 'overest_error', 'underest_error']] *= 100
    pdf['overest_error'] += 1
    pdf['underest_error'] += 1
    pdf.rename(columns=labels, inplace=True)
    pdf = pdf.replace(LABELS.keys(), LABELS.values())

    def cust_aggfunc_pm(x):
        m = np.floor(np.median(x)).astype(int)
        std = np.ceil(robust.scale.mad(x)).astype(int)
        # std = np.ceil(np.std(x)).astype(int)

        return "{:d}+-{:2d}".format(m, std)

    def cust_aggfunc_star(x):
        m = np.median(x)
        # m = np.mean(x)

        std = np.ceil(robust.scale.mad(x)).astype(int)
        # std = np.ceil(np.std(x)).astype(int)
        appendix = '*' * (min(std // 5, 5))

        if x.name == labels['overest_error']:
            return "{:.1f}\\textsuperscript{{{}}}".format(m, appendix)
        else:
            m = np.floor(m).astype(int)
            return "{:d}\\textsuperscript{{{}}}".format(m, appendix)

    # Full Table
    out_df = pdf.pivot_table(index=labels['problem'], columns=labels['estimator'], margins=False, margins_name='All',
                             values=[labels['true_pairs'], labels['pruned_ratio'], labels['overest_error'],
                                     labels['underest_error']],
                             aggfunc=cust_aggfunc_pm)
    out_df.columns = out_df.columns.swaplevel(0, 1)
    out_df.sort_index(level=0, axis=1, inplace=True)
    # del out_df['All']  # No separate all columns

    if outputdir:
        output_string = out_df.to_latex(multicolumn=True, multicolumn_format='c',
                                        column_format='l' + "|".join(
                                            ['rrrr' for _ in range(pdf[labels['estimator']].nunique())]),
                                        index_names=False, escape=False)
        output_string = output_string.replace('{} & Feas', 'Problem & Feas')
        output_string = output_string.replace('\\midrule\n', '')
        output_string = output_string.replace('Pruned \\\\\n', 'Pruned \\\\\n\\midrule\n')
        output_string = output_string.replace('\\\n2DBP', '\\\n\\midrule\n2DBP')
        output_string = output_string.replace('+-', '\\(\\pm\\)')

        if 1 in df.outputs.unique():
            filename = 'estimation_o1_full.tex'
        else:
            filename = 'estimation_o2_full.tex'

        open(os.path.join(outputdir, filename), 'w').write(output_string)
        out_df.to_csv(os.path.join(outputdir, filename + '.csv'))

    print(out_df)
    out_df.to_clipboard()
    out_df.to_html(run_name + '_estimation.html')
    out_df.to_csv(run_name + '_estimation.csv')

    # Small table
    out_df = pdf.pivot_table(index=labels['problem'], columns=labels['estimator'], margins=False, margins_name='All',
                             values=[labels['pruned_ratio'], labels['overest_error'], labels['underest_error']],
                             aggfunc=cust_aggfunc_star)
    out_df.columns = out_df.columns.swaplevel(0, 1)
    out_df.sort_index(level=0, axis=1, inplace=True)

    if outputdir:
        output_string = out_df.to_latex(multicolumn=True, multicolumn_format='c', column_format='l' + "|".join(
            ['rrr' for _ in range(pdf[labels['estimator']].nunique())]),
                                        index_names=False, escape=False)
        output_string = output_string.replace('{} & Feas', 'Problem & Feas')
        output_string = output_string.replace('\\midrule\n', '')
        output_string = output_string.replace('Pruned \\\\\n', 'Pruned \\\\\n\\midrule\n')
        output_string = output_string.replace('\\\n2DBP', '\\\n\\midrule\n2DBP')
        output_string = output_string.replace('±', '\\(\\pm\\)')

        if 1 in df.outputs.unique():
            filename = 'estimation_o1.tex'
        else:
            filename = 'estimation_o2.tex'

        open(os.path.join(outputdir, filename), 'w').write(output_string)


def max_formatter(x, max_value):
    if x == max_value:
        return '\\textbf{%s}' % x
    else:
        return str(x)


def estimation_bars(logfile, outputdir=None):
    if outputdir:
        _, ax = plt.subplots(figsize=figsize_text(1.0, height_ratio=0.6))
    else:
        _, ax = plt.subplots()

    df = load_stats_log(logfile)

    df.loc[(df.estimator == 'network') & (df.loss == 'shiftedmse'), 'estimator'] = 'networka'
    df[['true_pairs', 'pruned_ratio', 'pruned_domain']] *= 100

    # gdf = df.groupby(['estimator', 'adjustment'], as_index=False).mean()
    # gdf.sort_values(['true_pairs', 'pruned_ratio'], inplace=True)
    # gdf.plot.bar(x=['estimator', 'adjustment'], y=['pruned_domain'], ax=ax)
    # sns.boxplot('estimator', 'pruned_domain', hue='adjustment', data=df, ax=ax)
    sns.barplot('estimator', 'pruned_domain', hue='adjustment', data=df, ax=ax)
    ax.set_ylim([0, 100])
    # ax.yaxis.set_ticks(np.arange(0, 101, 5))
    ax.set_ylabel(LABELS['pruned_domain'])
    ax.set_xlabel(LABELS['estimator'])
    ax.legend(title=LABELS['adjustment'])
    ax.set_title('Number of instances for which the domain was pruned by boundary estimation')

    sns.despine(ax=ax)

    if outputdir:
        plt.savefig(os.path.join(outputdir, 'estimationbars.pgf'), dpi=500, bbox_inches='tight', pad_inches=0)


def boundaryeffects_table(files):
    df = solve.helper.read_stats(files)
    unbounded = (df['LowerBound'] == -1) & (df['UpperBound'] == -1)
    only_lower = (df['LowerBound'] != -1) & (df['UpperBound'] == -1)
    only_upper = (df['LowerBound'] == -1) & (df['UpperBound'] != -1)
    both_bounds = (df['LowerBound'] != -1) & (df['UpperBound'] != -1)

    df.loc[unbounded, 'Bounds'] = 'No'
    df.loc[only_lower, 'Bounds'] = 'Lower'
    df.loc[only_upper, 'Bounds'] = 'Upper'
    df.loc[both_bounds, 'Bounds'] = 'Both'

    if 'search_time':
        df.loc[df.Solver == 'chuffed', 'Runtime'] = df.loc[df.Solver == 'chuffed', 'search_time'] * 1000
        del df['search_time']

    assert (unbounded.sum() + only_lower.sum() + only_upper.sum() + both_bounds.sum() == len(df))

    x = df.groupby(['Problem', 'Instance', 'Solver']).transform(lambda x: x['UpperBound'] - x['LowerBound'])
    print(x)


def instances_table(problems, outputdir=None):
    fields = ['Problem', 'Instances']

    rows = []

    for p in problems:
        rows.append((p.name, len(p.get_dzns())))

    df = pd.DataFrame(rows, columns=fields)
    df.sort_values(['Instances', 'Problem'], ascending=[False, True], inplace=True)
    df = df.replace(LABELS.keys(), LABELS.values())

    if outputdir:
        df.to_latex(os.path.join(outputdir, 'instances.tex'), index=False)

    print(df)


def agg_mult_results(x):
    if len(x.status.unique() == 1):
        res = x.median()
        res['status'] = x[0]['status']
    else:
        res = x

    return res


def solver_performance_graph(bounded_logs, unbounded_logs, outputdir, combined_log=None):
    if combined_log:
        df = pd.read_csv(combined_log)
        df = join_on_common_tasks(df[df.bounds != 'No'], df[df.bounds == 'No'])
    else:
        df = join_on_common_tasks(pd.read_csv(bounded_logs), pd.read_csv(unbounded_logs))

    num_solvers = df.solver.nunique()

    if outputdir:
        _, axes = plt.subplots(nrows=1, ncols=num_solvers, figsize=figsize_text(1.0, height_ratio=0.6))
    else:
        _, axes = plt.subplots(nrows=num_solvers, ncols=1, sharex=True)

    df['mzn'] = df['mzn'].str.replace('_boundest.mzn', '.mzn')

    aggdf = df.reset_index()
    aggdf = aggdf[aggdf.status == 'COMPLETE'].groupby(['solver', 'mzn', 'bounds', 'time_solver']).count().groupby(
        level=[0, 1, 2]).cumsum().reset_index()

    legends = []
    num_mzns = df.mzn.nunique()

    for solver_ax, solver in zip(axes, sorted(df.solver.unique().tolist())):
        solver_ax.set_title(LABELS[solver])

        for c, p in zip(sns.color_palette("hls", num_mzns), df.mzn.unique().tolist()):
            df_filter = (aggdf.solver == solver) & (aggdf.mzn == p)

            print(p, solver, df_filter.sum())
            bnd = 'Both'

            if aggdf[df_filter & (aggdf.bounds == bnd)].shape[0] == 0 or \
                    aggdf[df_filter & (aggdf.bounds == 'No')].shape[0] == 0:
                continue

            l = aggdf[df_filter & (aggdf.bounds == bnd)].plot(x='time_solver', y='dzn', ax=solver_ax, linestyle='-',
                                                              c=c, label='{} ({})'.format(LABELS[p], bnd))
            aggdf[df_filter & (aggdf.bounds == 'No')].plot(x='time_solver', y='dzn', ax=solver_ax, linestyle='--',
                                                           c=c, label='{} (None)'.format(LABELS[p]))

            if solver == 'gecode':
                legends.append(mpatches.Patch(color=c, label=LABELS[p]))

        #        solver_ax.legend_.remove()
        solver_ax.set_xlabel('Time (in s)')
        solver_ax.set_xlim([0, 1200])

        if solver == 'chuffed':
            solver_ax.set_ylabel('Completed Instances')

    sns.despine()

    #    axes[1].legend()

    if outputdir:
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, 'solver.pgf'), dpi=500, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def solver_performance_table(bounded_logs, unbounded_logs, outputdir, combined_log=None):
    # Per-solver and problem
    # - No. Complete
    # - Avg. runtime of complete
    # - Avg. quality of incomplete
    if combined_log:
        df = pd.read_csv(combined_log)
        df = join_on_common_tasks(df[df.bounds != 'No'], df[df.bounds == 'No'])
    else:
        df = join_on_common_tasks(pd.read_csv(bounded_logs), pd.read_csv(unbounded_logs))

    cats = ['COMPLETE', 'SOLFOUND', 'UNSATISFIABLE', 'UNKNOWN', 'FAILED', 'INTERMEDIATE']
    df['status'] = df['status'].astype("category")  # .cat.reorder_categories(cats, ordered=True)
    df['status'].cat.set_categories(cats, ordered=True)
    df['status'] = df['status'].cat.as_ordered()

    df = df[(df.status != 'INTERMEDIATE')]

    completion_df = df.groupby(['bounds', 'mzn', 'status'])['status'].count().reset_index(name="count")
    print(completion_df.pivot_table(values='count', columns='bounds', index=['mzn', 'status']))

    # Qualitative difference
    baseline = data_loader.get_best_results(dzn_filter=df.dzn.unique().tolist())

    diff_keys = ['solver', 'mzn', 'dzn', 'bounds', 'status', 'objective', 'time_solver']
    if 'backtracks' in df.columns:
        diff_keys += ['backtracks']

    if 'propagations' in df.columns:
        diff_keys += ['propagations']

    if 'normal_propagations' in df.columns:
        diff_keys += ['normal_propagations']

    diff_df = df[diff_keys]
    # diff_df = diff_df[diff_df.solver != 'gecode']
    diff_df = diff_df.groupby(['solver', 'mzn', 'dzn'], as_index=False)
    diff_df = diff_df.apply(lambda g: any_improvement(g))
    objective_df = diff_df.groupby(['solver', 'mzn', 'objective']).count().reset_index().rename(
        columns={'objective': 'impact', 'time': 'objective'})
    time_df = diff_df.groupby(['solver', 'mzn', 'time']).count().reset_index().rename(
        columns={'time': 'impact', 'objective': 'time'})
    xdf = objective_df.set_index(['solver', 'mzn', 'impact']).join(time_df.set_index(['solver', 'mzn', 'impact']),
                                                                   how='outer')
    # xdf = xdf.rename(columns={'objective': 'time', 'time': 'objective'})
    xdf = xdf.groupby(['solver', 'mzn']).apply(lambda x: x / x.sum() * 100)

    xdf = xdf.pivot_table(index='mzn', columns=['solver', 'impact'], values=['objective', 'time'])
    xdf = xdf.swaplevel(0, 1, axis='columns').sort_index(axis='columns', level=0)
    xdf = xdf.fillna(0)
    xdf = xdf.astype(int)

    # xdf.replace(LABELS.keys(), LABELS.values(), inplace=True)
    print(xdf)

    if outputdir:
        xdf.rename(columns=LABELS, inplace=True)

        column_format = 'lrrrrrr' + '|rrrrrr' * (len(df.solver.unique()) - 2)
        output_string = xdf.to_latex(multicolumn=True, multicolumn_format='c', column_format=column_format,
                                     index_names=False)
        output_string = output_string.replace('\\midrule\n', '')
        output_string = output_string.replace('OR-Tools \\\\\n', 'OR-Tools \\\\\n\\midrule\n')
        output_string = output_string.replace('\\\nBin Packing', '\\\n\\midrule\nBin Packing')
        output_string = output_string.replace('\\\nAll', '\\\n\\midrule\nAll')

        if 'Both' in df.bounds.unique():
            filename = 'solver_improvement_o2_num.tex'
        else:
            filename = 'solver_improvement_o1_num.tex'

        open(os.path.join(outputdir, filename), 'w').write(output_string)

    return

    diff_df = df[diff_keys]  # .groupby(['solver', 'mzn', 'dzn', 'bounds'], as_index=False).agg(agg_mult_results)
    diff_df = diff_df.groupby(['solver', 'mzn', 'dzn'], as_index=False)
    diff_df = diff_df.apply(lambda g: quantitative_difference(g, baseline))

    rows_with_impact = (diff_df['objective_diff'].notnull() & diff_df['objective_diff'] > 0.0) | (
            diff_df['time_diff'].notnull() & diff_df['time_diff'] > 0.0)

    imp_df = diff_df[rows_with_impact].reset_index()
    all_df = imp_df[['mzn', 'solver', 'time_diff', 'objective_diff']].groupby(['solver', 'mzn']).mean()
    all_df['count'] = imp_df[['mzn', 'solver', 'time_diff', 'objective_diff']].groupby(['solver', 'mzn']).count()[
                          'time_diff'] / diff_df.groupby(['solver', 'mzn']).count()['time_diff']
    all_df[['count', 'time_diff', 'objective_diff']] *= 100
    all_df.reset_index(inplace=True)
    all_df.replace(LABELS.keys(), LABELS.values(), inplace=True)
    all_df = all_df.pivot_table(index='mzn', columns=['solver'], margins=False, aggfunc='median',
                                values=['count', 'objective_diff', 'time_diff'])
    all_df = all_df.swaplevel(0, 1, axis='columns').sort_index(axis='columns', level=0)

    all_df.rename(columns=LABELS, inplace=True)
    all_df.fillna(0, inplace=True)
    all_df = all_df.round().astype(int)

    if outputdir:
        out_df = all_df  # .reset_index()[['mzn', 'solver', 'anyofthem']]
        # out_df.replace(LABELS.keys(), LABELS.values(), inplace=True)
        # out_df = out_df.pivot_table(index='mzn', columns='solver', margins=True, values='anyofthem', margins_name='All')
        # out_df.fillna(0, inplace=True)
        out_df.rename(columns=LABELS, inplace=True)

        column_format = 'lrrr' + '|rrr' * (len(df.solver.unique()) - 1)
        output_string = out_df.to_latex(multicolumn=True, multicolumn_format='c', column_format=column_format,
                                        index_names=False)
        output_string = output_string.replace('\\midrule\n', '')
        output_string = output_string.replace('OR-Tools \\\\\n', 'OR-Tools \\\\\n\\midrule\n')
        output_string = output_string.replace('\\\nBin Packing', '\\\n\\midrule\nBin Packing')
        output_string = output_string.replace('\\\nAll', '\\\n\\midrule\nAll')

        if 'Both' in df.bounds.unique():
            filename = 'solver_improvement_o2.tex'
        else:
            filename = 'solver_improvement_o1.tex'

        open(os.path.join(outputdir, filename), 'w').write(output_string)

    total_time_unbound = df[df['bounds'] == 'No']['time_solver'].sum()
    total_time_bound = df[df['bounds'] == 'Upper']['time_solver'].sum()
    print('Total time saved: {} - {} = {}'.format(total_time_unbound, total_time_bound,
                                                  total_time_unbound - total_time_bound))

    print(all_df)
    all_df.to_clipboard()

    # More detailed descriptions
    if any(diff_df['solver_diff'] != 0.0):
        output_string = description_table(diff_df, 'solver_diff')
        print('Solver Diff.')
        print(output_string)
        wilcox_df = diff_df.groupby(['mzn']).apply(lambda x: wilcoxon(x['solver_diff']))
        print(wilcox_df)

    if any(diff_df['objective_diff'] != 0.0):
        output_string = description_table(diff_df, 'objective_diff')
        print('Objective Diff.')
        print(output_string)
        wilcox_df = diff_df.groupby(['mzn']).apply(lambda x: wilcoxon(x['objective_diff']))
        print(wilcox_df)

    output_string = description_table(diff_df, 'time_diff')
    print('Time Diff.')
    print(output_string)
    wilcox_df = diff_df.groupby(['mzn']).apply(lambda x: wilcoxon(x['time_diff']))
    print(wilcox_df)

    open(os.path.join(outputdir, 'solver_describe.tex'), 'w').write(output_string)


def description_table(diff_df, column):
    describe_df = diff_df[diff_df[column] != 0.0].groupby(['mzn']).describe()[column] * 100
    describe_df['count'] /= 100
    describe_df.fillna(0, inplace=True)
    describe_df = describe_df.round().astype(int)

    describe_df['Mean'] = describe_df[['mean', 'std']].apply(lambda x: '±'.join([str(y) for y in x]), axis=1)
    # del describe_df['count']
    del describe_df['mean']
    del describe_df['std']

    describe_df = describe_df.reset_index()
    describe_df = describe_df.replace(LABELS.keys(), LABELS.values())
    describe_df.rename(columns=LABELS, inplace=True)
    describe_df.columns = map(str.capitalize, describe_df.columns)
    output_string = describe_df.to_latex(index=False)
    output_string = output_string.replace('±', '\\(\\pm\\)')
    return output_string


def join_on_common_tasks(bounded_df, unbounded_df):
    common_task_ids = np.intersect1d(bounded_df.taskid.unique(), unbounded_df.taskid.unique())

    bounded_df = bounded_df[bounded_df.taskid.isin(common_task_ids)]
    unbounded_df = unbounded_df[unbounded_df.taskid.isin(common_task_ids)]

    return pd.concat([bounded_df, unbounded_df])


def any_improvement(group):
    unbounded = group[group.bounds == 'No']
    bounded = group[group.bounds != 'No']

    assert len(unbounded) == 1, "Wrong number of unbounded results"

    base_complete = unbounded.iloc[0]['status'] == 'COMPLETE'

    if base_complete and bounded.iloc[0]['status'] == 'COMPLETE':
        time = np.sign(unbounded.iloc[0]['time_solver'] - bounded.iloc[0]['time_solver'])
    elif not base_complete and any(bounded['status'] == 'COMPLETE'):
        time = 1
    elif base_complete and not any(bounded['status'] == 'COMPLETE'):
        time = -1
    else:
        time = 0

    if unbounded.iloc[0]['status'] == 'SOLFOUND' and bounded.iloc[0]['status'] == 'SOLFOUND':
        objective = np.sign(unbounded.iloc[0]['objective'] - bounded.iloc[0]['objective'])

        if np.isnan(objective):
            if np.isnan(unbounded.iloc[0]['objective']) and np.isnan(bounded.iloc[0]['objective']):
                objective = 0
            elif np.isnan(unbounded.iloc[0]['objective']):
                objective = 1
            else:
                objective = -1
    elif unbounded.iloc[0]['status'] in ('SOLFOUND', 'COMPLETE') and bounded.iloc[0]['status'] not in (
            'SOLFOUND', 'COMPLETE'):
        objective = -1
    else:
        objective = 0

    return pd.Series([int(time), int(objective)], index=['time', 'objective'], dtype=np.int)


def quantitative_difference(group, baseline):
    unbounded = group[group.bounds == 'No']
    bounded = group[group.bounds != 'No']

    if len(unbounded) > 1:
        unbounded = group[group.bounds == 'No'].groupby(['solver', 'mzn', 'dzn'], as_index=False).median()
        unbounded['status'] = group[group.bounds == 'No']['status'].min()

    if len(bounded) > 1:
        bounded = group[group.bounds != 'No'].groupby(['solver', 'mzn', 'dzn'], as_index=False).median()
        bounded['status'] = group[group.bounds != 'No']['status'].min()

    assert len(unbounded) == 1, "Wrong number of unbounded results ({}): {}".format(len(unbounded), bounded)
    assert len(bounded) == 1, "Wrong number of bounded results ({}): {}".format(len(bounded), unbounded)

    time_ratio = 0.0
    gap_diff = 0.0
    solver_diff = 0.0

    if all(unbounded.status == 'COMPLETE') and all(bounded.status == 'COMPLETE'):
        assert (all(unbounded['objective'].isnull()) and all(bounded['objective'].isnull())) or \
               all(unbounded['objective'].values == bounded['objective'].values), 'Complete with different objectives'

        unbounded_time = unbounded.iloc[0]['time_solver']
        bounded_time = bounded.iloc[0]['time_solver']

        if abs(unbounded_time - bounded_time) >= 5:
            time_ratio = (unbounded_time - bounded_time) / unbounded_time

        STAT_KEYS = {
            'choco': 'backtracks',
            'chuffed': 'propagations',
            'ortools': 'normal_propagations'
        }

        if unbounded.solver.values[0] in STAT_KEYS:
            stat_key = STAT_KEYS[unbounded.solver.values[0]]

            unbounded_stat = unbounded.iloc[0][stat_key]
            bounded_stat = bounded.iloc[0][stat_key]
            solver_diff = (unbounded_stat - bounded_stat) / unbounded_stat
    elif all(unbounded.status == 'SOLFOUND') and all(bounded['status'].isin(['SOLFOUND'])):
        mzn_filter = baseline.mzn == group.mzn.values[0]
        dzn_filter = baseline.dzn == group.dzn.values[0]
        optimum = min([baseline[mzn_filter & dzn_filter]['objective'].min(),
                       unbounded.iloc[0]['objective'],
                       bounded.iloc[0]['objective']])

        unbounded_gap = unbounded.iloc[0]['objective'] - optimum
        bounded_gap = bounded.iloc[0]['objective'] - optimum

        # assert unbounded_gap >= 0, "Unbounded better than prev. best: {}".format(group.dzn.values[0])
        # assert bounded_gap >= 0, "Bounded better than prev. best: {}".format(group.dzn.values[0])

        if unbounded_gap != 0:
            gap_diff = (unbounded_gap - bounded_gap) / unbounded_gap

        # if gap_diff < 0:
        #    print(group, optimum, gap_diff)

    return pd.Series([time_ratio, gap_diff, solver_diff],
                     index=['time_diff', 'objective_diff', 'solver_diff'],
                     dtype=np.float)


def loss_functions(outputdir=None):
    if outputdir:
        _, ax = plt.subplots(figsize=figsize_column(1.0, height_ratio=0.7))
    else:
        _, ax = plt.subplots()

    x = np.linspace(-2, 2, 300)

    # MSE
    ax.plot(x, x ** 2, color='k', linestyle='--', label='Symmetric Loss (Squared Error)')

    # Shifted MSE
    a = -0.8
    y = x ** 2 * np.power(np.sign(x) + a, 2)
    ax.plot(x, y, color='b', label='Asymmetric Loss (a = {:.1f})'.format(a))

    ax.set_ylabel('Loss')
    ax.set_yticklabels([])
    ax.yaxis.labelpad = -2
    ax.set_xlabel('Residual')
    ax.set_xticklabels([])
    ax.xaxis.labelpad = -2
    ax.set_xlim([x.min(), x.max()])
    ax.axvline(0, c='k', linestyle='--', linewidth=0.5, ymin=0, ymax=0.6)
    ax.legend(loc=1, frameon=True)
    ax.grid(False)

    sns.despine(ax=ax)

    if outputdir:
        plt.tight_layout()
        plt.savefig(os.path.join(outputdir, 'losses.pgf'), dpi=500, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()


def get_row_idx(df, mzn, dzn, solver):
    return (df.mzn == mzn) & (df.dzn == dzn) & (df.solver == solver)


def equivalent_solver_time(idx, df, soldf, mzn, dzn, solver):
    first_objective = df.loc[idx, 'first_objective'].values[0]
    first_sol_time = df.loc[idx, 'first_sol_time'].values[0]
    sol_idx = get_row_idx(soldf, mzn, dzn, solver) & (soldf.objective <= first_objective)
    unb_equal_time = soldf.loc[sol_idx, 'time'].dropna().min()

    if abs(unb_equal_time - first_sol_time) >= 1:
        est = 100 * (first_sol_time - unb_equal_time) / unb_equal_time
    else:
        est = 0

    return est


def solver_performance_tables_split(logfile, logfile_fixed, outputdir):
    df = pd.read_csv(logfile)
    df['origin'] = 'est'

    soldf = pd.read_csv(logfile.replace('.csv', '_solutions.csv'))
    soldf = pd.DataFrame(soldf[soldf.bounds == 'No'])

    dfno = pd.DataFrame(df[df.bounds == 'No'])
    dfno['boundstype'] = 'no'

    dfo1 = pd.DataFrame(df[(df.bounds == 'Upper') & (df.boundstype == 'hard')])
    dfo2 = pd.DataFrame(df[(df.bounds == 'Both') & (df.boundstype == 'hard')])

    dffixed = pd.read_csv(logfile_fixed)
    dffixed['origin'] = 'fixed'

    solfixed = pd.read_csv(logfile_fixed.replace('.csv', '_solutions.csv'))

    for s in df.solver.unique():
        res_base = get_result_columns(dffixed[dffixed.solver == s], dfno[dfno.solver == s], soldf[soldf.solver == s])
        res_base['experiment'] = 'fixed'

        res_o1 = get_result_columns(dfo1[dfo1.solver == s], dfno[dfno.solver == s], soldf[soldf.solver == s])
        res_o1['experiment'] = 'o1'

        res_o2 = get_result_columns(dfo2[dfo2.solver == s], dfno[dfno.solver == s], soldf[soldf.solver == s])
        res_o2['experiment'] = 'o2'

        result = pd.concat([res_base, res_o1, res_o2])
        result = result.replace(LABELS.keys(), LABELS.values())
        pdf = pd.pivot_table(result, index='mzn', values=['qof', 'est', 'ttc'], columns='experiment')
        pdf.rename(columns={
            'est': 'EST',
            'qof': 'QOF',
            'ttc': 'TTC',
            'fixed': 'Fixed',
            'o1': 'Upper',
            'o2': 'Both'
        }, inplace=True)
        output_string = pdf.to_latex(multicolumn=True, multicolumn_format='c',
                                     column_format='lRRR|RRR|RRR',
                                     index_names=False, escape=False)
        output_string = output_string.replace('\\begin{tabular}', '\\begin{tabularx}{0.97\\textwidth}')
        output_string = output_string.replace('\\end{tabular}', '\\end{tabularx}')
        print(pdf)
        #open(os.path.join(outputdir, 'solver_effects_%s.tex' % s), 'w').write(output_string)

    adf = df[['bounds', 'solver', 'mzn', 'first_objective']]
    adf['is_complete'] = df['status'] == 'COMPLETE'
    adf = adf.groupby(['bounds', 'solver', 'mzn']).agg({'first_objective': 'count', 'is_complete': 'sum'}) / 30 * 100
    adf = adf.pivot_table(values=['first_objective', 'is_complete'], index='mzn', columns=['solver', 'bounds'])
    adf = adf.round().astype(int)
    out = adf.to_latex(multicolumn=True, multicolumn_format='c', index_names=False)
    print(adf)
    open(os.path.join(outputdir, 'solver_hassol.tex'), 'w').write(out)


def get_result_columns(df, dfno, soldfno):
    df = pd.DataFrame(df)

    df['est'] = 0

    for mzn, dzn, solver in df.set_index(['mzn', 'dzn', 'solver']).index:
        hard_idx = get_row_idx(df, mzn, dzn, solver)
        df.loc[hard_idx, 'est'] = equivalent_solver_time(hard_idx, df, soldfno, mzn, dzn, solver)

    df.set_index(['mzn', 'dzn', 'solver'], inplace=True)
    dfno.set_index(['mzn', 'dzn', 'solver'], inplace=True)
    df['qof'] = np.round(100 * (df['first_objective'] - dfno['first_objective']) / dfno['first_objective'], 0)
    df['ttf'] = np.round(100 * (df['first_sol_time'] - dfno['first_sol_time']) / dfno['first_sol_time'], 0)
    df['cmpl_diff'] = (df['status'] == 'COMPLETE').astype(int) - (dfno['status'] == 'COMPLETE').astype(int)

    df_complete = df[df.status == 'COMPLETE'].join(dfno[dfno.status == 'COMPLETE'], rsuffix='_no')
    df['ttc'] = np.round(100 * (df_complete['time_solver'] - df_complete['time_solver_no']) / df_complete['time_solver_no'], 0)
    df.reset_index(inplace=True)
    return df[['mzn', 'qof', 'ttf', 'est', 'ttc']].groupby(['mzn'], as_index=False).mean().round(1)


def solver_performance_table_o2(logfile, outputdir):
    df = pd.read_csv(logfile)

    soldf = pd.read_csv(logfile.replace('.csv', '_solutions.csv'))
    soldf = pd.DataFrame(soldf[soldf.bounds == 'No'])

    dfno = pd.DataFrame(df[df.bounds == 'No'])
    dfno['boundstype'] = 'no'

    dfhard = pd.DataFrame(df[(df.bounds == 'Both') & (df.boundstype == 'hard')])
    dfsoft = pd.DataFrame(df[(df.bounds == 'Both') & (df.boundstype == 'soft')])

    dfhard['est'] = 0.
    dfsoft['est'] = 0.

    for mzn, dzn, solver in dfhard.set_index(['mzn', 'dzn', 'solver']).index:
        hard_idx = get_row_idx(dfhard, mzn, dzn, solver)
        dfhard.loc[hard_idx, 'est'] = equivalent_solver_time(hard_idx, dfhard, soldf, mzn, dzn, solver)

        soft_idx = get_row_idx(dfsoft, mzn, dzn, solver)
        if soft_idx.any():
            dfsoft.loc[soft_idx, 'est'] = equivalent_solver_time(soft_idx, dfsoft, soldf, mzn, dzn, solver)
        else:
            dfsoft.loc[soft_idx, 'est'] = 1200

    dfno.set_index(['mzn', 'dzn', 'solver'], inplace=True)

    dfhard.set_index(['mzn', 'dzn', 'solver'], inplace=True)
    dfhard['obj_diff'] = np.round(100 * (1 - (dfhard['first_objective'] / dfno['first_objective'])), 0)
    dfhard['time_diff'] = np.round(100 * (dfno['first_sol_time'] - dfhard['first_sol_time']) / dfno['first_sol_time'],
                                   0)

    dfsoft.set_index(['mzn', 'dzn', 'solver'], inplace=True)
    dfsoft['obj_diff'] = np.round(100 * (1 - (dfsoft['first_objective'] / dfno['first_objective'])), 0)
    dfsoft['time_diff'] = np.round(100 * (dfno['first_sol_time'] - dfsoft['first_sol_time']) / dfno['first_sol_time'],
                                   0)

    # Ignore time differences less than 1s
    dfhard.loc[(dfno['first_sol_time'] - dfhard['first_sol_time']).abs() < 1, "time_diff"] = 0
    dfsoft.loc[(dfno['first_sol_time'] - dfsoft['first_sol_time']).abs() < 1, "time_diff"] = 0

    labels = {
        'obj_diff': 'QOF',
        'time_diff': 'TTF',
        'est': 'EST',
        'hard': 'Boundary Constraints',
        'soft': 'Bounds-Aware Search'
    }

    def cust_aggfunc_star(x):
        m = np.median(x.dropna())
        std = np.ceil(robust.scale.mad(x.dropna())).astype(int)

        if std <= 5:
            appendix = ''
        elif std <= 10:
            appendix = '+'
        elif std <= 20:
            appendix = '*'
        elif std <= 30:
            appendix = '**'
        elif std <= 40:
            appendix = '***'
        else:
            appendix = '{:d}'.format(std)

        m = np.round(m).astype(int)
        return "{:d}\\textsuperscript{{{}}}".format(m, appendix)

    # mdf = pd.concat([dfhard.reset_index(), dfsoft.reset_index()])
    mdf = dfhard.reset_index()
    mdf.dropna(subset=['obj_diff', 'time_diff'], inplace=True)

    mdf.rename(columns=labels, inplace=True)
    mdf = mdf.replace(LABELS.keys(), LABELS.values())

    out_df = mdf.pivot_table(index='mzn', columns=['boundstype', 'solver'],
                             values=[labels['obj_diff'], labels['est']],  # , labels['time_diff']], #, 'est'],
                             aggfunc=lambda x: np.round(np.mean(x), 1),
                             margins=True,
                             margins_name='Average',
                             fill_value='-')
    out_df.columns = out_df.columns.swaplevel(0, 1)
    # out_df.sort_values(('Average', 'EST'), ascending=True, inplace=True)
    del out_df['Average']
    out_df.columns = out_df.columns.swaplevel(1, 2)
    out_df.sort_index(level=0, axis=1, inplace=True)

    output_string = out_df.to_latex(multicolumn=True, multicolumn_format='c',
                                    column_format='lrr|rr|rr|rr|rr|rr',
                                    index_names=False, escape=False)
    # output_string = output_string.replace('\\begin{tabular}', '\\begin{tabularx}{0.97\\textwidth}')
    # output_string = output_string.replace('\\end{tabular}', '\\end{tabularx}')

    open(os.path.join(outputdir, 'solver_o2.tex'), 'w').write(output_string)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['estimation', 'estimation2', 'boundaryeffect', 'instances', 'solver2',
                                           'estimationbars', 'adjustment', 'adjustmentg', 'solver', 'solverg',
                                           'losses', 'splittables'])
    parser.add_argument('-f', '--files', nargs='+')
    parser.add_argument('-n', '--name', default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    args = parser.parse_args()

    OUTPUTDIR = '/home/helge/Dropbox/Apps/ShareLaTeX/Boundary Estimation 2/figures/'
    # OUTPUTDIR = '.'

    if args.action == 'estimation':
        estimation_table(args.files, OUTPUTDIR)
    elif args.action == 'estimation2':
        estimation_table_o2(args.files, OUTPUTDIR)
    elif args.action == 'adjustment':
        adjustment_table(args.files, OUTPUTDIR)
    elif args.action == 'adjustmentg':
        # adjustment_graph(args.files, OUTPUTDIR, column='true_pairs')  # only relevant for 1 output
        adjustment_graph(args.files, OUTPUTDIR, column='pruned_domain')
    elif args.action == 'estimationbars':
        estimation_bars(args.files, OUTPUTDIR)
        plt.show()
    elif args.action == 'boundaryeffect':
        boundaryeffects_table(args.files)
    elif args.action == 'instances':
        instances_table(config.LARGE_PROBLEMS, OUTPUTDIR)
    elif args.action == 'solver':
        if len(args.files) == 2:
            solver_performance_table(args.files[0], args.files[1], OUTPUTDIR)
        elif len(args.files) == 1:
            solver_performance_table(None, None, OUTPUTDIR, combined_log=args.files[0])
        else:
            raise NotImplementedError("Unexpected number of input files")
    elif args.action == 'solver2':
        solver_performance_table_o2(args.files[0], OUTPUTDIR)
    elif args.action == 'solverg':
        if len(args.files) == 2:
            solver_performance_graph(args.files[0], args.files[1], OUTPUTDIR)
        elif len(args.files) == 1:
            solver_performance_graph(None, None, OUTPUTDIR, combined_log=args.files[0])
        else:
            raise NotImplementedError("Unexpected number of input files")
    elif args.action == 'splittables':
        if len(args.files) == 2:
            solver_performance_tables_split(args.files[0], args.files[1], OUTPUTDIR)
        else:
            raise NotImplementedError("Unexpected number of input files")
    elif args.action == 'losses':
        loss_functions(OUTPUTDIR)
