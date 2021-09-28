import argparse
import seaborn as sns
import pandas as pd
import numpy as np
import config
import os
from data_loader import get_baseline

def figsize_column(scale, height_ratio=1.0):
    fig_width_pt = 244  # Get this from LaTeX using \the\columnwidth
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale  # width in inches
    fig_height = fig_width * golden_mean * height_ratio  # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size


def figsize_text(scale, height_ratio=1.0):
    fig_width_pt = 506  # Get this from LaTeX using \the\textwidth
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
    "figure.figsize": figsize_column(1.0),  # default fig size of 0.9 textwidth
    "pgf.preamble": [
        r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts becasue your computer can handle it :)
        r"\usepackage[T1]{fontenc}",  # plots will be generated using this preamble
    ]
}

sns.set_style("whitegrid", pgf_with_latex)
sns.set_context("paper")
import matplotlib.pyplot as plt


def problem_result_distribution(problem, df=None, ax=None):
    if not df:
        df = get_baseline(problem)

    if not ax:
        fig, ax = plt.subplots(figsize=(20,10))

    rdf = df[(df.Problem == problem.name) & (df.Objective.notnull())][['DZN', 'Objective']]
    rdf.groupby('DZN').min().plot.hist(ax=ax)
    ax.set_title(problem.name.title())
    #sns.distplot(rdf.groupby('DZN', as_index=False).min())


def learning_curves(history, problem, filename=None):
    datasets = list(history.keys())
    metrics = history[datasets[0]].history.keys()

    fig, axes = plt.subplots(len(metrics), 1, figsize=(20,10))
    fig.suptitle(problem.name.capitalize())

    for ax, k in zip(axes, metrics):
        for ds, hist in history.items():
            x = hist.epoch
            y = hist.history[k]
            ax.plot(x, y, label='DS %d' % ds)
        ax.set_title(k.title())

    if filename:
        plt.tight_layout()
        plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
        #plt.savefig(filename + '.pgf', dpi=600, bbox_inches='tight')
    else:
        plt.show()


def prediction_scatter_plot(prediction, problem, ax=None, filename=None):
    if not ax:
        fig, ax = plt.subplots(figsize=(20,10))
        ax.set_title(problem.name.title())

    _, predicted, truth = zip(*prediction)

    outputs = len(predicted[0])

    if outputs == 1:
        if problem.minmax == 'min':
            markers = ['v']
        else:
            markers = ['^']
    else:
        markers = ['^', 'v']

    for dim, m in zip(range(prediction.shape[1]), markers):
        ax.scatter(truth, predicted[:, dim], marker=m)

    ax.plot([min(truth), max(truth)], [min(truth), max(truth)], 'k--', lw=2)
    ax.set_xlabel('Truth')
    ax.set_ylabel('Prediction')

    if filename:
        plt.tight_layout()
        plt.savefig(filename + '.png', dpi=300, bbox_inches='tight')
        #plt.savefig(filename + '.pgf', dpi=600, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['show', 'save'])
    parser.add_argument('figure', choices=['result_distribution', 'losses'])
    args = parser.parse_args()

    if args.figure == 'result_distribution':
        for p in config.PROBLEMS:
            print(p.name)
            problem_result_distribution(p)
            filename = os.path.join(config.OUTPUT_DIR, 'figures', 'result_distribution_%s' % p.name)

            if args.action == 'save':
                plt.tight_layout()
                #plt.savefig(filename + '.png', dpi=500, bbox_inches='tight')
                plt.savefig(filename + '.pgf', dpi=600, bbox_inches='tight', pad_inches=0)
            else:
                plt.show()


