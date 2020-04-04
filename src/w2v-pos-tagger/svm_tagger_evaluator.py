#!/usr/bin/env python3

import json
import re
from os import listdir

import matplotlib.pyplot as plt

from data_loader import *
from svm_tagger_train import get_options

# making stuff more human readable
AXES = {
    'f1_micro': 'F_1 micro score',
    'f1_macro': 'F_1 macro score',
    'f1_weighted': 'F_1 weighted score',
    'train_time': 'Training time in seconds',
    'test_time': 'Test time in seconds',
}
LABELS = {
    'cb': 'CBOW',
    'sg': 'Skip-gram',
    -1: 'no limit',
    'train_size': 'train-size',
    'test_size': 'test-size',
}
TITLES = {
    'embedding_model': 'Word2Vec model',
    'embedding_size': 'Dimensionality of embedding',
    'max_iter': 'Limit training to maximum number of iterartions (max_iter)',
    'scale': 'Normalizing feature space before training and testing',
    'test_size': 'Size of the Test set',
    'train_size': 'Size of the Training set',
    'C': 'Value of parameter C',
    'lowercase': 'Using lowercase w2v-embeddings',
    'shrinking': 'Using the shrinking heuristic',
}
# and adding some fancy colors for the scatter plots
INDEXES = {
    'embedding_size': (12, 25, 50, 100),
    'embedding_model': ('sg', 'cb'),
    'max_iter': (-1, 1000),
    'scale': (False, True),
    'test_size': (2000000, 4853410),
    'train_size': (100000, 400000, 888237),
    'C': (1, 2, 1000),
    'lowercase': (False, True),
    'shrinking': (False, True),
}
COLORS = ('red', 'green', 'blue', 'yellow', 'orange', 'violet')
MARKERS = ('o', '^', 's', 'p')


def plot_group(df, columns, x='train_time', y='f1_micro', ylim=(0, 1), show=False, log=False):
    """
    creates a scatter plot for a given key usually with the train/test time on the x axis
    and the f1_micro measure on the y axis. You can give in two keys at the same time
    (for example combined with the train_size) which will plot with a different marker shape
    for the second key.
    """
    fname = path.join('figures', '{}__{}__{}'.format('_'.join(columns), x, y))
    multidim = len(columns) > 1

    fig, ax = plt.subplots()
    for key, group in df.groupby(columns):
        # define the label for the legend
        if multidim:
            label1 = LABELS[key[0]] if key[0] in LABELS else str(key[0])
            column2 = LABELS[columns[1]] if columns[1] in LABELS else str(columns[1])
            label2 = LABELS[key[1]] if key[1] in LABELS else str(key[1])
            label = '{} | {}: {}'.format(label1, column2, label2)
        else:
            label = LABELS[key] if key in LABELS else key

        # plot each group
        if multidim:
            color = COLORS[INDEXES[columns[0]].index(key[0])]
            marker = MARKERS[INDEXES[columns[1]].index(key[1])]
        else:
            color = COLORS[INDEXES[columns[0]].index(key)]
            marker = None
        group.plot.scatter(ax=ax, x=x, y=y, label=label, color=color, marker=marker, s=7)

    plt.xlabel(AXES[x])
    plt.ylabel(AXES[y])
    plt.title(TITLES[columns[0]] if columns[0] in TITLES else columns[0])
    plt.legend(loc=4)

    print('saving', fname)
    ax.set_autoscaley_on(False)
    ax.set_ylim(ylim)
    fig.savefig(fname + '.pdf')
    if log:
        # saving also a log scaled x-axis
        ax.set_xscale('log')
        fig.savefig(fname + '__log.pdf')
    if show:
        plt.show()


def aggregate_group(dfs, df_descriptions, keys, to_latex=False):
    """
    dfs can be a single DataFrame or list/tuple of DataFrame.
    prints out some interesting aggregations for given keys.
    """
    print('********************************************************************************************************')
    print('AGGREGATIONS FOR KEYED GROUP:', keys)
    print()

    if isinstance(dfs, pd.DataFrame):
        df_list = [dfs]
    else:
        df_list = dfs

    for df, df_d in zip(df_list, df_descriptions):
        print('on data set: ', df_d)
        group = df.groupby(keys)

        print('row of maximum for f1_micro per group:')
        # get the row with the maximum value of 'f1_micro' per group
        df_f1micmax_for_group = df.loc[group['f1_micro'].idxmax()]
        # reset the index
        df_f1micmax_for_group.set_index(keys, inplace=True)
        tprint(df_f1micmax_for_group.sort_values('f1_micro', ascending=False), to_latex=to_latex)

        print('maximum of all columns - sorted by f1_micro:')
        # to avoid missing keys
        columns_full = ['f1_micro', 'f1_macro', 'f1_weighted', 'train_time', 'train_size', 'test_time']
        columns_reduced = []
        for column in columns_full:
            if column not in keys:
                columns_reduced.append(column)
        df_max = group.max()[columns_reduced].sort_values('f1_micro', ascending=False)
        tprint(df_max, to_latex=to_latex)

        print('mean of all columns - sorted by f1_micro:')
        tprint(group.mean()[columns_reduced].sort_values('f1_micro', ascending=False), to_latex=to_latex)

        print('minimum train and test time - sorted by train_time:')
        columns = ['train_time', 'test_time']
        tprint(group.min()[columns].sort_values('train_time', ascending=True), to_latex=to_latex)

    print('********************************************************************************************************')


def svm_evaluator_main(directory=None):
    """
    a given directory overwrites the defaults. The function will look for all test-result files in this directory.
    """
    print('start evaluating')
    dir_name = CORPORA_DIR if directory is None else directory
    files = [f for f in listdir(dir_name) if re.match(r'.*_testresults_.*\.json$', f)]

    # loading test-results
    results = []
    for fname in files:
        result = json.load(open(path.join(dir_name, fname)))
        fname_short = re.sub(r'_testresults_.*\.json$', '', fname)
        result['model'] = fname_short

        if not all(x in result for x in ['train_time', 'test_time']):
            options = json.load(open(path.join(dir_name, fname_short + '_options.json')))

            # this is to compensate a bug during testing: the testing time is saved under the same key as the
            # training time. Therefore we reconstruct the training time from the options. If both are equal
            # (possible for early tests) than it is considered to be the training time and test time is set to nan.
            if 'train_time' not in result:
                result['train_time'] = options['time']
            if 'test_time' not in result:
                if options['time'] == result['time']:
                    result['test_time'] = float('nan')
                else:
                    result['test_time'] = result['time']
            del result['time']

        results.append(result)

    # putting everything in a nice DataFrame
    df = pd.DataFrame(results)
    df = df[['f1_micro', 'f1_macro', 'f1_weighted',
             'train_time', 'train_size',
             'test_time', 'test_size',
             'embedding_size', 'embedding_model', 'lowercase', 'scale',
             'C', 'max_iter',
             # 'shrinking',
             'model', ]]
    df['train_size'].replace(to_replace={0: 888237}, inplace=True)
    df['test_size'].replace(to_replace={0: 4853410}, inplace=True)
    tprint(df.sort_values('f1_micro', ascending=False))

    # since the max_iter parameter did a really bad job, we want to exclude those results from our aggregations
    df_nolimit = df[df.max_iter == -1]
    df_100k = df_nolimit[df_nolimit.train_size == 100000]
    df_400k = df_nolimit[df_nolimit.train_size == 400000]
    df_888k = df_nolimit[df_nolimit.train_size == 888237]
    # tprint(df_nolimit.sort_values('f1_micro', ascending=False))

    dfs = [df_nolimit, df_100k, df_400k, df_888k]
    dfs_descriptions = ['all training set sizes, max_iter: no limit',
                        'training: size: 100000, max_iter: no limit',
                        'training: size: 400000, max_iter: no limit',
                        'training: size: 888237, max_iter: no limit',
                        ]

    aggregate_group(dfs[0], [dfs_descriptions[0]], ['train_size'])
    # aggregate(df_nolimit, ['test_size'])  # of little interest - most tests were done on the full corpus
    additional_keys = []  # ['train_size']

    for c in [['embedding_size'], ['embedding_model'], ['lowercase'], ['scale'], ['C']]:
        aggregate_group(dfs, dfs_descriptions, c + additional_keys)

    aggregate_group([df], ['full data'], ['max_iter'])

    # for plotting we keep the bad performing max_iter items
    plot = True
    show = False
    data = df_nolimit
    ylim = [0.5, 1.0]
    additional_keys = ['train_size']
    if plot:
        for x in ['train_time', 'test_time']:
            plot_group(data, ['train_size'], x=x, show=show, ylim=ylim)
            # plot_group(data, 'test_size', x=x, show=show, ylim=ylim)
            plot_group(data, ['embedding_size'] + additional_keys, x=x, show=show, ylim=ylim)
            plot_group(data, ['embedding_model'] + additional_keys, x=x, show=show, ylim=ylim)
            plot_group(data, ['lowercase'] + additional_keys, x=x, show=show, ylim=ylim)
            plot_group(data, ['scale'] + additional_keys, x=x, show=show, ylim=ylim)
            plot_group(data, ['C'] + additional_keys, x=x, show=show, ylim=ylim)
            # plot_group(data, 'shrinking', x=x, show=show, ylim=ylim)
            plot_group(df, ['max_iter'] + additional_keys, x=x, show=show)


if __name__ == '__main__':
    OPTIONS = get_options()
    svm_evaluator_main(directory=OPTIONS['model_dir'])
