#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
from constants import HDT, UNIV
from dataio import tprint, MODELS_DIR, IMG_DIR

from w2v_pos_tagger.svm_tagger import load_model

AXES = {
    'precision': 'Precision',
    'recall': 'Recall',
    'f1_weighted': 'F_1 weighted',
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
    'max_iter': 'Limit training to maximum number of iterations (max_iter)',
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


def plot_group(df, columns, x='train_time', y='f1_weighted', ylim=(0, 1), show=False, log=False):
    """
    Creates a scatter plot for a given key usually with the train/test time on the x axis
    and the f1_weighted measure on the y axis. You can give in two keys at the same time
    (for example combined with the train_size) which will plot with a different marker shape
    for the second key.
    """

    multidimensional = len(columns) > 1

    fig, ax = plt.subplots()
    for key, group in df.groupby(columns):
        try:
            # define the label for the legend
            if multidimensional:
                label1 = LABELS[key[0]] if key[0] in LABELS else str(key[0])
                column2 = LABELS[columns[1]] if columns[1] in LABELS else str(columns[1])
                label2 = LABELS[key[1]] if key[1] in LABELS else str(key[1])
                label = '{} | {}: {}'.format(label1, column2, label2)
            else:
                label = LABELS[key] if key in LABELS else key

            # plot each group
            if multidimensional:
                color = COLORS[INDEXES[columns[0]].index(key[0])]
                marker = MARKERS[INDEXES[columns[1]].index(key[1])]
            else:
                color = COLORS[INDEXES[columns[0]].index(key)]
                marker = None
        except ValueError:
            continue

        group.plot.scatter(ax=ax, x=x, y=y, label=label, color=color, marker=marker, s=7)

    plt.xlabel(AXES[x])
    plt.ylabel(AXES[y])
    plt.title(TITLES[columns[0]] if columns[0] in TITLES else columns[0])
    plt.legend(loc=4)

    file_name = str(IMG_DIR / f"{'_'.join(columns)}__{x}__{y}")
    IMG_DIR.mkdir(exist_ok=True, parents=True)
    print('saving', file_name)
    ax.set_autoscaley_on(False)
    ax.set_ylim(ylim)
    fig.savefig(file_name + '.pdf')
    if log:
        # saving also a log scaled x-axis
        ax.set_xscale('log')
        fig.savefig(file_name + '__log.pdf')
    if show:
        plt.show()


def aggregate_group(dfs, df_descriptions, keys, to_latex=False):
    """
    dfs can be a single DataFrame or list/tuple of DataFrame.
    Prints out aggregations for the given keys.
    """

    width = 200
    print('*' * width)
    print('AGGREGATIONS FOR KEYED GROUP:', keys)
    print()

    if isinstance(dfs, pd.DataFrame):
        df_list = [dfs]
    else:
        df_list = dfs

    for df, df_d in zip(df_list, df_descriptions):
        print('on data set: ', df_d)
        group = df.groupby(keys)

        print('row of maximum for f1_weighted per group:')
        # get the row with the maximum value of 'f1_weighted' per group
        df_f1_for_group = df.loc[group['f1_weighted'].idxmax()]
        # reset the index
        df_f1_for_group.set_index(keys, inplace=True)
        tprint(
            df_f1_for_group.sort_values('f1_weighted', ascending=False),
            to_latex=to_latex
        )

        print('maximum of all columns - sorted by f1_weighted:')

        # to avoid missing keys
        columns_full = [
            'precision', 'recall', 'f1_weighted', 'train_time', 'train_size', 'test_time'
        ]
        columns_reduced = []
        for column in columns_full:
            if column not in keys:
                columns_reduced.append(column)

        df_max = group.max()[columns_reduced].sort_values('f1_weighted', ascending=False)
        tprint(
            df_max,
            to_latex=to_latex
        )

        print('mean of all columns - sorted by f1_weighted:')
        tprint(
            group.mean()[columns_reduced].sort_values('f1_weighted', ascending=False),
            to_latex=to_latex
        )

        print('minimum train and test time - sorted by train_time:')
        columns = ['train_time', 'test_time']
        tprint(group.min()[columns].sort_values('train_time', ascending=True), to_latex=to_latex)

    print('*' * width)


def main():
    """Plots metric files in ``out/models/``. Plotting is restricted to HDT/Universal tagset."""

    dir_name = MODELS_DIR
    dirs = [d for d in dir_name.iterdir() if d.is_dir()]

    # loading test-results
    results = []
    for model_path in dirs:
        _, _, _, config, metrics = load_model(model_path)
        try:
            result = metrics[HDT][UNIV]
            result['train_size'] = metrics.get('train_size')
            result['train_time'] = metrics.get('train_time')
            result['test_size'] = metrics[HDT].get('test_size')
            result['test_time'] = metrics[HDT].get('test_time')
            result['embedding_size'] = config['dimensionality']
            result['embedding_model'] = config['architecture']
            result['lowercase'] = config['lowercase']
            result['scale'] = config['scale']
            result['C'] = config['C']
            result['max_iter'] = config['max_iter']
            result['shrinking'] = config['shrinking']
            result['model'] = config['model']
            results.append(result)
        except Exception as e:
            print(e)

    if not results:
        print(f"No evaluation files found in {dir_name}")
        exit()

    # putting everything in a nice DataFrame
    df = pd.DataFrame(results)
    tprint(df)

    df = df[[
        'accuracy',
        'precision',
        'recall',
        'f1_weighted',
        'train_time',
        'train_size',
        'test_time',
        'test_size',
        'embedding_size',
        'embedding_model',
        'lowercase',
        'scale',
        'C',
        'max_iter',
        'model',
        'shrinking',
    ]]
    df['train_size'].replace(to_replace={0: 888237}, inplace=True)
    df['test_size'].replace(to_replace={0: 4853410}, inplace=True)
    tprint(df.sort_values('f1_weighted', ascending=False))

    # since the max_iter parameter did not perform well, it is excluded from the aggregations
    df_no_limit = df[df.max_iter == -1]
    df_100k = df_no_limit[df_no_limit.train_size == 100000]
    df_400k = df_no_limit[df_no_limit.train_size == 400000]
    df_888k = df_no_limit[df_no_limit.train_size == 888237]

    dfs = [df_no_limit, df_100k, df_400k, df_888k]
    dfs_descriptions = [
        'all training set sizes, max_iter: no limit',
        'training: size: 100000, max_iter: no limit',
        'training: size: 400000, max_iter: no limit',
        'training: size: 888237, max_iter: no limit',
    ]

    aggregate_group(dfs[0], [dfs_descriptions[0]], ['train_size'])
    additional_keys = []

    for c in [['embedding_size'], ['embedding_model'], ['lowercase'], ['scale'], ['C']]:
        aggregate_group(dfs, dfs_descriptions, c + additional_keys)

    aggregate_group([df], ['full data'], ['max_iter'])

    # for plotting we keep the bad performing max_iter items
    plot = True
    show = False
    data = df_no_limit
    ylim = [0.5, 1.0]
    additional_keys = ['train_size']
    if plot:
        for x in ['train_time', 'test_time']:
            plot_group(data, ['train_size'], x=x, show=show, ylim=ylim)
            plot_group(data, ['embedding_size'] + additional_keys, x=x, show=show, ylim=ylim)
            plot_group(data, ['embedding_model'] + additional_keys, x=x, show=show, ylim=ylim)
            plot_group(data, ['lowercase'] + additional_keys, x=x, show=show, ylim=ylim)
            plot_group(data, ['scale'] + additional_keys, x=x, show=show, ylim=ylim)
            plot_group(data, ['C'] + additional_keys, x=x, show=show, ylim=ylim)
            plot_group(df, ['max_iter'] + additional_keys, x=x, show=show)


if __name__ == '__main__':
    main()
