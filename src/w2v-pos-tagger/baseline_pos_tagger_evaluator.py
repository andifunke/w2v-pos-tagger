#!/usr/bin/env python3

"""
Evaluates spaCy and NLTK tagging based on Precision, Recall, and F1 score.
"""

import argparse
from time import time

import numpy as np
import pandas as pd

from constants import (
    PRED_TXT, PRED_TAG, GOLD_TXT, GOLD_TAG, TP, FP, FN, PREC, RECL, F1, TIGER, HDT, STTS, UNIV,
    SPACY, NLTK
)
from corpora_analyser import get_tagset
from data_loader import FORM, get_preprocessed_corpus, get_selftagged_corpus, tprint, OUT_DIR


def concat(predictions, reference, tagset):
    """Concatenates ground truth and predictions."""

    try:
        assert len(predictions) == len(reference)
    except AssertionError as e:
        e.args += f'\nprediction length={len(predictions)}, gold length={len(reference)}'
        print(predictions[FORM].tail(3))
        print(reference[FORM].tail(3))
        raise

    df = pd.concat(
        [predictions[FORM], predictions[tagset], reference[FORM], reference[tagset]],
        axis=1, keys=[PRED_TXT, PRED_TAG, GOLD_TXT, GOLD_TAG]
    )
    assert (df[PRED_TXT] == df[GOLD_TXT]).all()

    return df


def calculate_metrics(row):
    """Calculates Precision, Recall and F1 score per row."""

    # Define Precision and Recall as 1 if the denominator is 0.
    precision = 1 if (row[TP] + row[FP]) == 0 else row[TP] / (row[TP] + row[FP])
    recall = 1 if (row[TP] + row[FN]) == 0 else row[TP] / (row[TP] + row[FN])

    # Simplified F_1 measure because of beta == 1. Define F1 as 0 if the denominator is 0.
    f1 = 0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)

    return pd.Series([precision, recall, f1])


def evaluate(df):
    """Evaluates a corpus/framework/tagset combination based in precision, recall and F1 score."""

    df['correct'] = df.PRED_TAG == df.GOLD_TAG

    gold_grp = df.groupby(GOLD_TAG)
    pred_grp = df.groupby(PRED_TAG)
    count = gold_grp[GOLD_TAG].count()
    tp = gold_grp['correct'].sum()
    fp = pred_grp['correct'].count() - pred_grp['correct'].sum()
    fn = count - tp

    classes = pd.concat([count, tp, fp, fn], keys=['Count', TP, FP, FN], axis=1)
    classes = classes.fillna(0.)

    # calculate Precision, Recall, F1
    classes[[PREC, RECL, F1]] = classes.apply(calculate_metrics, axis=1)

    # calculating sum (for accuracy) and weighted average
    cl_mean = classes.mean()
    cl_wavg = np.average(
        classes.loc[:, [PREC, RECL, F1]].values, weights=classes['Count'].values, axis=0
    )
    classes.loc['sum'] = classes.sum()
    classes.loc['sum', [PREC, RECL, F1]] = calculate_metrics(classes.loc['sum']).to_list()
    classes.loc['weighted avg'] = cl_mean
    classes.loc['weighted avg', [PREC, RECL, F1]] = cl_wavg
    classes = classes.astype({'Count': 'int', TP: 'int', FP: 'int', FN: 'int'})

    # TODO:

    return classes


def analyse_tagset(df, corpus, framework, tagset):
    """Analyses a given tagset."""

    print(">>> Analysing given tagsets.")

    tagset = get_tagset(df, tagset)
    print(
        f'corpus: {corpus}\n'
        f'framework: {framework}\n'
        f'tagset: {sorted(tagset.keys())}\n'
        f'size: {len(tagset)}\n'
    )


def parse_args() -> argparse.Namespace:
    """
    Parses module-specific arguments. Solves argument dependencies
    and returns cleaned up arguments.

    :returns: arguments object
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--analyse', action='store_true',
        help="Analyse the tagging instead of calculating metrics."
    )
    args = parser.parse_args()
    return args


def main():
    """
    Evaluates the tagging with several metrics. Each corpus for each framework.
    STTS and Universal tagsets can be evaluated.
    """

    args = parse_args()

    t0 = time()
    for corpus in [TIGER, HDT]:
        # --- load ground truth ---
        gold = get_preprocessed_corpus(corpus)

        for framework in [SPACY, NLTK]:
            sample_size = 0
            # --- load predictions ---
            pred = get_selftagged_corpus(corpus, framework, show_sample=sample_size)

            for tagset in [STTS, UNIV]:
                df = concat(pred, gold, tagset)

                if args.analyse:
                    analyse_tagset(pred, corpus, framework, tagset)
                    break

                print(
                    f">>> Evaluating Accuracy, Precision, Recall and F_1 measure for "
                    f"{corpus}/{framework}/{tagset}.\n"
                )
                results = evaluate(df)

                tprint(results)
                print(f"Accuracy:           {results.loc['sum', PREC]:.3f}")
                print(f"Weighted Precision: {results.loc['weighted avg', PREC]:.3f}")
                print(f"Weighted Recall:    {results.loc['weighted avg', RECL]:.3f}")
                print(f"Weighted F1 score:  {results.loc['weighted avg', F1]:.3f}")
                print()

                file_path = OUT_DIR / f'{corpus}_{framework}_{tagset}_results.csv'
                print(f"Writing to {file_path}")
                results.to_csv(file_path, sep='\t', float_format='%.3f')
                print()

    print(f"All done in {time()-t0:.2f}s")


if __name__ == '__main__':
    main()
