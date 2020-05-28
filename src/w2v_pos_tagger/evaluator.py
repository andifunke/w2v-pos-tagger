#!/usr/bin/env python3

"""
Evaluates spaCy and NLTK tagging based on Precision, Recall, and F1 score.
"""

import argparse
import json
from time import time

import numpy as np
import pandas as pd

from w2v_pos_tagger.constants import (
    PRED_TXT, PRED_TAG, GOLD_TXT, GOLD_TAG, TP, FP, FN, PREC, RECL, F1, TIGER, HDT, STTS, UNIV,
    SPACY, NLTK
)
from w2v_pos_tagger.corpus_analyser import get_tagset
from w2v_pos_tagger.dataio import (
    FORM, get_preprocessed_corpus, get_baseline_corpus, tprint, EVAL_DIR, get_svm_annotations
)


def baseline_args(argv=None) -> argparse.Namespace:
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
    parser.set_defaults(analyse=False)
    args = parser.parse_args(argv)
    return args


def svm_args(argv=None) -> argparse.Namespace:
    """
    Parses module-specific arguments. Solves argument dependencies
    and returns cleaned up arguments.

    :returns: arguments object
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('--corpus', type=str, default=HDT, choices=[HDT, TIGER])
    args = parser.parse_args(argv)
    return args


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

    return classes


def save_classes(classes, name):
    file_path = (EVAL_DIR / name).with_suffix('.csv')
    print(f"Writing detailed results to {file_path}\n")
    EVAL_DIR.mkdir(exist_ok=True)
    classes.to_csv(file_path, sep='\t', float_format='%.3f')


def summarize_score(classes, corpus, name):
    tprint(classes)

    accuracy = classes.loc['sum', PREC]
    precision = classes.loc['weighted avg', PREC]
    recall = classes.loc['weighted avg', RECL]
    f1_weighted = classes.loc['weighted avg', F1]

    scores = dict()
    scores['corpus'] = corpus
    scores['test_size'] = int(classes.loc['sum', 'Count'])
    scores['accuracy'] = round(accuracy, 3)
    scores['precision'] = round(precision, 3)
    scores['recall'] = round(recall, 3)
    scores['f1_weighted'] = round(f1_weighted, 3)

    print(
        f"Accuracy:           {classes.loc['sum', PREC]:.3f}\n"
        f"Weighted Precision: {classes.loc['weighted avg', PREC]:.3f}\n"
        f"Weighted Recall:    {classes.loc['weighted avg', RECL]:.3f}\n"
        f"Weighted F1 score:  {classes.loc['weighted avg', F1]:.3f}\n"
    )

    save_path = (EVAL_DIR / name).with_suffix('.json')
    print('Writing summary to', save_path)
    with open(save_path, 'w') as f:
        json.dump(scores, f, indent=2)


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


def baseline(argv=None):
    """
    Evaluates the tagging with several metrics. Each corpus for each framework.
    STTS and Universal tagsets can be evaluated.
    """

    args = baseline_args(argv)
    t0 = time()

    for corpus in [TIGER, HDT]:
        # --- load ground truth ---
        gold = get_preprocessed_corpus(corpus)

        for framework in [SPACY, NLTK]:
            sample_size = 0
            # --- load predictions ---
            pred = get_baseline_corpus(corpus, framework, show_sample=sample_size)

            for tagset in [STTS, UNIV]:
                df = concat(pred, gold, tagset)

                if args.analyse:
                    analyse_tagset(pred, corpus, framework, tagset)
                    break

                print(
                    f">>> Evaluating Accuracy, Precision, Recall and F_1 measure for "
                    f"{corpus}/{framework}/{tagset}.\n"
                )
                classes = evaluate(df)

                name = f'{corpus}_{framework}_{tagset}'
                save_classes(classes, name)
                summarize_score(classes, corpus, name=name)

    print(f"All done in {time()-t0:.2f}s")


def svm(argv=None):
    args = svm_args(argv)
    t0 = time()

    model = args.model
    corpus = args.corpus
    pred = get_svm_annotations(model, corpus)

    # --- load ground truth ---
    gold = get_preprocessed_corpus(args.corpus)

    tagset = UNIV
    df = concat(pred, gold[:len(pred)], tagset)

    print(
        f">>> Evaluating Accuracy, Precision, Recall and F_1 measure of "
        f"model {model} on corpus {corpus}.\n"
    )
    classes = evaluate(df)

    name = f'{corpus}_{model}'
    save_classes(classes, name)
    summarize_score(classes, corpus, name=name)

    print(f"All done in {time()-t0:.2f}s")
