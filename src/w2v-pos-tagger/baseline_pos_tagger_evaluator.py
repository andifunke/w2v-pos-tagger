#!/usr/bin/env python3

from time import time

import pandas as pd

from data_loader import tprint, OUT_DIR, get_preprocessed_corpus, get_selftagged_corpus
from constants import (
    SPACY, NLTK, TIGER, HDT, FORM, SELF_TXT, SELF_TAG, GOLD_TXT, GOLD_TAG, TP,
    FP, FN, PREC, RECL, F1, STTS, UNIV, TN
)


def concat_df(selftagged, reference):
    print(">>> concatenating DataFrames.")

    try:
        assert len(selftagged) == len(reference)
    except AssertionError as e:
        e.args += '\nselftagged length={:d}, gold length={:d}'.format(len(selftagged), len(reference)),
        print(selftagged[FORM].tail(3))
        print(reference[FORM].tail(3))
        raise

    df = pd.concat([selftagged[FORM], selftagged[TAGSET], reference[FORM], reference[TAGSET]],
                   axis=1, keys=[SELF_TXT, SELF_TAG, GOLD_TXT, GOLD_TAG])
    print('...')
    tprint(df.tail(100))
    return df


def calculate_metrics(row):
    # define Precision and Recall as 1 if the denominator is 0
    row[PREC] = 1 if (row[TP] + row[FP]) == 0 else row[TP] / (row[TP] + row[FP])
    row[RECL] = 1 if (row[TP] + row[FN]) == 0 else row[TP] / (row[TP] + row[FN])
    # simplified F_1 measure because of beta == 1
    # define F1 as 0 if the denominator is 0
    row[F1] = 0 if (row[PREC] + row[RECL]) == 0 else (2 * row[PREC] * row[RECL]) / (row[PREC] + row[RECL])


def evaluate(df):
    count_tn = False
    metrics = [TP, TN, FP, FN, PREC, RECL, F1]
    if not count_tn:
        metrics.remove(TN)

    # initialize DataFrame for tags x metrics
    classes = pd.DataFrame(columns=metrics, index=sorted(TAGSET_TAGS))
    classes.fillna(0.0, inplace=True)

    # slow! probably due to iterrows() and testing term-equality...
    # try: Simply converting from the pandas representation to a NumPy representation via the
    # [careful: Series].values field yields an almost full order of magnitude performance improvement in
    # the sum function.
    #
    # OR: try groupby SELF_TAG, GOLD_TAG + count,
    # maybe apply a filter (== as well as !=) to aggregate TP, FP and FN
    # example: titanic.groupby('class')['survived'].count()
    # timeit
    for index, row in df.iterrows():
        self_txt = row[SELF_TXT]
        self_tag = row[SELF_TAG]
        gold_txt = row[GOLD_TXT]
        gold_tag = row[GOLD_TAG]

        # make sure the terms in predictions and references are equal
        # slows process down, but safety first!
        try:
            assert self_txt == self_txt
        except AssertionError as e:
            e.args += '\nself={} != {}=gold'.format(self_txt, gold_txt),
            raise

        if self_tag == gold_tag:
            classes.loc[gold_tag][TP] += 1
            # One could also count the True Negatives for all other classes, but since
            # they aren't needed and computational expensive, we avoid that.
            if False:
                for tag in TAGSET_TAGS - gold_tag:
                    classes.loc[tag][TN] += 1
        else:
            # print(self_txt, self_tag, gold_tag)
            classes.loc[self_tag][FP] += 1
            classes.loc[gold_tag][FN] += 1

    # calculate Precision, Recall, F1
    for clazz, row in classes.iterrows():
        calculate_metrics(row)

    df_sum = classes.sum()
    calculate_metrics(df_sum)
    # TODO: average gewichten: TP(class)/TP(sum)
    df_mean = classes.mean()
    classes.loc['sum'] = df_sum
    classes.loc['mean'] = df_mean
    return classes


def run_evaluation_for(gold, pred, tagset):
    combined = concat_df(gold, pred)

    # TODO: HDT enthält PIDAT, das nicht in der Tiger-Trainingsmenge vorkommt
    # TODO: und daher nicht von NLTK gelernt werden konnte. Am nächsten verwandt: PIAT => ersetzen
    results = evaluate(combined)
    return results


def main():
    """
    Evaluates the tagging with several metrics. Each corpus for each framework.
    STTS and Universal tagsets can be evaluated.
    """

    t0 = time()

    for corpus in [TIGER, HDT]:
        # --- load ground truth ---
        gold = get_preprocessed_corpus(corpus)

        for framework in [SPACY, NLTK]:
            # --- load predictions ---
            pred = get_selftagged_corpus(corpus, framework)

            print(">>> Evaluating Accuracy, Precision, Recall and F_1 measure for {}/{}."
                  "This may take a while!".format(corpus, framework))
            results = run_evaluation_for(gold, pred, STTS)
            tprint(results)
            results.to_csv(
                OUT_DIR / f'{corpus}_{framework}_{tagset}_results_new.csv', sep='\t'
            )

    print(f"Done in {time() - t0:f}s")


if __name__ == '__main__':
    main()
