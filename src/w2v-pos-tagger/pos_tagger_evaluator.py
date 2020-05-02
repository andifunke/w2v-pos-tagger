#!/usr/bin/env python3
import os

from constants import SELF_TXT, SELF_TAG, GOLD_TXT, \
    GOLD_TAG, TP, TN, FP, FN, PREC, RECL, F1, STTS_TAGS, UNIV_TAGS
from corpora_analyser import get_tagset
from data_loader import *


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


def run_evaluation_for(corpus, framework, analyse=False):
    sample_size = 0
    df_self = get_selftagged_corpus(corpus, framework, show_sample=sample_size)
    df_gold = TO_DF if corpus == TIGER else HO_DF

    if analyse:
        analyse_tagset(df_self, corpus, framework)
        return

    combined = concat_df(df_self, df_gold)

    print(">>> evaluating Accuracy, Precision, Recall and F_1 measure for {}/{}.\n"
          "This may take a while!".format(corpus, framework))

    # TODO: HDT enthält PIDAT, das nicht in der Tiger-Trainingsmenge vorkommt
    # TODO: und daher nicht von NLTK gelernt werden konnte. Am nächsten verwandt: PIAT => ersetzen
    results = evaluate(combined)
    tprint(results)
    results.to_csv(os.path.join(OUT_DIR, '{}_{}_{}_results_new.csv'.format(corpus, framework, TAGSET)), sep='\t')


def analyse_tagset(df, corpus, framework):
    print(">>> analysing given tagsets")

    tagset = get_tagset(df, TAGSET)
    print('{} {} tagset:\n'.format(corpus, framework), sorted(tagset.keys()), '\nlength:', len(tagset))


def tag_evaluator_main():
    """
    Evaluates the tagging with several metrics. Each corpus for each framework.
    STTS and Universal tagsets can be evaluated.
    """

    print(">>> start evaluating tagged corpora on tagset", TAGSET)
    t0 = time()

    # could be more efficient if SPACY and NLTK would be processed at the same time
    # instead of one after the other, but this makes the code clearer.
    # Same goes for STTS vs. Universal tagsets
    run_evaluation_for(TIGER, SPACY)
    run_evaluation_for(TIGER, NLTK)
    run_evaluation_for(HDT, SPACY)
    run_evaluation_for(HDT, NLTK)
    print("done in {:f}s".format(time() - t0))


if __name__ == '__main__':
    # --- load ground truth ---
    TO_DF = get_preprocessed_corpus(TIGER)
    HO_DF = get_preprocessed_corpus(HDT)

    # Evaluate for both STTS and Universal tagsets by default.
    TAGSET = STTS
    TAGSET_TAGS = STTS_TAGS
    tag_evaluator_main()

    TAGSET = UNIV
    TAGSET_TAGS = UNIV_TAGS.keys()
    tag_evaluator_main()
