#!/usr/bin/env python3

"""
Provides functions to load and normalize the corpora.

Maps the STTS tagset to the reduced Universal tagset.

Run this script as __main__ to cache the conversions into csv files in ``./corpora/out``.
"""

from pathlib import Path
from time import time

import pandas as pd
from tabulate import tabulate

from constants import (
    SPACY, NLTK, TIGER, HDT, MINIMAL, DEFAULT, PREPROCESSED, SELFTAGGED, SENT_ID,
    TOKN_ID, FORM, LEMM, STTS, UNIV, CORP, KEYS, CORPUS_BUGS, STTS_UNI_MAP
)

# --- project paths ---
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
CORPORA_DIR = PROJECT_DIR / 'corpora'
OUT_DIR = CORPORA_DIR / 'out'
TIGER_DIR = CORPORA_DIR / 'tiger-conll'
HDT_DIR = CORPORA_DIR / 'hamburg-dependency-treebank-conll'

# corpora file names
FILES = {
    TIGER: [TIGER_DIR / 'tiger_release_aug07.corrected.16012013.conll09'],
    HDT: [HDT_DIR / f for f in ['part_A.conll', 'part_B.conll', 'part_C.conll']],
    PREPROCESSED: lambda corpus: OUT_DIR / f'{corpus}_preprocessed.csv',
    SELFTAGGED: lambda corpus, framework: OUT_DIR / f'{corpus}_pos_by_{framework}.csv'
}

CURRENT_SENTENCE_ID = 1
LAST_TOKEN_ID = 0


def conv_tags(tag):
    """Replaces wrong tags in a corpus."""

    if tag in CORPUS_BUGS:
        return CORPUS_BUGS[tag]

    return tag


def conv_token_id(token_id):
    return int(token_id.split('_')[1])


def read_raw(file, keys, converters=None, raw=False):
    """
    Reads a file with given column names (keys) and applies converters.

    Returns a DataFrame.
    """

    if raw:
        use_cols = None
        dtype = None
    else:
        use_cols = KEYS[MINIMAL]
        if converters is not None and TOKN_ID in converters:
            dtype = None
        else:
            dtype = {TOKN_ID: int}
    sbl = True

    df = pd.read_csv(
        file, sep="\t", names=keys, header=None, usecols=use_cols, dtype=dtype,
        skip_blank_lines=sbl, quotechar='\x07', converters=converters, na_filter=False
    )

    if not raw:
        def add_sent_id(token_id):
            global CURRENT_SENTENCE_ID
            global LAST_TOKEN_ID
            if token_id <= LAST_TOKEN_ID:
                CURRENT_SENTENCE_ID += 1
            LAST_TOKEN_ID = token_id
            return CURRENT_SENTENCE_ID

        def add_univ(stts):
            return STTS_UNI_MAP[stts]

        def conv_lemm(form, lemm):
            return form if lemm == '--' else lemm

        df[UNIV] = df.apply(lambda row: add_univ(row[STTS]), axis=1)
        df[LEMM] = df.apply(lambda row: conv_lemm(row[FORM], row[LEMM]), axis=1)
        df[SENT_ID] = df.apply(lambda row: add_sent_id(row[TOKN_ID]), axis=1)

    return df


def get_preprocessed_corpus(corpus):
    assert corpus in {TIGER, HDT}
    print(f'Reading preprocessed {corpus} corpus')
    return pd.read_csv(
        FILES[PREPROCESSED](corpus), sep="\t", dtype={TOKN_ID: int, SENT_ID: int}, na_filter=False
    )


CONVERTERS = {
    TIGER: {STTS: conv_tags, TOKN_ID: conv_token_id},
    HDT: {STTS: conv_tags},
    SPACY: {STTS: conv_tags},
    NLTK: None
}


def get_original_corpus(corpus, show_sample=0, raw=False):
    assert corpus in {TIGER, HDT}

    global CURRENT_SENTENCE_ID
    global LAST_TOKEN_ID
    CURRENT_SENTENCE_ID = 1
    LAST_TOKEN_ID = 0

    print(f'Reading original {corpus} corpus')
    df = pd.concat([
        read_raw(file, KEYS[corpus], converters=CONVERTERS[corpus], raw=raw)
        for file in FILES[corpus]
    ])
    df[CORP] = corpus
    if not raw:
        df = df[KEYS[DEFAULT]]
    if show_sample:
        tprint(df, show_sample)

    return df


def get_selftagged_corpus(corpus=TIGER, framework=SPACY, show_sample=0):
    assert corpus in {TIGER, HDT}
    assert framework in {SPACY, NLTK}

    print(f'Reading {corpus} corpus, annotated with {framework}')
    df = pd.read_csv(
        FILES[SELFTAGGED](corpus, framework), sep="\t", names=KEYS[SELFTAGGED],
        header=None, dtype={SENT_ID: int, TOKN_ID: int}, converters=CONVERTERS[framework],
        skip_blank_lines=True, quotechar='\x07', na_filter=False
    )
    if show_sample:
        tprint(df, show_sample)
    return df


def tprint(df: pd.DataFrame, head=0, to_latex=False):
    """Prints a DataFrame as a well formatted table."""

    if head > 0:
        df = df.head(head)
    elif head < 0:
        df = df.tail(-head)
    print(tabulate(df, headers="keys", tablefmt="pipe", floatfmt=".3f") + '\n')

    if to_latex:
        print(df.to_latex(bold_rows=True))


def main():
    """Loads the original corpora, applies normalization and caches the process in csv files."""

    OUT_DIR.mkdir(exist_ok=True)

    process_corpora = [TIGER, HDT]
    for corpus in process_corpora:
        t0 = time()
        df = get_original_corpus(corpus, show_sample=-50)
        df.to_csv(FILES[PREPROCESSED](corpus), sep='\t', index=False)
        print(f"{corpus} done in {time() - t0:.2f}s")


if __name__ == '__main__':
    main()
