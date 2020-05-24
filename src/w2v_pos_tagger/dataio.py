#!/usr/bin/env python3

"""
Provides functions to load and normalize the corpora.

Maps the STTS tagset to the reduced Universal tagset.

Run this script as __main__ to cache the conversions into csv files in ``./corpora/out``.
"""
from pathlib import Path
from time import time
from typing import Union

import numpy as np
import pandas as pd
from gensim.models.word2vec import Word2Vec
from pandarallel import pandarallel
from tabulate import tabulate

from w2v_pos_tagger.constants import (
    SPACY, NLTK, TIGER, HDT, MINIMAL, DEFAULT, PREPROCESSED, PREDICTIONS, SENT_ID,
    TOKN_ID, FORM, LEMM, STTS, UNIV, CORP, KEYS, CORPUS_BUGS, STTS_UNI_MAP, UNIV_TAGS
)

# --- project paths ---
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_DIR = SRC_DIR.parent
CORPORA_DIR = PROJECT_DIR / 'corpora'
OUT_DIR = PROJECT_DIR / 'out'
EVAL_DIR = OUT_DIR / 'evaluation'
EMBEDDINGS_DIR = OUT_DIR / 'embeddings'
MODELS_DIR = OUT_DIR / 'models'
ANNOTATIONS_DIR = OUT_DIR / 'annotations'
TIGER_DIR = CORPORA_DIR / 'tiger-conll'
HDT_DIR = CORPORA_DIR / 'hamburg-dependency-treebank-conll'

# corpora file names
FILES = {
    TIGER: [TIGER_DIR / 'tiger_release_aug07.corrected.16012013.conll09'],
    HDT: [HDT_DIR / f for f in ['part_A.conll', 'part_B.conll', 'part_C.conll']],
    PREPROCESSED: lambda corpus: OUT_DIR / f'{corpus}_preprocessed.csv',
    PREDICTIONS: lambda corpus, framework: ANNOTATIONS_DIR / f'{corpus}_pos_by_{framework}.csv'
}


def conv_tags(tag, mapping: dict = CORPUS_BUGS):
    """Replaces wrong tags in a corpus."""

    return mapping.get(tag, tag)


def conv_token_id(token_id):
    return int(token_id.split('_')[1])


CONVERTERS = {
    TIGER: {STTS: conv_tags, TOKN_ID: conv_token_id},
    HDT: {STTS: lambda x: conv_tags(x, {**CORPUS_BUGS, 'PIDAT': 'PIAT'})},
    SPACY: {STTS: conv_tags},
    NLTK: None
}


def get_preprocessed_corpus(corpus):
    assert corpus in {TIGER, HDT}
    print(f'Reading preprocessed {corpus} corpus')
    return pd.read_csv(
        FILES[PREPROCESSED](corpus), sep="\t", dtype={TOKN_ID: int, SENT_ID: int}, na_filter=False
    )


def get_baseline_corpus(corpus=TIGER, framework=SPACY, show_sample=0):
    assert corpus in {TIGER, HDT}, f'{corpus} is an unknown corpus.'
    assert framework in {SPACY, NLTK}, f'{framework} is an unknown framework.'

    print(f'Reading {corpus} corpus, annotated with {framework}')
    df = pd.read_csv(
        FILES[PREDICTIONS](corpus, framework), sep="\t", dtype={SENT_ID: int, TOKN_ID: int},
        converters=CONVERTERS[framework], skip_blank_lines=True, quotechar='\x07', na_filter=False
    )
    if show_sample:
        tprint(df, show_sample)
    return df


def get_svm_annotations(show_sample=0):

    annotations = [file for file in ANNOTATIONS_DIR.iterdir() if 'SVM' in file.as_posix()]

    for file in annotations:
        print(f'Reading annotated corpus from {file}')
        df = pd.read_csv(
            file, sep="\t", dtype={SENT_ID: int, TOKN_ID: int},
            skip_blank_lines=True, quotechar='\x07', na_filter=False
        )
        if show_sample:
            tprint(df, show_sample)

        yield df, file.stem


def get_original_corpus(corpus, print_sample=0, raw=False):
    """
    Reads a given corpus and applies converters.

    Returns a DataFrame.
    """
    assert corpus in {TIGER, HDT}, f'{corpus} is an unknown corpus.'

    print(f'>>> Reading original {corpus} corpus')
    converters = CONVERTERS[corpus]

    if raw:
        use_cols = None
        dtype = None
    else:
        use_cols = KEYS[MINIMAL]
        if converters is not None and TOKN_ID in converters:
            dtype = None
        else:
            dtype = {TOKN_ID: int}

    df = []
    for file in FILES[corpus]:
        print(f'Reading file {file}')
        df.append(
            pd.read_csv(
                file, sep="\t", names=KEYS[corpus], header=None, usecols=use_cols, dtype=dtype,
                skip_blank_lines=True, quotechar='\x07', converters=converters, na_filter=False,
            )
        )
    df = pd.concat(df)
    df[CORP] = corpus

    if not raw:
        def add_univ(stts):
            return STTS_UNI_MAP[stts]

        def conv_lemm(form, lemm):
            return form if lemm == '--' else lemm

        print('Adding Universal Tagset')
        df[UNIV] = df.parallel_apply(lambda row: add_univ(row[STTS]), axis=1)
        print()

        print('Correcting Lemmata')
        df[LEMM] = df.parallel_apply(lambda row: conv_lemm(row[FORM], row[LEMM]), axis=1)
        print()

        print('Adding Sentence IDs')
        sent_split = df[TOKN_ID] <= df[TOKN_ID].shift(fill_value=1)
        df[SENT_ID] = sent_split.cumsum()

        df = df[KEYS[DEFAULT]]

    if print_sample:
        tprint(df, print_sample)

    return df


def trainset(corpus, size=0, dimensionality=25, architecture='sg', lowercase=False):

    lc = '_lc' if lowercase else ''

    emb_path = EMBEDDINGS_DIR / f'{architecture}_{dimensionality:03d}{lc}.w2v'
    print('Loading embeddings from', emb_path)
    model = Word2Vec.load(str(emb_path))
    word_vectors = model.wv

    size = None if size < 1 else size
    df = get_preprocessed_corpus(corpus)[[FORM, UNIV]]
    df = df[:size]

    X = np.stack(df.FORM.map(word_vectors.word_vec))
    y = df.UNIV.map(UNIV_TAGS.get).values

    return X, y


def tprint(data: Union[dict, pd.DataFrame], head=0, to_latex=False):
    """Prints a DataFrame as a well formatted table."""

    if isinstance(data, dict):
        data = pd.Series(data).to_frame()

    if head > 0:
        data = data.head(head)
    elif head < 0:
        data = data.tail(-head)

    formats = [".0f"] + [
        ".3f" if dt in [np.dtype('float64'), np.dtype('float32')]
        else ".0f"
        for dt in data.dtypes
    ]
    print(tabulate(data, headers="keys", tablefmt="pipe", floatfmt=formats) + '\n')

    if to_latex:
        print(data.to_latex(bold_rows=True))


def main():
    """Loads the original corpora, applies normalization and caches the process in csv files."""

    OUT_DIR.mkdir(exist_ok=True)

    for corpus in [TIGER, HDT]:
        t0 = time()
        df = get_original_corpus(corpus, print_sample=-50)
        print(f'Writing {FILES[PREPROCESSED](corpus)}')
        df.to_csv(FILES[PREPROCESSED](corpus), sep='\t', index=False)
        print(f'{corpus} done in {time() - t0:.2f}s\n')


if __name__ == '__main__':
    try:
        pandarallel.initialize(progress_bar=True, use_memory_fs=True)
    except SystemError:
        pandarallel.initialize(progress_bar=True)

    main()
