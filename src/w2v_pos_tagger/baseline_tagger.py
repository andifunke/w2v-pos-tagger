#!/usr/bin/env python3

"""
Applies part-of-speech tagging via spaCy and NLTK to the TIGER and HDT corpus.
"""

import argparse
import csv
import multiprocessing as mp
import pickle
from functools import partial
from time import time

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

from w2v_pos_tagger.constants import (
    SPACY, NLTK, TIGER, HDT, SENT_ID, STTS, UNIV, STTS_UNI_MAP_EXTENDED, KEYS, PREDICTIONS
)
from w2v_pos_tagger.dataio import OUT_DIR, get_preprocessed_corpus, ANNOTATIONS_DIR


_SPACY = None
_NLTK = None


def parse_args(argv=None) -> argparse.Namespace:
    """
    Parses module-specific arguments. Solves argument dependencies
    and returns cleaned up arguments.

    :returns: arguments object
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=-1)
    args = parser.parse_args(argv)
    return args


def spacy_tagger(sentence):
    """Tags a sentence with a spaCy POS-tagger."""

    doc = _SPACY.tokenizer.tokens_from_list(sentence)
    _SPACY.tagger(doc)
    return [token.tag_ for token in doc]


def nltk_tagger(sentence):
    """Tags a sentence with an NLTK POS-tagger."""

    doc = _NLTK.tag(sentence)
    return [token[1] for token in doc]


def tag_sentence(group, tagger):
    sent = group.FORM.tolist()
    tagged_sent = tagger(sent)
    group[STTS] = tagged_sent
    group[UNIV] = group.STTS.map(STTS_UNI_MAP_EXTENDED)

    return group[KEYS[PREDICTIONS]]


def tag_chunk(chunk, tagger):
    df = [tag_sentence(grp, tagger) for _, grp in tqdm(chunk)]
    df = pd.concat(df)
    return df


def tag_corpus(corpus, tagger, workers: int = -1):
    """Tags a corpus (dataframe) sentence-wise tagger function."""

    groups = corpus.groupby(SENT_ID)

    workers = mp.cpu_count() if workers < 1 else workers
    print(f'Parallelized using {workers} workers.')

    ctx = mp.get_context('fork')
    with ctx.Pool(workers) as pool:
        groups = np.array_split(groups, workers)
        df = pd.concat(pool.map(partial(tag_chunk, tagger=tagger), groups))

    return df


def main(argv=None):
    global _SPACY, _NLTK

    args = parse_args(argv)
    tqdm.pandas()

    # --- load nlp frameworks ---
    print('Loading', SPACY)
    _SPACY = spacy.load('de')

    print('Loading', NLTK)
    with open(OUT_DIR / 'nltk_german_classifier_data.pickle', 'rb') as fp:
        _NLTK = pickle.load(fp)

    t0 = time()

    # --- apply spacy and nltk on Tiger and HDT ---
    for name in [TIGER, HDT]:
        corpus = get_preprocessed_corpus(name)

        for framework in [SPACY, NLTK]:
            print(f'>>> Starting {name} POS tagging with {framework}')
            tagger = spacy_tagger if framework == SPACY else nltk_tagger
            df = tag_corpus(corpus, tagger, workers=args.workers)

            # --- save results ---
            ANNOTATIONS_DIR.mkdir(exist_ok=True, parents=True)
            file_path = ANNOTATIONS_DIR / f'{name}_pos_by_{framework}.csv'
            print(f'Writing {file_path}')
            df.to_csv(file_path, index=False, sep='\t', quoting=csv.QUOTE_NONE)

    print(f"Done in {time() - t0:0.2f}s")


if __name__ == '__main__':
    main()
