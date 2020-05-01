#!/usr/bin/env python3

"""
Applies part-of-speech tagging via spaCy and NLTK to the TIGER and HDT corpus.
"""

import csv
import pickle
from functools import partial
from time import time

import spacy
from tqdm import tqdm

from data_loader import OUT_DIR, get_preprocessed_corpus
from constants import (
    SPACY, NLTK, TIGER, HDT, SENT_ID, TOKN_ID, FORM, STTS, UNIV, CORP, STTS_UNI_MAP_EXTENDED
)


def tag_corpus(tagger, corpus):
    """Tags a corpus (dataframe) sentence-wise tagger function."""

    def handle_sentence(group):
        sent = group[FORM].tolist()
        tagged_sent = tagger(sent)
        assert len(tagged_sent) == len(sent)

        group[STTS] = tagged_sent
        group[UNIV] = group[STTS].map(STTS_UNI_MAP_EXTENDED)

        return group[[CORP, SENT_ID, TOKN_ID, FORM, STTS, UNIV]]

    df = corpus.groupby(SENT_ID).progress_apply(handle_sentence)
    return df


def tag_sentence_with_spacy(sentence, tagger):
    """Tags a sentence with a spaCy POS-tagger."""

    doc = tagger.tokenizer.tokens_from_list(sentence)
    tagger.tagger(doc)
    return [token.tag_ for token in doc]


def tag_sentence_with_nltk(sentence, tagger):
    """Tags a sentence with an NLTK POS-tagger."""

    doc = tagger.tag(sentence)
    return [token[1] for token in doc]


def main():

    # --- load taggers ---
    print('Loading', SPACY)
    nlp = spacy.load('de')
    spacy_tagger = partial(tag_sentence_with_spacy, tagger=nlp)

    print('Loading', NLTK)
    with open(OUT_DIR / 'nltk_german_classifier_data.pickle', 'rb') as fp:
        nlp = pickle.load(fp)
    nltk_tagger = partial(tag_sentence_with_nltk, tagger=nlp)

    # --- apply spacy and nltk on Tiger and HDT ---
    for name in [TIGER, HDT]:
        corpus = get_preprocessed_corpus(name)

        for framework in [SPACY, NLTK]:
            print(f'>>> Starting {name} POS tagging with {framework}')
            tagger = spacy_tagger if framework == SPACY else nltk_tagger
            df = tag_corpus(tagger, corpus)

            # --- save results ---
            file_path = OUT_DIR / f'{name}_pos_by_{framework}.csv'
            print(f'Writing {file_path}')
            df.to_csv(file_path, header=None, index=None, sep='\t', quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    tqdm.pandas()
    t0 = time()
    main()
    print(f"Done in {time() - t0:0.2f}s")
