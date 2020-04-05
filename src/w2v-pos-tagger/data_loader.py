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

# --- constants ---
SPACY = 'SPACY'
NLTK = 'NLTK'
TIGER = 'TIGER'
HDT = 'HDT'
MINIMAL = 'MINIMAL'
DEFAULT = 'DEFAULT'
PREPROCESSED = 'PREPROCESSED'
SELFTAGGED = 'SELFTAGGED'

ID = 'ID'
SENT_ID = 'SENT_ID'
TOKN_ID = 'TOKN_ID'
FORM = 'FORM'
LEMM = 'LEMM'
STTS = 'STTS'
REDU = 'REDU'
UNIV = 'UNIV'
CORP = 'CORP'

SELF_TXT = 'SELF_TXT'
SELF_TAG = 'SELF_TAG'
GOLD_TXT = 'GOLD_TXT'
GOLD_TAG = 'GOLD_TAG'
TP = 'TP'
TN = 'TN'
FP = 'FP'
FN = 'FN'
PREC = 'PREC'
RECL = 'RECL'
F1 = 'F1'


# --- tagsets ---

# Tiger CONLL09 columns
KEYS = {
    TIGER: [
        TOKN_ID, FORM, LEMM, 'PLEMMA', STTS, 'PPOS', 'FEAT', 'PFEAT', 'HEAD', 'PHEAD', 'DEPREL',
        'PDEPREL', 'FILLPRED', 'PRED', 'APREDS'
    ],
    HDT: [TOKN_ID, FORM, LEMM, REDU, STTS, 'FEAT', 'HEAD', 'DEPREL', 'UNKNOWN_1', 'UNKNOWN_2'],
    MINIMAL: [TOKN_ID, FORM, LEMM, STTS],
    DEFAULT: [CORP, SENT_ID, TOKN_ID, FORM, LEMM, STTS, UNIV],
    SPACY: [ID, FORM, LEMM, 'POS', STTS, 'DEP', 'SHAPE', 'ALPHA', 'STOP'],
    NLTK: [ID, FORM, STTS],
    SELFTAGGED: [CORP, SENT_ID, TOKN_ID, FORM, STTS, UNIV]
}

"""
These are the common tags from STTS website. PAV was replaced by PROAV since all 
corpora do so as well. The mapping is according to de-tiger.map
https://github.com/slavpetrov/universal-pos-tags/blob/master/de-tiger.map
PIDAT -> PRON according to de-negra.map
https://github.com/slavpetrov/universal-pos-tags/blob/master/de-negra.map
NNE was removed from the mapping since no official STTS tag.
"""
STTS_UNI_MAP = {
    '$(': '.',
    '$,': '.',
    '$.': '.',
    'ADJA': 'ADJ',
    'ADJD': 'ADJ',
    'ADV': 'ADV',
    'APPO': 'ADP',
    'APPR': 'ADP',
    'APPRART': 'ADP',
    'APZR': 'ADP',
    'ART': 'DET',
    'CARD': 'NUM',
    'FM': 'X',
    'ITJ': 'X',
    'KOKOM': 'CONJ',
    'KON': 'CONJ',
    'KOUI': 'CONJ',
    'KOUS': 'CONJ',
    'NE': 'NOUN',
    'NN': 'NOUN',
    'PDAT': 'PRON',
    'PDS': 'PRON',
    'PIDAT': 'PRON',
    'PIAT': 'PRON',
    'PIS': 'PRON',
    'PPER': 'PRON',
    'PPOSAT': 'PRON',
    'PPOSS': 'PRON',
    'PRELAT': 'PRON',
    'PRELS': 'PRON',
    'PRF': 'PRON',
    'PROAV': 'PRON',
    'PTKA': 'PRT',
    'PTKANT': 'PRT',
    'PTKNEG': 'PRT',
    'PTKVZ': 'PRT',
    'PTKZU': 'PRT',
    'PWAT': 'PRON',
    'PWAV': 'PRON',
    'PWS': 'PRON',
    'TRUNC': 'X',
    'VAFIN': 'VERB',
    'VAIMP': 'VERB',
    'VAINF': 'VERB',
    'VAPP': 'VERB',
    'VMFIN': 'VERB',
    'VMINF': 'VERB',
    'VMPP': 'VERB',
    'VVFIN': 'VERB',
    'VVIMP': 'VERB',
    'VVINF': 'VERB',
    'VVIZU': 'VERB',
    'VVPP': 'VERB',
    'XY': 'X'
}
STTS_TAGS = STTS_UNI_MAP.keys()
STTS_DEFAULT = set(STTS_TAGS - {'PROAV'}).union({'PAV'})

# universal tagset
UNIV_TAGS = {
    '.': 0,
    'ADJ': 1,
    'ADV': 2,
    'ADP': 3,
    'DET': 4,
    'NUM': 5,
    'CONJ': 6,
    'NOUN': 7,
    'PRON': 8,
    'PRT': 9,
    'VERB': 10,
    'X': 11
}
UNIV_TAGS_BACKWARDS = {(v, k) for k, v in UNIV_TAGS.items()}

# corpus fixes
CORPUS_BUGS = {'NNE': 'NE', 'PPOSSAT': 'PPOSAT', 'VAIZU': 'VVIZU'}
STTS_UNI_MAP_EXTENDED = STTS_UNI_MAP.copy()
STTS_UNI_MAP_EXTENDED.update({'NNE': 'NOUN', 'PPOSSAT': 'PRON', 'VAIZU': 'VERB'})


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
    PREPROCESSED: lambda corpus: OUT_DIR / '{}_preprocessed.csv'.format(corpus),
    SELFTAGGED: lambda corpus, framework: OUT_DIR / '{}_pos_by_{}.csv'.format(corpus, framework)
}


def conv_tags(tag):
    """replace wrong tags in corpora"""

    if tag in CORPUS_BUGS:
        return CORPUS_BUGS[tag]
    return tag


def conv_token_id(token_id):
    return int(token_id.split('_')[1])


CONVERTERS = {
    TIGER: {STTS: conv_tags, TOKN_ID: conv_token_id},
    HDT: {STTS: conv_tags},
    SPACY: {STTS: conv_tags},
    NLTK: None
}

CURRENT_SENTENCE_ID = 1
LAST_TOKEN_ID = 0


def read_raw(file, keys, converters=None, raw=False):
    """
    Read a file with given column names (keys) and apply converters.

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
    print('Reading preprocessed {} corpus'.format(corpus))
    return pd.read_csv(
        FILES[PREPROCESSED](corpus), sep="\t", dtype={TOKN_ID: int, SENT_ID: int}, na_filter=False
    )


def get_original_corpus(corpus, show_sample=0, raw=False):
    assert corpus in {TIGER, HDT}

    print('Reading original {} corpus'.format(corpus))
    # read to pandas DataFrame
    global CURRENT_SENTENCE_ID
    global LAST_TOKEN_ID
    CURRENT_SENTENCE_ID = 1
    LAST_TOKEN_ID = 0

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

    print('reading {} corpus, annotated with {}'.format(corpus, framework))
    # read to pandas data frame
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
        print("{} done in {:.2f}s".format(corpus, time() - t0))


if __name__ == '__main__':
    main()
