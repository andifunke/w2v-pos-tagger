#!/usr/bin/env python3

"""
Applies part-of-speech tagging via a specified SVC model to the TIGER and HDT corpus.
"""

import argparse
import csv
import json
import pickle
from time import time
from pathlib import Path

from sklearn.svm import SVC

from w2v_pos_tagger.dataio import featureset, get_preprocessed_corpus, ANNOTATIONS_DIR, MODELS_DIR
from w2v_pos_tagger.constants import (
    HDT, MODEL_SUFFIX, SCALER_SUFFIX, CONFIG_SUFFIX, TIGER, UNIV,
    UNIV_TAGS_BACKWARDS, KEYS, PREDICTIONS, STTS
)


def parse_args(argv=None) -> argparse.Namespace:
    """
    Parses module-specific arguments.

    Solves argument dependencies and returns cleaned up arguments.

    :returns: arguments object
    """

    parser = argparse.ArgumentParser(
        description="Applies part-of-speech tagging via a specified SVC model "
                    "to the TIGER and HDT corpus."
    )
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help="Either an absolute path to a model folder or a folder name inside `out/models`.",
    )
    parser.add_argument('--corpus', type=str, default=HDT, choices=[HDT, TIGER])
    parser.add_argument('--test-size', type=int, default=0)
    args = parser.parse_args(argv)

    return args


def main(argv=None):
    """A given directory and/or filename overwrites the defaults."""

    print('Start Part-of-Speech Tagging')

    args = parse_args(argv)

    # --- Load model ---
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = MODELS_DIR / model_path
    if not (model_path.exists() and model_path.is_dir()):
        raise ValueError(f'Model path {model_path} does not exist.')

    files = [f for f in model_path.iterdir()]

    clf = None
    config = None
    scaler = None

    for file in files:
        if file.suffix == MODEL_SUFFIX:
            print('loading model from', file)
            with open(file, 'rb') as fp:
                clf: SVC = pickle.load(fp)
        elif file.suffix == SCALER_SUFFIX:
            print('loading scaler from', file)
            with open(file, 'rb') as fp:
                scaler = pickle.load(fp)
        elif file.suffix == CONFIG_SUFFIX:
            print('loading config from', file)
            with open(file) as fp:
                config = json.load(fp)

    if clf is None:
        print("no clf file found. exit")
        return

    if config is None:
        print("no options file found. exit")
        return

    # --- Load features ---
    embedding_path = config.get('embedding_file')
    if embedding_path is not None:
        embedding_path = Path(embedding_path)
        if not embedding_path.is_absolute():
            embedding_path = model_path / embedding_path
        dimensionality = None
        architecture = None
        lowercase = None
    else:
        dimensionality = config['dimensionality']
        architecture = config['architecture']
        lowercase = config['lowercase']

    corpus = args.corpus

    X, y_true = featureset(
        corpus, size=args.test_size, dimensionality=dimensionality,
        architecture=architecture, lowercase=lowercase, embedding_path=embedding_path
    )

    # applying scaling only if the model was trained on normalized features
    if scaler is not None:
        print('scaling values')
        X = scaler.transform(X)

    # --- Predict ---
    print(f'Annotating the HDT corpus using {len(X)} examples. This may take a while')
    t0 = time()

    y_pred = clf.predict(X)

    print(f"Done in {time() - t0:0.2f}s")

    # --- Annotate the corpus ---
    df = get_preprocessed_corpus(corpus)
    df = df[KEYS[PREDICTIONS]].drop(STTS, axis=1)
    df = df[:args.test_size]
    df.loc[:, UNIV] = y_pred
    df.UNIV = df.UNIV.map(UNIV_TAGS_BACKWARDS.get)

    # --- Save ---
    ANNOTATIONS_DIR.mkdir(exist_ok=True, parents=True)
    lc = '_lc' if lowercase else ''
    file_path = ANNOTATIONS_DIR / f'{corpus}_pos_by_{model_path.name}.csv'
    print(f'Writing {file_path}')
    df.to_csv(file_path, index=False, sep='\t', quoting=csv.QUOTE_NONE)


if __name__ == '__main__':
    main()
