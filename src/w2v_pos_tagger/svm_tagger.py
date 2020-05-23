#!/usr/bin/env python3

"""Applies part-of-speech tagging via a specified SVC model to the TIGER and HDT corpus."""
import argparse
import json
import pickle
from time import time
from pathlib import Path

from sklearn.metrics import f1_score

from w2v_pos_tagger.constants import HDT, MODEL_SUFFIX, SCALER_SUFFIX, CONFIG_SUFFIX
from w2v_pos_tagger.svm_trainer import trainset


def parse_args() -> argparse.Namespace:
    """
    Parses module-specific arguments. Solves argument dependencies
    and returns cleaned up arguments.

    :returns: arguments object
    """

    parser = argparse.ArgumentParser(
        description="Applies part-of-speech tagging via a specified SVC model "
                    "to the TIGER and HDT corpus."
    )
    parser.add_argument('-m', '--model', required=True, type=str)
    parser.add_argument('--test-size', default=0, type=int)
    args = parser.parse_args()

    return args


def main():
    """A given directory and/or filename overwrites the defaults."""

    args = parse_args()

    print('Start Part-of-Speech Tagging')

    model_path = Path(args.model)
    if not (model_path.exists() and model_path.is_dir()):
        raise ValueError(f'Model path {model_path} does not exist.')

    files = [f for f in model_path.iterdir()]

    clf = None
    options = None
    scaler = None

    for file in files:
        if file.suffix == MODEL_SUFFIX:
            print('loading model from', file)
            with open(file, 'rb') as fp:
                clf = pickle.load(fp)
        elif file.suffix == SCALER_SUFFIX:
            print('loading scaler from', file)
            with open(file, 'rb') as fp:
                scaler = pickle.load(fp)
        elif file.suffix == CONFIG_SUFFIX:
            print('loading config from', file)
            with open(file) as fp:
                options = json.load(fp)

    if clf is None:
        print("no clf file found. exit")
        return

    if options is None:
        print("no options file found. exit")
        return

    # gathering information from model_file to load the correct embedding file
    embedding_info = model_path.name.split('_')
    embedding_model = embedding_info[2]
    embedding_size = int(embedding_info[3])
    lowercase = True if (len(embedding_info) > 4 and embedding_info[4] == 'lc') else False

    X, y_true = trainset(
        HDT, size=args.test_size, dimensionality=embedding_size,
        architecture=embedding_model, lowercase=lowercase
    )

    # using the scaler only if the data was trained on scaled values
    if scaler is not None:
        print('scaling values')
        X = scaler.transform(X)

    print(f'Annotating the HDT corpus using {len(X)} examples. This may take a while')
    t0 = time()
    y_pred = clf.predict(X)
    options['test_time'] = time() - t0
    print(f"Done in {options['test_time']:0.2f}s")

    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    print(f'f1_micro score: {f1_micro:0.3f}')
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    print(f'f1_macro score: {f1_macro:0.3f}')
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    print(f'f1_weighted score: {f1_weighted:0.3f}')

    options['test_size'] = args.test_size
    options['f1_micro'] = f1_micro
    options['f1_macro'] = f1_macro
    options['f1_weighted'] = f1_weighted

    file_prefix = "_".join(embedding_info[2:])
    save_path = model_path / f'{file_prefix}_testresults_{args.test_size}.json'

    print('writing results to', save_path)
    print(options)
    with open(save_path, 'w') as f:
        json.dump(options, f)


if __name__ == '__main__':
    main()
