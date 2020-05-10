#!/usr/bin/env python3

"""
Train an SVM Part-of-Speech-Tagger.
"""

import argparse
import json
import pickle
from datetime import datetime
from time import time

import numpy as np
from gensim.models.word2vec import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from constants import TIGER, FORM, UNIV, UNIV_TAGS
from data_loader import MODEL_DIR, EMBEDDINGS_DIR, get_preprocessed_corpus


def parse_args() -> argparse.Namespace:
    """
    Parses module-specific arguments. Solves argument dependencies
    and returns cleaned up arguments.

    :returns: arguments object
    """

    parser = argparse.ArgumentParser(description='nlp exercise 2')

    # --- log progress ---
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    # --- Choice of pretrained embedding variation ---
    parser.add_argument('-a', '--architecture', default='sg', type=str, choices=['cb', 'sg'])
    parser.add_argument('-d', '--dimensionality', default=25, type=int)
    parser.add_argument('--lowercase', dest='lowercase', action='store_true')
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false')
    parser.set_defaults(lowercase=False)
    parser.add_argument(
        '--train_size', default=0, type=int,
        help='Train only on a slice of the trainset with length `train_size`.'
    )

    # --- SVC parameters ---
    parser.add_argument('--shrinking', dest='shrinking', action='store_true')
    parser.add_argument('--no-shrinking', dest='shrinking', action='store_false')
    parser.set_defaults(shrinking=False)

    parser.add_argument(
        '--scale', dest='scale', action='store_true', help='Normalize the feature vectors.'
    )
    parser.add_argument('--no-scale', dest='scale', action='store_false')
    parser.set_defaults(scale=False)

    parser.add_argument('--C', default=1.0, type=float, help="Soft-margin parameter.")
    parser.add_argument(
        '--cache-size', default=2000, type=int,
        help="Specify the size of the kernel cache (in MB)."
    )

    parser.add_argument('--max-iter', default=-1, type=int, help="Limit the number of iterations.")
    parser.add_argument('--kernel', default='rbf', type=str, choices=['linear', 'poly', 'rbf'])

    args = parser.parse_args()
    print(args)

    return args


def trainset(corpus, size=0, dimensionality=12, architecture='sg', lowercase=False):
    """ embedding_size, embedding_model and lowercase are only applicable, if pretrained is False.
    In this case the custom trained embeddings are used. Values must correspond to existing w2v model files. """

    # gensim cannot be used on hpc
    lc = '_lc' if lowercase else ''
    emb_path = EMBEDDINGS_DIR / f'{architecture}_{dimensionality:03d}{lc}.w2v'
    model = Word2Vec.load(str(emb_path))
    word_vectors = model.wv
    print('loading embeddings from', emb_path)

    print('loading corpora')
    df = get_preprocessed_corpus(corpus)[[FORM, UNIV]]

    # this is using only the token as feature
    if size == 0:
        X = df[FORM].values
        y = df[UNIV].values
    else:
        X = df[FORM].values[:size]
        y = df[UNIV].values[:size]

    if lowercase:
        X = [word_vectors[token.lower()] for token in X]
    else:
        X = [word_vectors[token] for token in X]

    y = [UNIV_TAGS[label] for label in y]
    X = np.asarray(X, dtype=float, order='C')
    y = np.asarray(y, dtype=int, order='C')
    assert len(X) == len(y)
    return X, y


def __trainset(corpus, size=0, dimensionality=25, architecture='sg', lowercase=False):

    lc = '_lc' if lowercase else ''

    emb_path = EMBEDDINGS_DIR / f'{architecture}_{dimensionality:03d}{lc}.w2v'
    print('Loading embeddings from', emb_path)
    model = Word2Vec.load(str(emb_path))
    word_vectors = model.wv

    size = None if size < 1 else size
    df = get_preprocessed_corpus(corpus)[[FORM, UNIV]]
    df = df[:size]

    # X = df.FORM.apply(lambda x: pd.Series(word_vectors[x])).values
    # y = df.
    # quit()

    # y = [UNIV_TAGS[label] for label in y]
    # X = np.asarray(X, dtype=float, order='C')
    # y = np.asarray(y, dtype=int, order='C')
    # assert len(X) == len(y)

    # return X, y


def main():
    args = parse_args()

    embedding_size = args.dimensionality
    embedding_model = args.architecture
    lowercase = args.lowercase

    t0 = time()
    dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

    save_dir = MODEL_DIR / dt
    train_id = f"{embedding_model}_{embedding_size}{'_lc' if lowercase else ''}"

    X, y = trainset(
        TIGER,
        size=args.train_size,
        dimensionality=embedding_size,
        architecture=embedding_model,
        lowercase=lowercase
    )

    if args.scale:
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        scaler_path = save_dir / f'{train_id}.scaler'
        print('Saving scaler to', scaler_path)
        with open(scaler_path, 'wb') as fp:
            pickle.dump(scaler, fp)

    print('Starting training')
    clf = SVC(
        C=args.C, cache_size=args.cache_size, class_weight=None, kernel=args.kernel,
        decision_function_shape='ovr', gamma='auto', max_iter=args.max_iter, random_state=None,
        shrinking=args.shrinking, tol=0.001, verbose=args.verbose
    )
    print(f'\n{clf}')

    print('fitting...')
    clf.fit(X, y)

    model_file = save_dir / f'{train_id}.model'
    print('\nsaving clf to', model_file)
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)

    args.time_train = time() - t0
    config_file = save_dir / f'{train_id}.config'
    print('saving options to', config_file)
    with open(config_file, 'w') as fp:
        json.dump(vars(args), fp)

    print(f"Done in {args.time_train:.3f}s")


if __name__ == '__main__':
    main()
