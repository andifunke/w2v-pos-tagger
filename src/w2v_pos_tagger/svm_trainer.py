#!/usr/bin/env python3

"""Train an SVM Part-of-Speech-Tagger."""

import argparse
import json
import pickle
from datetime import datetime
from time import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from w2v_pos_tagger.constants import TIGER, MODEL_SUFFIX, CONFIG_SUFFIX, SCALER_SUFFIX
from w2v_pos_tagger.dataio import MODELS_DIR, featureset


def parse_args(argv=None) -> argparse.Namespace:
    """
    Parses module-specific arguments. Solves argument dependencies
    and returns cleaned up arguments.

    :returns: arguments object
    """

    parser = argparse.ArgumentParser(description='Train an SVM Part-of-Speech-Tagger.')

    # --- log progress ---
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)

    # --- Choice of pretrained embedding variation ---
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-e', '--embedding', type=str, required=False, default=None,
        help="Path to a pretrained embedding. Will override architecture and dimensionality."
             "Make sure to set the `--lowercase` flag if the embedding was trained on a "
             "lower-cased vocabulary.",
    )
    parser.add_argument('-a', '--architecture', default='sg', type=str, choices=['cb', 'sg'])
    parser.add_argument('-d', '--dimensionality', default=25, type=int)
    parser.add_argument('--lowercase', dest='lowercase', action='store_true')
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false')
    parser.set_defaults(lowercase=False)
    parser.add_argument(
        '--train-size', default=0, type=int,
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

    parser.add_argument(
        '-m', '--model', type=str, required=False, default=None,
        help="Specify a custom name for the model. Otherwise a unique model id will be created.",
    )

    args = parser.parse_args(argv)
    print(args)

    return args


def main(argv=None):
    args = parse_args(argv)

    embedding_size = args.dimensionality
    embedding_model = args.architecture
    lowercase = args.lowercase

    t0 = time()
    dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

    if args.model is None:
        train_id = f"{dt}_{embedding_model}_{embedding_size}{'_lc' if lowercase else ''}"
    else:
        train_id = args.model

    save_dir = MODELS_DIR / train_id
    save_dir.mkdir(exist_ok=True, parents=True)

    X, y = featureset(
        TIGER,
        size=args.train_size,
        dimensionality=embedding_size,
        architecture=embedding_model,
        lowercase=lowercase,
        embedding_path=args.embedding
    )

    if args.scale:
        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        scaler_path = save_dir / f'{train_id}.{SCALER_SUFFIX}'
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

    model_file = save_dir / f'{train_id}{MODEL_SUFFIX}'
    print('\nsaving model to', model_file)
    with open(model_file, 'wb') as f:
        pickle.dump(clf, f)

    args.time_train = time() - t0
    config_file = save_dir / f'{train_id}{CONFIG_SUFFIX}'
    print('saving options to', config_file)
    with open(config_file, 'w') as fp:
        json.dump(vars(args), fp)

    print(f"Done in {args.time_train:.3f}s")


if __name__ == '__main__':
    main()
