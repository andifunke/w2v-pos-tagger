#!/usr/bin/env python3

"""
In parts inspired by
http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
"""

import argparse
import json
from datetime import datetime

import numpy as np
import six.moves.cPickle as cPickle
from sklearn import svm, preprocessing

from data_loader import *


# argument parsing and setting default values
def get_options():
    parser = argparse.ArgumentParser(description='nlp exercise 2')

    parser.add_argument('--prepare', dest='prepare', action='store_true')
    parser.add_argument('--no-prepare', dest='prepare', action='store_false')
    parser.set_defaults(prepare=False)
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--no-train', dest='train', action='store_false')
    parser.set_defaults(train=False)
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--no-test', dest='test', action='store_false')
    parser.set_defaults(test=False)
    parser.add_argument('--shrinking', dest='shrinking', action='store_true')
    parser.add_argument('--no-shrinking', dest='shrinking', action='store_false')
    parser.set_defaults(shrinking=False)
    parser.add_argument('--verbose', dest='verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.set_defaults(verbose=True)
    parser.add_argument('--pretrained', dest='pretrained', action='store_true')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=False)
    parser.add_argument('--scale', dest='scale', action='store_true')
    parser.add_argument('--no-scale', dest='scale', action='store_false')
    parser.set_defaults(scale=False)
    parser.add_argument('--lowercase', dest='lowercase', action='store_true')
    parser.add_argument('--no-lowercase', dest='lowercase', action='store_false')
    parser.set_defaults(lowercase=False)
    parser.add_argument('--gensim', dest='gensim', action='store_true')
    parser.add_argument('--no-gensim', dest='gensim', action='store_false')
    parser.set_defaults(gensim=False)

    parser.add_argument('--train_size', default=0, type=int,
                        help='train only on slice of given length')
    parser.add_argument('--test_size', default=0, type=int,
                        help='test only on slice of given length')
    parser.add_argument('--C', default=1.0, type=float)
    parser.add_argument('--cache_size', default=2000, type=int)
    parser.add_argument('--max_iter', default=-1, type=int)
    parser.add_argument('--kernel', default='rbf', type=str,
                        choices=['linear', 'poly', 'rbf'])
    parser.add_argument('--embedding_size', default=25, type=int,
                        choices=[12, 25, 50, 100])
    parser.add_argument('--embedding_model', default='sg', type=str,
                        choices=['cb', 'sg'])
    parser.add_argument('--clf_file', default='', type=str)
    parser.add_argument('--model_file', default='', type=str)
    parser.add_argument('--model_dir', default=CORPORA_DIR, type=str)

    return vars(parser.parse_args())


def get_xy(corpus, size=0, embedding_size=12, embedding_model='sg', lowercase=False):
    """ embedding_size, embedding_model and lowercase are only applicable, if pretrained is False.
    In this case the custom trained embeddings are used. Values must correspond to existing w2v model files. """

    # gensim cannot be used on hpc
    lc = '_lc' if lowercase else ''
    fname = path.join(CORPORA_DIR, 'custom_embedding_{}_{:d}{}.vec'.format(embedding_model, embedding_size, lc))
    if 'OPTIONS' in globals() and OPTIONS['gensim']:
        import gensim.models.word2vec as wv
        model = wv.Word2Vec.load(fname)
        word_vectors = model.wv
    else:
        word_vectors = pd.read_pickle(fname + '.pickle')
    print('loading embeddings from', fname)

    print('loading corpora')
    df = get_preprocessed_corpus(corpus)[[FORM, UNIV]]

    # this is using only the token as feature
    if size == 0:
        X = df[FORM].as_matrix()
        y = df[UNIV].as_matrix()
    else:
        X = df[FORM].as_matrix()[:size]
        y = df[UNIV].as_matrix()[:size]

    if lowercase:
        X = [word_vectors[token.lower()] for token in X]
    else:
        X = [word_vectors[token] for token in X]

    y = [UNIV_TAGS[label] for label in y]
    X = np.asarray(X, dtype=float, order='C')
    y = np.asarray(y, dtype=int, order='C')
    assert len(X) == len(y)
    return X, y


def train_main():
    embedding_size = OPTIONS['embedding_size']
    embedding_model = OPTIONS['embedding_model']
    lowercase = OPTIONS['lowercase']

    t0 = time()
    dt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')

    train_fname = '{}_custom_{}_{}{}'.format(dt, embedding_model, embedding_size, '_lc' if lowercase else '')

    X, y = get_xy(TIGER,
                  size=OPTIONS['train_size'],
                  embedding_size=embedding_size,
                  embedding_model=embedding_model,
                  lowercase=lowercase)

    if OPTIONS['scale']:
        scaler = preprocessing.MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        scale_fname = path.join(CORPORA_DIR, train_fname + '_scale.pickle')
        print('saving scaler to', scale_fname)
        with open(scale_fname, 'wb') as f:
            cPickle.dump(scaler, f)

    print(OPTIONS)

    print('start training')
    clf = svm.SVC(C=OPTIONS['C'], cache_size=OPTIONS['cache_size'], class_weight=None, kernel=OPTIONS['kernel'],
                  decision_function_shape='ovr', gamma='auto', max_iter=OPTIONS['max_iter'], random_state=None,
                  shrinking=OPTIONS['shrinking'], tol=0.001, verbose=OPTIONS['verbose'])
    print('\n' + str(clf))

    print('fitting...')
    clf.fit(X, y)

    clf_fname = path.join(CORPORA_DIR, train_fname + '_clf.pickle')
    print('\nsaving clf to', clf_fname)
    with open(clf_fname, 'wb') as f:
        cPickle.dump(clf, f)

    OPTIONS['time_train'] = time() - t0
    options_fname = path.join(CORPORA_DIR, train_fname + '_options.json')
    print('saving options to', options_fname)
    with open(options_fname, 'w') as f:
        json.dump(OPTIONS, f)

    print("done in {:f}s".format(OPTIONS['time_train']))


if __name__ == '__main__':
    OPTIONS = get_options()
    train_main()
