#!/usr/bin/env python3

"""
In parts inspired by
http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#sphx-glr-auto-examples-text-document-classification-20newsgroups-py
"""

import json
import re
from os import listdir

from sklearn.metrics import f1_score

from data_loader import *
# argument parsing and setting default values
from svm_tagger_train import get_xy, get_options


def test_main(directory=None, file=None):
    """ a given directory and/or filename overwrites the defaults """
    print('start testing')
    t0 = time()

    dir_name = CORPORA_DIR if directory is None else directory
    file_name = OPTIONS['model_file'] if file is None else file
    files = [f for f in listdir(dir_name) if re.match(file_name, f)]

    clf = None
    options = None
    scaler = None
    for fname in files:
        if '_clf.' in fname:
            print('loading clf from', fname)
            clf = pd.read_pickle(path.join(dir_name, fname))
        elif '_scale.' in fname:
            print('loading scaler from', fname)
            scaler = pd.read_pickle(path.join(dir_name, fname))
        elif '_options.' in fname:
            print('loading options from', fname)

            options = json.load(open(path.join(dir_name, fname)))

    if clf is None:
        print("no clf file found. exit")
        return

    if options is None:
        print("no options file found. exit")
        return

    # gathering information from model_file to load the correct embedding file
    embedding_infos = file_name.split('_')
    pretrained = True if embedding_infos[2] == 'pretrained' else False
    embedding_model = embedding_infos[3]
    embedding_size = int(embedding_infos[4])
    lowercase = True if (len(embedding_infos) > 5 and embedding_infos[5] == 'lc') else False

    X, y_true = get_xy(HDT, size=OPTIONS['test_size'], pretrained=pretrained, embedding_size=embedding_size,
                       embedding_model=embedding_model, lowercase=lowercase)

    # using the scaler only if the data was trained on scaled values
    if scaler is not None:
        print('scaling values')
        X = scaler.transform(X)

    print('predict')
    y_pred = clf.predict(X)

    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    print('f1_micro score:', f1_micro)
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    print('f1_macro score:', f1_macro)
    f1_weighted = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
    print('f1_weighted score:', f1_weighted)

    options['test_size'] = OPTIONS['test_size']
    options['f1_micro'] = f1_micro
    options['f1_macro'] = f1_macro
    options['f1_weighted'] = f1_weighted

    result_fname = path.join(dir_name, file_name + '_testresults_{:d}.json'.format(OPTIONS['test_size']))
    options['test_time'] = time() - t0

    print('writing results to', result_fname)
    print(options)
    with open(result_fname, 'w') as f:
        json.dump(options, f)

    print("done in {:f}s".format(options['test_time']))


if __name__ == '__main__':
    OPTIONS = get_options()

    test_main(directory=OPTIONS['model_dir'], file=OPTIONS['model_file'])
