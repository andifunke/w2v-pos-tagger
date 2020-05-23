#!/usr/bin/env python3

"""
Trains a German part-of-speech tagger on the TIGER corpus by using the ClassifierBasedGermanTagger.

(https://github.com/ptnplanet/NLTK-Contributions/tree/master/ClassifierBasedGermanTagger)

The training script is based on this tutorial:
https://datascience.blog.wzb.eu/2016/07/13/accurate-part-of-speech-tagging-of-german-texts-with-nltk/
"""

import argparse
import pickle
import random

import nltk

from ClassifierBasedGermanTagger import ClassifierBasedGermanTagger
from w2v_pos_tagger.dataio import TIGER_DIR, OUT_DIR


def parse_args() -> argparse.Namespace:
    """
    Parses module-specific arguments.

    :returns: arguments object
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--evaluate', action='store_true', required=False,
        help='Evaluate the model after training. Will reduce the number of samples for training.'
    )
    parser.set_defaults(evaluate=False)
    args = parser.parse_args()

    return args


def read_corpus():
    """Reads the TIGER corpus as an NLTK conll corpus."""

    print('Reading TIGER corpus')

    return nltk.corpus.ConllCorpusReader(
        root=TIGER_DIR.as_posix(),
        fileids='tiger_release_aug07.corrected.16012013.conll09',
        columntypes=['ignore', 'words', 'ignore', 'ignore', 'pos'],
        encoding='utf-8'
    )


def train_tagger(corpus, evaluate=False):
    """Trains an NLTK based POS-tagger and evaluates the model on a test set."""

    tagged_sents = corpus.tagged_sents()

    if evaluate:
        tagged_sents = [sentence for sentence in tagged_sents]
        random.shuffle(tagged_sents)
        # set a split ratio: 90% for training, 10% for testing
        test_fraction = 0.1
        split_size = int(len(tagged_sents) * test_fraction)
    else:
        # use the full corpus in non-evaluate (i.e. full) mode
        split_size = 0

    train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]

    print('Training POS tagger on TIGER corpus')
    tagger = ClassifierBasedGermanTagger(train=train_sents)

    if evaluate:
        print('Evaluating POS tagger')
        accuracy = tagger.evaluate(test_sents)
        print(f'Accuracy: {accuracy:0.3f}')

    return tagger


def main():
    args = parse_args() 
    evaluate = args.evaluate

    corpus = read_corpus()
    tagger = train_tagger(corpus, evaluate=evaluate)

    model_path = OUT_DIR / f'nltk_german_classifier_data.pickle'
    print('Saving POS tagger to', model_path)
    with open(model_path, 'wb') as fp:
        pickle.dump(tagger, fp, protocol=2)


if __name__ == '__main__':
    main()
