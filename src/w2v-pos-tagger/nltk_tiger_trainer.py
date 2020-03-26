import nltk
import random
import pickle
from os import path
from data_loader import DATA_DIR, OUT_DIR
from ClassifierBasedGermanTagger import ClassifierBasedGermanTagger

"""
    based on this tutorial:
    https://datascience.blog.wzb.eu/2016/07/13/accurate-part-of-speech-tagging-of-german-texts-with-nltk/
"""


def read_corpus():
    print('reading TIGER corpus')
    return nltk.corpus.ConllCorpusReader(DATA_DIR,
                                         'tiger_release_aug07.corrected.16012013.conll09',
                                         ['ignore', 'words', 'ignore', 'ignore', 'pos'],
                                         encoding='utf-8')


def train_tagger(corpus, evaluate=False):
    tagged_sents = corpus.tagged_sents()

    # marginal accuracy-boost from 0.9417 -> 0.9423 in evaluate mode
    # therefore I skip the suggested shuffeling, especially in full mode
    if evaluate and False:
        tagged_sents = [sentence for sentence in tagged_sents]
        random.shuffle(tagged_sents)

    if evaluate:
        # set a split size: use 90% for training, 10% for testing
        split_perc = 0.1
        split_size = int(len(tagged_sents) * split_perc)
    else:
        # use the full corpus in non-evaluate (i.e. full) mode
        split_size = 0

    train_sents, test_sents = tagged_sents[split_size:], tagged_sents[:split_size]

    print('training POS tagger on corpus')
    tagger = ClassifierBasedGermanTagger(train=train_sents)

    if evaluate:
        print('accuracy:', tagger.evaluate(test_sents))

    return tagger


if __name__ == '__main__':
    evaluate = False

    corp = read_corpus()
    taggr = train_tagger(corp, evaluate=evaluate)

    print('saving tagger')
    suffix = '' if evaluate else '_full'
    with open(path.join(OUT_DIR, 'nltk_german_classifier_data{}.pickle'.format(suffix)), 'wb') as f:
        pickle.dump(taggr, f, protocol=2)
