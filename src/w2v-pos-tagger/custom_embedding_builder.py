#!/usr/bin/env python3

import gensim.models.word2vec as wv

from data_loader import *


def sentences_from_corpus(corpus, lowercase=False):
    """ returns a list of lists of strings.
    The outer list contains all sentences while the inner lists contain all tokens as string. """
    df = get_preprocessed_corpus(corpus)

    print('\n>>> getting sentences from {}{}'.format(corpus, ' - lowercase' if lowercase else ''))
    sentences = []
    sentence = []
    last_sentence_id = 1
    for row in df.itertuples():
        sentence_id = row[2]
        token = row[4]
        if sentence_id > last_sentence_id:
            sentences.append(sentence)
            sentence = []
            last_sentence_id = sentence_id
        sentence.append(token.lower() if lowercase else token)
    sentences.append(sentence)
    return sentences


def custom_embedding_builder_main():
    """ training a word2vec model from both corpora.
        Different dimensionality can be used as well as different preprocessing (lowercase or not)
        as well as the CBOW and the Skip-gram model.
        Change any other parameter to taste.
    """
    t0 = time()

    for lowercase in [False, True]:
        lc = '_lc' if lowercase else ''
        sentences = sentences_from_corpus(TIGER, lowercase) + sentences_from_corpus(HDT, lowercase)
        for sg in [0, 1]:
            mdl = 'cb' if sg == 0 else 'sg'
            for size in [12, 25, 50, 100]:
                    print("starting w2v {} modelling, size={:d}".format(mdl, size))
                    model = wv.Word2Vec(sentences, size=size, alpha=0.025, window=5, min_count=0, max_vocab_size=None,
                                        sample=0.001, seed=1, workers=4, min_alpha=0.0001, sg=sg, hs=0, negative=5,
                                        cbow_mean=1, iter=5, null_word=0, trim_rule=None, sorted_vocab=1,
                                        batch_words=10000, compute_loss=False)

                    fname = path.join(CORPORA_DIR, 'custom_embedding_{}_{:d}{}_xyz.test'.format(mdl, size, lc))
                    print("saving vectors to {}".format(fname))
                    model.save(fname)

    print("done in {:f}s".format(time() - t0))


if __name__ == '__main__':
    custom_embedding_builder_main()
