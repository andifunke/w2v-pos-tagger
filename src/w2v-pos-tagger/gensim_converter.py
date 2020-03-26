import re

from data_loader import *
from os import listdir
import gensim.models.word2vec as wv
import six.moves.cPickle as cPickle


def gensim_wrapper_main():
    """
    since gensim cannot be installed on the hpc this small script replaces
    gensim KeyedVectors by plain old vanilla python dicts
    """

    files = [f for f in listdir(DATA_DIR) if re.match('custom_embedding_.*\.vec$', f)]

    for fname in files:
        print('loading model', fname)
        model = wv.Word2Vec.load(path.join(DATA_DIR, fname))
        print('building dict')
        vector_dict = {word: model.wv[word] for word in model.wv.vocab}
        print('saving dict')
        with open(path.join(DATA_DIR, fname + '.pickle'), 'wb') as f:
            cPickle.dump(vector_dict, f)


if __name__ == '__main__':
    gensim_wrapper_main()
