#!/usr/bin/env python3

"""
Learns word vectors from the combines TIGER and HDT corpus by applying word2vec.
"""
import argparse
import multiprocessing as mp
from time import time
from typing import List

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm

from constants import TIGER, HDT, SENT_ID, FORM
from data_loader import get_preprocessed_corpus, EMBEDDINGS_DIR


def parse_args() -> argparse.Namespace:
    """
    Parses module-specific arguments. Solves argument dependencies
    and returns cleaned up arguments.

    :returns: arguments object
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-a', '--architecture', type=str, nargs='+', default=['cb', 'sg'],
        choices=['cb', 'sg'],
        help="Choice(s) for word2vec architecture: cbow ('cb') and/or skip-gram ('sg')."
    )
    parser.add_argument(
        '-c', '--case-folding', type=str, nargs='+', default=['none', 'lower'],
        choices=['none', 'lower'],
        help="Choice(s) for case folding: no case folding ('none') or lowercasing ('lower')."
    )
    parser.add_argument(
        '-d', '--dimensionality', type=int, nargs='+', default=[12, 25, 50, 100],
        help="Choice(s) for embedding sizes: list of int > 0."
    )
    parser.add_argument(
        '-e', '--epochs', type=int, default=5,
        help="Number of word2vec training iterations."
    )
    args = parser.parse_args()

    print(args)
    args.architecture = [int(a == 'sg') for a in args.architecture]
    args.case_folding = [c == 'lower' for c in args.case_folding]

    return args


class EpochLogger(CallbackAny2Vec):
    """Callback to log information about training"""

    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f"Epoch #{self.epoch} started")

    def on_epoch_end(self, model):
        self.epoch += 1


def sentences_from_corpus(corpus: str, lowercase: bool = False) -> List[List[str]]:
    """
    Extract sentences from a corpus.

    :returns: a list of lists of strings. The outer list contains all sentences while
        the inner lists contain all tokens as string.
    """

    print(f"\n>>> Getting sentences from {corpus}{' - lowercase' if lowercase else ''}")
    df = get_preprocessed_corpus(corpus)

    groupby = df.groupby(SENT_ID, sort=False)[FORM]
    if lowercase:
        sentences = groupby.progress_apply(lambda grp: [t.lower() for t in grp])
    else:
        sentences = groupby.progress_apply(lambda grp: grp.tolist())
    sentences = sentences.tolist()

    return sentences


def main():
    """
    Trains a word2vec model from both corpora.

    Hyperparameters:
    - architecture (skip-gram and cbow)
    - case folding (lowercasing or none)
    - dimensionality (embedding sizes)
    - number of epochs

    The number of trained embeddings will be the cartesian product of the
    three lists of hyperparameters: len(a) x len(c) x len(d)
    """
    args = parse_args()

    for lowercase in args.case_folding:
        lc = '_lc' if lowercase else ''
        sentences = sentences_from_corpus(TIGER, lowercase) + sentences_from_corpus(HDT, lowercase)

        for sg in args.architecture:
            mdl = 'cb' if sg == 0 else 'sg'

            for size in args.dimensionality:
                t0 = time()
                print(f"\nstarting w2v {mdl} modelling, size={size}")

                workers = mp.cpu_count()
                epoch_logger = EpochLogger()
                model = Word2Vec(
                    sentences, size=size, alpha=0.025, window=5, min_count=0, max_vocab_size=None,
                    sample=0.001, seed=1, workers=workers, min_alpha=0.0001, sg=sg, hs=0,
                    negative=5, cbow_mean=1, iter=args.epochs, null_word=0, trim_rule=None,
                    sorted_vocab=1, batch_words=10000, compute_loss=False, callbacks=[epoch_logger]
                )

                file_name = EMBEDDINGS_DIR / f'{mdl}_{size:03d}{lc}.w2v'
                EMBEDDINGS_DIR.mkdir(exist_ok=True)
                print(f"saving vectors to {file_name}")
                model.callbacks = ()
                model.save(str(file_name))
                print(f"Done in {time() - t0:.2f}s")


if __name__ == '__main__':
    tqdm.pandas()

    main()
