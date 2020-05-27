#!/usr/bin/env python

import argparse
import sys


class Parser(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Single Token Part-of-Speech Tagging using '
                        'Support Vector Machines and Word Embedding Features',
            usage="w2vpos <command> [<args>]\n"
                  "\nexample:\n"
                  "\n``w2vpos tag --svm --model 2017-12-27_15-18-26-774110_sg_50``"
        )
        parser.add_argument(
            'command', type=str, choices=['analyse', 'preprocess', 'train', 'tag', 'evaluate'],
            help='Required w2vpos subprogram.'
        )

        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        getattr(self, args.command)()

    @staticmethod
    def analyse():
        argv = sys.argv[2:]
        from w2v_pos_tagger.corpus_analyser import main
        main(argv)

    @staticmethod
    def preprocess():
        argv = sys.argv[2:]
        from w2v_pos_tagger.dataio import main
        main(argv)

    @staticmethod
    def train():
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--baseline', action='store_true')
        group.add_argument('--word2vec', action='store_true')
        group.add_argument('--svm', action='store_true')
        args = parser.parse_args(sys.argv[2:3])

        if args.baseline:
            from w2v_pos_tagger.nltk_tiger_trainer import main
        elif args.word2vec:
            from w2v_pos_tagger.word2vec import main
        elif args.svm:
            from w2v_pos_tagger.svm_trainer import main
        else:
            raise ValueError

        main(sys.argv[3:])

    @staticmethod
    def tag():
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--baseline', action='store_true')
        group.add_argument('--svm', action='store_true')
        args = parser.parse_args(sys.argv[2:3])

        if args.baseline:
            from w2v_pos_tagger.baseline_tagger import main
        elif args.svm:
            from w2v_pos_tagger.svm_tagger import main
        else:
            raise ValueError

        main(sys.argv[3:])

    @staticmethod
    def evaluate():
        parser = argparse.ArgumentParser()
        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--baseline', action='store_true')
        group.add_argument('--svm', action='store_true')
        args = parser.parse_args(sys.argv[2:3])

        if args.baseline:
            from w2v_pos_tagger.evaluator import baseline as main
        elif args.svm:
            from w2v_pos_tagger.evaluator import svm as main
        else:
            raise ValueError

        main(sys.argv[3:])


def parse():
    Parser()


if __name__ == '__main__':
    parse()
