from data_loader import *
import spacy
import nltk
import pickle
import sys


def run_tagger(corpus_name, corpus_df, nlp_framework, dry_run=True):
    tprint(corpus_df, 20)
    print('\n>>> starting {1} POS tagging for {0}'.format(corpus_name, nlp_framework))
    file = path.join(OUT_DIR, '{}_pos_by_{}.csv'.format(corpus_name, nlp_framework))
    tagging = tag_sentence_with_spacy if nlp_framework == SPACY else tag_sentence_with_nltk

    def handle_sentence(sent, s_id):
        tagged_lines = tagging(sent, s_id, corpus_name)
        string = '\n'.join(tagged_lines) + '\n'
        if dry_run:
            print(string, end='')
        else:
            # be careful: appending to existing file!
            fw.write(string)

    with open(file, 'a') as fw:
        sentence = []
        last_sentence_id = 1
        for row in corpus_df.itertuples():
            sentence_id = row[2]
            token = row[4]
            if sentence_id > last_sentence_id:
                handle_sentence(sentence, last_sentence_id)
                sentence = []
                last_sentence_id = sentence_id
            sentence.append(token)
        handle_sentence(sentence, last_sentence_id)


def tag_sentence_with_spacy(sentence, sentence_index, corpus):
    doc = SPACY_TAGGER.tokenizer.tokens_from_list(sentence)
    SPACY_TAGGER.tagger(doc)
    return ['{0}\t{1:d}\t{2:d}\t{3}\t{4}\t{5}'
            .format(corpus, sentence_index, token_index, token.text, token.tag_, STTS_UNI_MAP_EXTENDED[token.tag_])
            for token_index, token in enumerate(doc, 1)]


def tag_sentence_with_nltk(sentence, sentence_index, corpus):
    doc = NLTK_TAGGER.tag(sentence)
    return ['{0}\t{1:d}\t{2:d}\t{3}\t{4}\t{5}'
            .format(corpus, sentence_index, token_index, token[0], token[1], STTS_UNI_MAP_EXTENDED[token[1]])
            for token_index, token in enumerate(doc, 1)]


def main():
    print('loading', TIGER, 'corpus')
    t = TIGER, get_preprocessed_corpus(TIGER)

    print('loading', HDT, 'corpus')
    h = HDT, get_preprocessed_corpus(HDT)

    dry_run = (len(sys.argv) > 1 and sys.argv[1] == '--dry_run')
    run_tagger(*t, nlp_framework=SPACY, dry_run=dry_run)
    run_tagger(*h, nlp_framework=SPACY, dry_run=dry_run)
    run_tagger(*t, nlp_framework=NLTK, dry_run=dry_run)
    run_tagger(*h, nlp_framework=NLTK, dry_run=dry_run)


if __name__ == '__main__':
    t0 = time()
    print('loading', SPACY)
    SPACY_TAGGER = spacy.load('de')

    print('loading', NLTK)
    nltk.data.load('tokenizers/punkt/german.pickle')
    with open(path.join(DATA_DIR, 'nltk_german_classifier_data_full.pickle'), 'rb') as f:
        NLTK_TAGGER = pickle.load(f)

    main()
    print("done in {:f}s".format(time() - t0))
