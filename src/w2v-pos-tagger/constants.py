SPACY = 'SPACY'
NLTK = 'NLTK'
TIGER = 'TIGER'
HDT = 'HDT'
MINIMAL = 'MINIMAL'
DEFAULT = 'DEFAULT'
PREPROCESSED = 'PREPROCESSED'
PREDICTIONS = 'SELFTAGGED'
ID = 'ID'
SENT_ID = 'SENT_ID'
TOKN_ID = 'TOKN_ID'
FORM = 'FORM'
LEMM = 'LEMM'
STTS = 'STTS'
REDU = 'REDU'
UNIV = 'UNIV'
CORP = 'CORP'
PRED_TXT = 'PRED_TXT'
PRED_TAG = 'PRED_TAG'
GOLD_TXT = 'GOLD_TXT'
GOLD_TAG = 'GOLD_TAG'
TP = 'True Pos'
TN = 'True Neg'
FP = 'False Pos'
FN = 'False Neg'
PREC = 'Precision'
RECL = 'Recall'
F1 = 'F1 score'

# Tiger CONLL09 columns
KEYS = {
    TIGER: [
        TOKN_ID, FORM, LEMM, 'PLEMMA', STTS, 'PPOS', 'FEAT', 'PFEAT', 'HEAD', 'PHEAD', 'DEPREL',
        'PDEPREL', 'FILLPRED', 'PRED', 'APREDS'
    ],
    HDT: [TOKN_ID, FORM, LEMM, REDU, STTS, 'FEAT', 'HEAD', 'DEPREL', 'UNKNOWN_1', 'UNKNOWN_2'],
    MINIMAL: [TOKN_ID, FORM, LEMM, STTS],
    DEFAULT: [CORP, SENT_ID, TOKN_ID, FORM, LEMM, STTS, UNIV],
    SPACY: [ID, FORM, LEMM, 'POS', STTS, 'DEP', 'SHAPE', 'ALPHA', 'STOP'],
    NLTK: [ID, FORM, STTS],
    PREDICTIONS: [CORP, SENT_ID, TOKN_ID, FORM, STTS, UNIV]
}

"""
These are the common tags from STTS website. PAV was replaced by PROAV since all 
corpora do so as well. The mapping is according to de-tiger.map
https://github.com/slavpetrov/universal-pos-tags/blob/master/de-tiger.map
PIDAT -> PRON according to de-negra.map
https://github.com/slavpetrov/universal-pos-tags/blob/master/de-negra.map
NNE was removed from the mapping since no official STTS tag.
"""
STTS_UNI_MAP = {
    '$(': '.',
    '$,': '.',
    '$.': '.',
    'ADJA': 'ADJ',
    'ADJD': 'ADJ',
    'ADV': 'ADV',
    'APPO': 'ADP',
    'APPR': 'ADP',
    'APPRART': 'ADP',
    'APZR': 'ADP',
    'ART': 'DET',
    'CARD': 'NUM',
    'FM': 'X',
    'ITJ': 'X',
    'KOKOM': 'CONJ',
    'KON': 'CONJ',
    'KOUI': 'CONJ',
    'KOUS': 'CONJ',
    'NE': 'NOUN',
    'NN': 'NOUN',
    'PDAT': 'PRON',
    'PDS': 'PRON',
    'PIDAT': 'PRON',
    'PIAT': 'PRON',
    'PIS': 'PRON',
    'PPER': 'PRON',
    'PPOSAT': 'PRON',
    'PPOSS': 'PRON',
    'PRELAT': 'PRON',
    'PRELS': 'PRON',
    'PRF': 'PRON',
    'PROAV': 'PRON',
    'PTKA': 'PRT',
    'PTKANT': 'PRT',
    'PTKNEG': 'PRT',
    'PTKVZ': 'PRT',
    'PTKZU': 'PRT',
    'PWAT': 'PRON',
    'PWAV': 'PRON',
    'PWS': 'PRON',
    'TRUNC': 'X',
    'VAFIN': 'VERB',
    'VAIMP': 'VERB',
    'VAINF': 'VERB',
    'VAPP': 'VERB',
    'VMFIN': 'VERB',
    'VMINF': 'VERB',
    'VMPP': 'VERB',
    'VVFIN': 'VERB',
    'VVIMP': 'VERB',
    'VVINF': 'VERB',
    'VVIZU': 'VERB',
    'VVPP': 'VERB',
    'XY': 'X'
}
STTS_TAGS = STTS_UNI_MAP.keys()
STTS_DEFAULT = set(STTS_TAGS - {'PROAV'}).union({'PAV'})

# universal tagset
UNIV_TAGS = {
    '.': 0,
    'ADJ': 1,
    'ADV': 2,
    'ADP': 3,
    'DET': 4,
    'NUM': 5,
    'CONJ': 6,
    'NOUN': 7,
    'PRON': 8,
    'PRT': 9,
    'VERB': 10,
    'X': 11
}
UNIV_TAGS_BACKWARDS = {(v, k) for k, v in UNIV_TAGS.items()}

# corpus fixes
CORPUS_BUGS = {'NNE': 'NE', 'PPOSSAT': 'PPOSAT', 'VAIZU': 'VVIZU'}
STTS_UNI_MAP_EXTENDED = STTS_UNI_MAP.copy()
STTS_UNI_MAP_EXTENDED.update({'NNE': 'NOUN', 'PPOSSAT': 'PRON', 'VAIZU': 'VERB'})
