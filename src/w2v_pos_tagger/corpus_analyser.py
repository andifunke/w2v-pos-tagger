#!/usr/bin/env python3

"""
Analyses the used tagsets from the TIGER and HDT corpora.

Additionally provides a mapping for the reduced HDT tagset.
"""

import pandas as pd

from w2v_pos_tagger.constants import TIGER, HDT, TOKN_ID, STTS, REDU, STTS_DEFAULT
from w2v_pos_tagger.dataio import get_original_corpus, tprint


def get_tagset(df, columns):
    """Returns the tagset of a given corpus as dict."""

    print('getting tagset from corpus')
    df_count = df.groupby(columns).count()
    return {k: v[TOKN_ID] for k, v in df_count.iterrows()}


def compare_to_stts(corpus, df, tagset_key=STTS):
    """Compare the tagset of a corpus to the default STTS tagset. Print the results."""

    corp = corpus
    tagset = set(filter(lambda x: x not in {'_', ''}, df[tagset_key].unique()))
    print(f"\n{corp}/{tagset_key} tagset:\n{sorted(tagset)}")
    print(f"length: {len(tagset)}")
    print(f"{corp}/{tagset_key} tags missing in STTS: {sorted(tagset - STTS_DEFAULT)}")
    print(f"STTS tags missing in {corp}/{tagset_key}: {sorted(STTS_DEFAULT - tagset)}\n")


def main():
    print(f"STTS tagset:\n{sorted(STTS_DEFAULT)}")
    print(f"length: {len(STTS_DEFAULT)}\n")

    t_df = get_original_corpus(TIGER, raw=True)
    tprint(t_df.head(50))

    # analyzing STTS tagset in Tiger
    compare_to_stts(TIGER, t_df)

    h_df = get_original_corpus(HDT, raw=True)
    tprint(h_df.head(50))

    # analyzing STTS tagset in HDT
    compare_to_stts(HDT, h_df)
    # analyzing reduced tagset in Hdt
    compare_to_stts(HDT, h_df, tagset_key=REDU)

    # this prints a mapping from the HDT STTS tagset to the HDT reduced tagset
    h_mapping = dict(filter(lambda x: x[0] != x[1], get_tagset(h_df, [STTS, REDU]).keys()))
    h_map_df = pd.DataFrame.from_dict(h_mapping, orient='index').rename(columns={0: 'reduced tag'})
    print("HDT mapping from STTS to the reduced tagset (where different):")
    print(h_map_df.sort_values(by='reduced tag'))


if __name__ == '__main__':
    main()
