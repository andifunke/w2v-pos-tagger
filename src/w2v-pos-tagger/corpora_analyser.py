from data_loader import *


def get_tagset(df, columns):
    print('getting tagset from corpus')

    df_count = df.groupby(columns).count()
    return {k: v[TOKN_ID] for k, v in df_count.iterrows()}


def compare_to_stts(corpus, df, tagset_key=STTS):
    corp = corpus
    tagset = set(filter(lambda x: x not in REMOVAL, df[tagset_key].unique()))
    print("\n{}/{} tagset:\n{}".format(corp, tagset_key, sorted(tagset)))
    print("length:", len(tagset))
    print("{}/{} tags missing in STTS: {}".format(corp, tagset_key, sorted(tagset - STTS_DEFAULT)))
    print("STTS tags missing in {}/{}: {}\n".format(corp, tagset_key, sorted(STTS_DEFAULT - tagset)))


def main():
    show_sample = False

    print("STTS tagset:\n{}".format(sorted(STTS_DEFAULT)))
    print("length: {}\n".format(len(STTS_DEFAULT)))

    t_df = get_original_corpus(TIGER, raw=True)
    if show_sample:
        tprint(t_df.head(50))

    # analyzing STTS tagset in Tiger
    compare_to_stts(TIGER, t_df)

    h_df = get_original_corpus(HDT, raw=True)
    if show_sample:
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
    REMOVAL = {'_', ''}

    main()
