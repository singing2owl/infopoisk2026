from collections import defaultdict, Counter
import math
from dancebooks import bib_parser, config
from dancebooks.config import config
import infopoisk_data_prep

def build_inverted_index(corpus):
    """
    corpus: dict[id -> list of tokens]
    
    returns:
    dict[term → [(doc_id, tf), (doc_id, tf), ...]]
    """

    index = defaultdict(list)

    for doc_id, tokens in corpus.items():

        counts = Counter(tokens)

        for token, freq in counts.items():
            index[token].append((doc_id, freq))

    return dict(index)

def search_tf(index, query_tokens, k=5):

    scores = defaultdict(int)

    for token in query_tokens:

        if token not in index:
            continue

        for doc_id, tf in index[token]:
            scores[doc_id] += tf

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[:k]

def compute_idf(index, N):

    idf = {}

    for token, postings in index.items():

        df = len(postings)

        idf[token] = math.log((N - df + 0.5) / (df + 0.5) + 1)

    return idf

def compute_doc_stats(corpus):
    """
    corpus: dict[doc_id -> list(tokens)]
    """
    doc_len = {doc_id: len(tokens) for doc_id, tokens in corpus.items()}
    avgdl = sum(doc_len.values()) / len(doc_len)
    N = len(corpus)

    return doc_len, avgdl, N

def search_bm25(index, idf, query_tokens, doc_len, avgdl, k=5, k1=1.5, b=0.75):

    scores = defaultdict(float)

    for token in query_tokens:

        if token not in index:
            continue

        for doc_id, tf in index[token]:

            dl = doc_len[doc_id]

            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * dl / avgdl)

            scores[doc_id] += idf[token] * numerator / denominator

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return ranked[:k]

def preprocess(text):
    return infopoisk_data_prep.lemmatize_with_cleaning(text, lang=("ru", "en"))

if __name__ == "__main__":

    items, item_index = bib_parser.BibParser().parse_folder(config.parser.bibdata_dir)
    param_list = ['title',
                  'author',
                  'altauthor',
                  'booktitle',
                  'incipit',
                  'journaltitle',
                  'keywords',
                  'langid', 
                  'location',
                  'origauthor',
                  'origlanguage',
                  'pseudo_author',
                  'translator',
                  'type']
    
    # This is for russian data
    # TODO: make it more general for other languages (e.g. by using lang_map from infopoisk_data_prep)
    ru_data_dict = infopoisk_data_prep.parse_folder_into_json(config.parser.bibdata_dir, param_list)["russian.bib"]
    
    # Preprocess and lemmatize each document
    corpus = {doc_id: preprocess(text) for doc_id, text in ru_data_dict.items()}
    
    inverted_index = build_inverted_index(corpus)
    
    # # Print some entries from the inverted index
    # for token in list(inverted_index.keys())[:3]:  # print first 3 tokens
    #     print(f"{token}: {inverted_index[token]}")

    query = input("Enter search query: ")
    search_type = input("Choose search type (tf/bm25): ").strip().lower()
    query_tokens = preprocess(query)
    doc_len, avgdl, N = compute_doc_stats(corpus)
    idf = compute_idf(inverted_index, N)
    if search_type == "tf":
        results = search_tf(inverted_index, query_tokens)
    elif search_type == "bm25":
        results = search_bm25(inverted_index, idf, query_tokens, doc_len, avgdl)
    else:
        print("Invalid search type. Defaulting to TF.")
        results = search_tf(inverted_index, query_tokens)

    print("Search results:")
    for doc_id, score in results:
        print(f"{doc_id}: {score}")
