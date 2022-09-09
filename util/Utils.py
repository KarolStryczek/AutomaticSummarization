import math
import os
import pickle
import re
import spacy

from typing import List
from bs4 import BeautifulSoup
from spacy.tokens.token import Token

nlp = spacy.load("pl_core_news_sm")


# def read_sample_file(filename):
#     with open("data/texts/" + filename, encoding=__default_encoding) as f:  # TODO relative path
#         return f.read()
#
#
# def __get_path(filename, subdirectory=None):
#     directory = os.path.dirname(__file__)
#     if subdirectory is not None:
#         directory = os.path.join(os.path.dirname(__file__), subdirectory)
#     return os.path.join(directory, filename)


__default_language = 'polish'
__default_encoding = 'utf-8'
__NKJP_dir = 'data/NKJP'
__idf_filepath = 'util/idf.p'


def tokenize_and_clean_text(text: str) -> List[str]:
    """
        Text tokenization with lemmatization, lowercase, stopwords and punctuation marks removal.

        :param text: input text to tokenize and clean
        :return: list of cleaned tokens
    """
    return [token.lemma_.lower() for token in nlp(text) if should_include_token(token)]


def should_include_token(token: Token) -> bool:
    """
        Determines if token should be included in tokens list.
        Removes stopwords, whitespaces and punctuation marks..

    :param token: spaCy token to be checked
    :return: flag indicating if token should be included in tokens list
    """
    return not token.is_stop and not token.is_space and not token.is_punct


def get_nkjp_idf_values():
    try:
        return pickle.load(open(__idf_filepath, mode="rb"))
    except FileNotFoundError:
        idfs = calculate_idf_values(load_all_NKJP_texts())
        pickle.dump(idfs, open(__idf_filepath, mode="wb"))
        return idfs


def load_all_NKJP_texts():
    all_dirs = get_all_subdirectories(__NKJP_dir)
    texts = list()
    for directory in all_dirs:
        with open(directory + '/text.xml', encoding=__default_encoding) as f:
            texts.append(read_and_clean_xml_file(f))
    return texts


def get_all_subdirectories(nkjp_dir):
    return [f.path for f in os.scandir(nkjp_dir) if f.is_dir()]


def read_and_clean_xml_file(file):
    text = BeautifulSoup(file, features='xml').TEI.text
    text = re.sub('\n', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def calculate_idf_values(corpus):
    idfs = dict()
    for text in corpus:
        words = tokenize_and_clean_text(text)
        unique_words = set(words)
        for word in unique_words:
            if word not in idfs:
                idfs[word] = 1
            else:
                idfs[word] += 1
    return {k: math.log10(len(corpus)/v) for k, v in idfs.items()}


# def get_bag_of_words(corpus, sorted_list=False):
#     bow = dict()
#     for text in corpus:
#         words = tokenize_and_clean_text(text)
#         for word in words:
#             if word not in bow:
#                 bow[word] = 1
#             else:
#                 bow[word] += 1
#     return sort_dict_by_value(bow) if sorted_list else bow


def sort_dict_by_value(dic):
    return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}


def get_ranking(scores, n):
    scores_dict = {i: scores[i] for i in range(len(scores))}
    sorted_scores = sort_dict_by_value(scores_dict)
    return list(sorted_scores.keys())[:n]


def calculate_tf_values(tokenized_sentence):
    tfs = dict()
    for word in tokenized_sentence:
        if word not in tfs:
            tfs[word] = 1
        else:
            tfs[word] += 1
    return tfs
