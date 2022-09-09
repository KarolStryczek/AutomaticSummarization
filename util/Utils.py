from spacy.tokens.token import Token
from typing import List, Dict, Any, TextIO
from bs4 import BeautifulSoup
import pickle
import spacy
import math
import os
import re

nlp = spacy.load("pl_core_news_sm")


__default_language = 'polish'
__default_encoding = 'utf-8'
__NKJP_dir = 'data/NKJP'
__idf_filepath = 'util/idf.p'


def get_nkjp_idf_values() -> Dict[str, float]:
    """
        Calculates IDF values for all words from texts in NKJP one million sub-corpus.
        Values are serialized after calculation so they cen be used again without calculating them again.

        :return: Dictionary of (word: IDF value) pairs
    """
    try:
        return pickle.load(open(__idf_filepath, mode="rb"))
    except FileNotFoundError:
        idfs = calculate_idf_values(load_all_NKJP_texts())
        pickle.dump(idfs, open(__idf_filepath, mode="wb"))
        return idfs


def load_all_NKJP_texts() -> List[str]:
    """
        Loads all texts from NKJP directory. That includes XML parsing, and pre-processing.

        :return: List of loaded texts
    """
    all_dirs = get_all_subdirectories(__NKJP_dir)
    texts = list()
    for directory in all_dirs:
        with open(directory + '/text.xml', encoding=__default_encoding) as f:
            texts.append(read_and_clean_nkjp_xml_file(f))
    return texts


def get_all_subdirectories(directory: str) -> List[str]:
    """
        Returns all subdirectories of given directory.

        :param directory: The main directory
        :return: List of subdirectories
    """
    return [f.path for f in os.scandir(directory) if f.is_dir()]


def read_and_clean_nkjp_xml_file(file: TextIO) -> str:
    """
        Loads and cleand NKJP file in TEI standard.

        :param file: NKJP file to be parsed and cleaned.
        :return: NKJP file text.
    """
    text = BeautifulSoup(file, features='xml').TEI.text
    text = re.sub('\n', ' ', text)
    text = re.sub(' +', ' ', text)
    return text.strip()


def calculate_idf_values(corpus: List[str]) -> Dict[str, float]:
    """
        Calculates IDF values for words within given corpus.

        :param corpus: Corpus of texts
        :return: Dictionary of (word: IDF value) pairs
    """
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


def tokenize_and_clean_text(text: str) -> List[str]:
    """
        Text tokenization with lemmatization, lowercase, stopwords and punctuation marks removal.

        :param text: Input text to tokenize and clean
        :return: List of cleaned tokens
    """
    return [token.lemma_.lower() for token in nlp(text) if should_include_token(token)]


def should_include_token(token: Token) -> bool:
    """
        Determines if token should be included in tokens list.
        Removes stopwords, whitespaces and punctuation marks.

        :param token: spaCy token to be checked
        :return: Flag indicating if token should be included in tokens list
    """
    return not token.is_stop and not token.is_space and not token.is_punct


def get_ranking(scores: List[float], n: int) -> List[int]:
    """
        Returns indexes of top ranked sentences based on their scores.

        :param scores: List of scores calculated for sentences
        :param n: Number of top indexes to return
        :return: List of top indexes sorted by score
    """
    scores_dict = {i: scores[i] for i in range(len(scores))}
    sorted_scores = sort_dict_by_value(scores_dict)
    return list(sorted_scores.keys())[:n]


def sort_dict_by_value(dic: Dict[Any, Any]) -> Dict[Any, Any]:
    """
        Sorts dictionary by its values in reversed order.

        :param dic: Dictionary to be sorted
        :return: Sorted dictionary
    """
    return {k: v for k, v in sorted(dic.items(), key=lambda item: item[1], reverse=True)}


def calculate_tf_values(tokenized_sentence: List[str]) -> Dict[str, int]:
    """
        Calculates TF values for words in given sentence.

        :param tokenized_sentence: Tokenized sentence
        :return: Dictionary of (token: TF value) pairs
    """
    tfs = dict()
    for word in tokenized_sentence:
        if word not in tfs:
            tfs[word] = 1
        else:
            tfs[word] += 1
    return tfs
