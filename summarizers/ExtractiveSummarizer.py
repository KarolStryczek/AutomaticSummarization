from abc import ABC
from spacy.tokens.token import Token
from .AbstractSummarizer import AbstractSummarizer
import nltk
from typing import List
import spacy
from util import Utils


class ExtractiveSummarizer(AbstractSummarizer, ABC):

    def __init__(self):
        self.nlp = spacy.load("pl_core_news_sm")
        self.language = "polish"

    @staticmethod
    def prepare_summary(sentences_text: List[str], indexes: List[int]) -> str:
        """
            Prepares summary based on sentence list and indexes chosen by algorithm to be included in summary.

        :param sentences_text: text split into list of sentences
        :param indexes: list of indexes to be included in summary
        :return: summary
        """
        return "\n".join([sentences_text[i] for i in sorted(indexes)])

    def get_sentences(self, text: str) -> List[str]:
        """
            Splits input text into list of sentences.

            :param text: text to split
            :return: list of sentences
        """
        return nltk.sent_tokenize(text, language=self.language)

    def clean_sentences(self, sentences: List[str]) -> List[List[str]]:
        """
            Splits sentences into words and cleaning words after that.

            :param sentences: list of sentences
            :return: list of cleaned sentences (sentence as list of words)
        """
        words = list()
        for sentence in sentences:
            words.append(Utils.tokenize_and_clean_text(sentence))
        return words

    @staticmethod
    def __should_include_token(token: Token) -> bool:
        """
            Determines if token should be included in tokens list.
            Removes stopwords, whitespaces and punctuation marks..

        :param token: spaCy token to be checked
        :return: flag indicating if token should be included in tokens list
        """
        return not token.is_stop and not token.is_space and not token.is_punct