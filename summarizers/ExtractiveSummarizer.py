from .AbstractSummarizer import AbstractSummarizer
from typing import List
from util import Utils
from abc import ABC
import nltk


class ExtractiveSummarizer(AbstractSummarizer, ABC):
    """
        Abstract super class for all extractive summarizers.

        Contains common methods used in specific implementations.
    """

    def __init__(self) -> None:
        self.language = "polish"

    @staticmethod
    def prepare_summary(sentences_text: List[str], indexes: List[int]) -> str:
        """
            Prepares summary based on list of all sentences and indexes chosen by algorithm to be included in summary.

            :param sentences_text: List of sentences
            :param indexes: List of indexes to be included in summary
            :return: Generated summary
        """
        return "\n".join([sentences_text[i] for i in sorted(indexes)])

    def get_sentences(self, text: str) -> List[str]:
        """
            Splits input text into list of sentences.

            :param text: Text to split
            :return: List of sentences
        """
        return nltk.sent_tokenize(text, language=self.language)

    @staticmethod
    def clean_sentences(sentences: List[str]) -> List[List[str]]:
        """
            Tokenizes sentences and cleans words after that.

            :param sentences: List of sentences
            :return: List of cleaned and tokenized sentences
        """
        words = list()
        for sentence in sentences:
            words.append(Utils.tokenize_and_clean_text(sentence))
        return words
