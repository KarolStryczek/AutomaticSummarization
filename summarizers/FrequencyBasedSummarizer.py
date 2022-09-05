from .ExtractiveSummarizer import ExtractiveSummarizer
from typing import List, Dict
from abc import ABC


class FrequencyBasedSummarizer(ExtractiveSummarizer, ABC):
    """
        Super class for frequency based extractive summarizers.

        Contains common method used in specific frequency based extractive summarizers implementation like:
         - tokenization
         - data cleaning
         - stopwords removal
         - metrics calculation (e.g. TF)
    """

    @staticmethod
    def calculate_tf_values(sentences: List[List[str]]) -> Dict[str, float]:
        """
            Calculates words TF values as number of occurrences divided by words total count.

            :param sentences: list of sentences (sentence as list of words)
            :return: dictionary of (words, TF) pairs
        """
        words_cnt = sum([len(sentence) for sentence in sentences])
        tfs = dict()
        for sentence in sentences:
            for word in sentence:
                if word not in tfs:
                    tfs[word] = 1
                else:
                    tfs[word] += 1
        for word in tfs:
            tfs[word] /= words_cnt
        return tfs
