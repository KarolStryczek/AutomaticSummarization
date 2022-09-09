from .ExtractiveSummarizer import ExtractiveSummarizer
from typing import List, Dict
from abc import ABC


class FrequencyBasedSummarizer(ExtractiveSummarizer, ABC):
    """
        Abstract super class for frequency based extractive summarizers.

        Contains common method used in specific frequency based extractive summarizers implementations.
    """

    @staticmethod
    def calculate_tf_values(sentences: List[List[str]]) -> Dict[str, float]:
        """
            Calculates words TF values as number of occurrences divided by words total count.

            :param sentences: Tokenized list of sentences
            :return: Dictionary of (words: TF) pairs
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
