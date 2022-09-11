from .PageRankBasedSummarizer import PageRankBasedSummarizer
from typing import List
import numpy as np


class TextRankSummarizer(PageRankBasedSummarizer):
    """
        Implementation of TextRank algorithm.

        It is based in summarization algorithm implemented in PageRankBasedSummarizer
        and implements only sentence similarity calculation function.
    """

    def calculate_similarity(self, sentence_x: List[str], sentence_y: List[str]) -> float:
        """
            Sentence similarity implementation for TextRank algorithm.
            It is based on normalized sum of common tokens in compared sentences.

        :param sentence_x: Tokenized sentence
        :param sentence_y: Tokenized sentence
        :return: Similarity measure between sentences
        """
        if len(sentence_x)*len(sentence_y) == 0:
            return 0

        similarity = 0
        for word in sentence_x:
            if word in sentence_y:
                similarity += 1
        similarity /= (np.log10(len(sentence_x)*len(sentence_y)))

        return similarity
