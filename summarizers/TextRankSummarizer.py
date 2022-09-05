from .PageRankBasedSummarizer import PageRankBasedSummarizer
import numpy as np


class TextRankSummarizer(PageRankBasedSummarizer):

    def calculate_sentence_similarity(self, sentence_x, sentence_y):
        if len(sentence_x)*len(sentence_y) == 0:
            return 0

        similarity = 0
        for word in sentence_x:
            if word in sentence_y:
                similarity += 1
        similarity /= (np.log10(len(sentence_x)*len(sentence_y)))
