from .PageRankBasedSummarizer import PageRankBasedSummarizer
from util import Utils
import numpy as np


class LexRankSummarizer(PageRankBasedSummarizer):
    def __init__(self):
        super().__init__()
        self.__idf = Utils.get_nkjp_idf_values()
        self.default_idf = max(self.__idf.values())

    def calculate_sentence_similarity(self, sentence_x, sentence_y):
        if len(sentence_x)*len(sentence_y) == 0:
            return 0

        x_tfs = Utils.calculate_tf_values(sentence_x)
        y_tfs = Utils.calculate_tf_values(sentence_y)

        a = sum([x_tfs[word] * y_tfs.get(word, 0) * self.idf(word)**2 for word in sentence_x])

        b = np.sqrt(sum([(x_tfs[word]*self.idf(word))**2 for word in sentence_x]))
        c = np.sqrt(sum([(y_tfs[word]*self.idf(word))**2 for word in sentence_y]))

        return a/b/c

    def idf(self, word):
        return self.__idf.get(word, self.default_idf)

