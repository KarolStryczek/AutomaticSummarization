from .PageRankBasedSummarizer import PageRankBasedSummarizer
from typing import List, Dict
from util import Utils
import numpy as np


class LexRankSummarizer(PageRankBasedSummarizer):
    """
        Implementation of LexRank algorithm.

        It is based in summarization algorithm implemented in PageRankBasedSummarizer
        and implements only sentence similarity calculation function.
    """

    def __init__(self, test_idfs: Dict[str, float] = None) -> None:
        super().__init__()
        if test_idfs is not None:
            self.idfs = test_idfs
        else:
            self.idfs = Utils.get_nkjp_idf_values()
        self.default_idf = max(self.idfs.values())

    def calculate_similarity(self, sentence_x: List[str], sentence_y: List[str]) -> float:
        """
            Sentence similarity implementation for LexRank algorithm.
            It is based on IDF modified cosine similarity of sentences.

            :param sentence_x: Tokenized sentence
            :param sentence_y: Tokenized sentence
            :return: Similarity measure between sentences
        """
        if len(sentence_x)*len(sentence_y) == 0:
            return 0

        x_tfs = Utils.calculate_tf_values(sentence_x)
        y_tfs = Utils.calculate_tf_values(sentence_y)

        a = sum([x_tfs[word] * y_tfs.get(word, 0) * self.get_idf(word)**2 for word in sentence_x])

        b = np.sqrt(sum([(x_tfs[word]*self.get_idf(word))**2 for word in sentence_x]))
        c = np.sqrt(sum([(y_tfs[word]*self.get_idf(word))**2 for word in sentence_y]))

        return a/b/c

    def get_idf(self, word: str) -> float:
        """
            Retrieves IDF value for word from dictionary calculated beforehand.
            If there is no IDF value for requested word, default value is returned.
            Default is calculated as max IDF value from dictionary.

            :param word: Word for which IDF value should be returned
            :return: IDF value for the word
        """
        return self.idfs.get(word, self.default_idf)

