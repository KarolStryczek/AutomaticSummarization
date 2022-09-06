from .ExtractiveSummarizer import ExtractiveSummarizer
from abc import ABC, abstractmethod
import numpy as np
from util import Utils


class PageRankBasedSummarizer(ExtractiveSummarizer, ABC):

    def __init__(self):
        super().__init__()
        self.d = 0.85
        self.steps = 10

    def summarize(self, text, n, percent=None):
        sentences_text = self.get_sentences(text)
        sentences_cleaned = self.clean_sentences(sentences_text)
        matrix = self.__calculate_similarity_matrix(sentences_cleaned)
        scores = np.array([1] * len(sentences_cleaned))

        for epoch in range(self.steps):
            scores = (1 - self.d) + self.d * np.dot(matrix, scores)

        return self.prepare_summary(sentences_text, Utils.get_ranking(scores, n))

    def __calculate_similarity_matrix(self, tokenized_sentences):
        n = len(tokenized_sentences)
        matrix = np.zeros([n, n])
        for i in range(n):
            sentence_i = tokenized_sentences[i]
            for j in range(i + 1, n):
                sentence_j = tokenized_sentences[j]
                similarity = self.calculate_sentence_similarity(sentence_i, sentence_j)
                matrix[i, j] = similarity
                matrix[j, i] = similarity

        norm = np.sum(matrix, axis=0)
        matrix = np.divide(matrix, norm, where=norm != 0)

        return matrix

    @abstractmethod
    def calculate_sentence_similarity(self, sentence_x, sentence_y):
        pass
