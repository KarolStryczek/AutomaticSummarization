from .ExtractiveSummarizer import ExtractiveSummarizer
from abc import ABC, abstractmethod
from typing import List
from util import Utils
import numpy as np


class PageRankBasedSummarizer(ExtractiveSummarizer, ABC):
    """
        Abstract super class for PageRank-based summarizers.
        Contains summarization algorithm that includes operation on similarity matrix
        that is calculated in the way corresponding to each algorithm.

        PageRank-based algorithms implementation should implement sentence similarity calculation function.
    """

    def __init__(self) -> None:
        super().__init__()
        self.d = 0.85
        self.steps = 10

    def summarize(self, text: str, size: int) -> str:
        """
            PageRank ranking algorithm is applied to graph-based representation of text.
            N top ranked sentences are extracted as summary.

            :param text:
            :param size:
            :return:
        """
        sentences_text = self.get_sentences(text)
        sentences_cleaned = self.clean_sentences(sentences_text)
        matrix = self.create_similarity_matrix(sentences_cleaned)
        scores = np.array([1] * len(sentences_cleaned))

        for epoch in range(self.steps):
            scores = (1 - self.d) + self.d * np.dot(matrix, scores)

        return self.prepare_summary(sentences_text, Utils.get_ranking(scores, size))

    def create_similarity_matrix(self, tokenized_sentences: List[List[str]]) -> np.matrix:
        """
            Calculates similarity matrix for sentences using abstract method to calculate similarity between sentences.

            :param tokenized_sentences: List of tokenized sentences
            :return: Similarity matrix as Numpy matrix
        """
        n = len(tokenized_sentences)
        matrix = np.zeros([n, n])
        for i in range(n):
            sentence_i = tokenized_sentences[i]
            for j in range(i + 1, n):
                sentence_j = tokenized_sentences[j]
                similarity = self.calculate_similarity(sentence_i, sentence_j)
                matrix[i, j] = similarity
                matrix[j, i] = similarity

        norm = np.sum(matrix, axis=0)
        matrix = np.divide(matrix, norm, where=norm != 0)

        return matrix

    @abstractmethod
    def calculate_similarity(self, sentence_x: List[str], sentence_y: List[str]) -> float:
        """
            Abstract method that should be implemented in PageRank-based algorithms.
            Should contain function that calculates similarity between tokenized sentences.

            :param sentence_x: Tokenized sentence
            :param sentence_y: Tokenized sentence
            :return: Similarity between sentences
        """
        pass
