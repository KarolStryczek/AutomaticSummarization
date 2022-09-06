from .FrequencyBasedSummarizer import FrequencyBasedSummarizer
from typing import List, Dict


class SumBasicSummarizer(FrequencyBasedSummarizer):
    """
        Allows automatic summarization of documents in Polish using SumBasic extractive summarization.

        SumBasic calculates probability of words' occurrences and uses it to calculate sentence score.
        Best sentence is chosen for summarization, probabilities for words in that sentences are updated and new
        iteration starts from score calculation.
    """

    def __init__(self) -> None:
        super().__init__()

    def summarize(self, text: str, n: int, percent=None) -> str:
        """
            Runs SumBasic algorithm.
            Steps:
                - split text into sentences
                - clean sentences (remove stopwords, lemmatize)
                - calculate word probabilities
                - calculate sentences scores
                - choose best sentence
                - update probabilities for words in chosen sentence
                - go to score calculation if more sentences needed

            :param text: text to summarize
            :param n: number of sentences to be included in summary
            :return: summary of input
        """
        sentences_text = self.get_sentences(text)
        sentences_cleaned = self.clean_sentences(sentences_text)
        summary_indexes = list()
        probabilities = self.calculate_tf_values(sentences_cleaned)

        for i in range(n):
            scores = [self.get_sentence_score(sentence, probabilities) for sentence in sentences_cleaned]
            best_sentence_index = scores.index(max(scores))
            best_sentence = sentences_cleaned[best_sentence_index]
            self.update_probabilities(best_sentence, probabilities)
            summary_indexes.append(best_sentence_index)

        return self.prepare_summary(sentences_text, summary_indexes)

    @staticmethod
    def update_probabilities(sentence: List[str], probabilities: Dict[str, float]) -> None:
        """
            Updates probabilities after sentence has been chosen to summary.
            Probability of words from that sentence is squared (so it is lower).

            :param sentence: sentence chosen to summary
            :param probabilities: probabilities dictionary
        """
        for word in sentence:
            probabilities[word] *= probabilities[word]

    @staticmethod
    def get_sentence_score(sentence: List[str], probabilities: Dict[str, float]) -> float:
        """
            Calculates scores for sentence based on probabilities dictionary.

            :param sentence: sentence to calculate score
            :param probabilities: probabilities dictionary
            :return: score for input sentence
        """
        if len(sentence) == 0:
            return 0

        score = 0
        for word in sentence:
            score += probabilities[word]
        score /= len(sentence)
        return score
