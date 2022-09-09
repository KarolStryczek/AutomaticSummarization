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

    def summarize(self, text: str, size: int) -> str:
        """
            Runs SumBasic algorithm which calculates sentence scores based on words probability calculated behorehand.
            Word probabilities for word from each chosen sentences are updated in each iteration
            to ensure that same or similar sentences are not included in summary.

            :param text: Text to summarize
            :param size: Number of sentences to extract
            :return: Generated summary
        """
        sentences_text = self.get_sentences(text)
        sentences_cleaned = self.clean_sentences(sentences_text)
        summary_indexes = list()
        probabilities = self.calculate_tf_values(sentences_cleaned)

        for i in range(size):
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

            :param sentence: Sentence chosen to summary
            :param probabilities: Updated probabilities dictionary
        """
        for word in sentence:
            probabilities[word] *= probabilities[word]

    @staticmethod
    def get_sentence_score(sentence: List[str], probabilities: Dict[str, float]) -> float:
        """
            Calculates scores for sentence based on probabilities dictionary.

            :param sentence: Sentence to calculate score for
            :param probabilities: Probabilities dictionary
            :return: Score for input sentence
        """
        if len(sentence) == 0:
            return 0

        score = 0
        for word in sentence:
            score += probabilities[word]
        score /= len(sentence)
        return score
