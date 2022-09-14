from .FrequencyBasedSummarizer import FrequencyBasedSummarizer
from typing import List, Dict
from util import Utils


class TFIDFSummarizer(FrequencyBasedSummarizer):
    """
        Implementation of TF-IDF-based text summarizer for Polish language.
    """

    def __init__(self, test_idfs: Dict[str, float] = None) -> None:
        super().__init__()
        if test_idfs is not None:
            self.idfs = test_idfs
        else:
            self.idfs = Utils.get_nkjp_idf_values()
        self.default_idf = max(self.idfs.values())

    def summarize(self, text: str, size: int) -> str:
        """
            TF-IDF algorithm implementation.
            Calculates sentence scores based on words TF-IDF values and creates summary from n top ranked sentences.

            :param text: Input text to summarize
            :param size: Number of sentences to include in summary
            :return: Generated summary
        """
        sentences_text = self.get_sentences(text)
        sentences_cleaned = self.clean_sentences(sentences_text)
        scores = self.calculate_scores(sentences_cleaned)
        return self.prepare_summary(sentences_text, Utils.get_ranking(scores, size))

    def calculate_scores(self, sentences: List[List[str]]) -> List[float]:
        """
            Calculates sentence scores as average TF-IDF value of its words.

            :param sentences: List of sentences
            :return: List of scores for sentences in the same order as input
        """
        scores = list()
        tfs = self.calculate_tf_values(sentences)
        for sentence in sentences:
            if len(sentence) == 0:
                scores.append(0)
                break

            sentence_tf_idfs = list()
            for word in sentence:
                sentence_tf_idfs.append(tfs[word] * self.get_idf(word))
            scores.append(sum(sentence_tf_idfs)/len(sentence))
        return scores

    def get_idf(self, word: str) -> float:
        """
            Retrieve IDF value for word calculated beforehand.
            If word was not found in IDF dictionary, returns default IDF which is max IDF from IDF dictionary.
            Assumes that word that did not appear in source corpus 1 time is as common as the rarest word in corpus.

            :param word: Word for which IDF value should be returned
            :return: IDF value for word
        """
        return self.idfs.get(word, self.default_idf)
