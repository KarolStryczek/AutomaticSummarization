from .FrequencyBasedSummarizer import FrequencyBasedSummarizer
from util import Utils


class TFIDFSummarizer(FrequencyBasedSummarizer):
    def __init__(self):
        super().__init__()
        self.idfs = Utils.get_nkjp_idf_values()
        self.default_idf = max(self.idfs.values())

    def summarize(self, text, n, percent=None):
        sentences_text = self.get_sentences(text)
        sentences_cleaned = self.clean_sentences(sentences_text)
        scores = self.calculate_sentences_scores(sentences_cleaned)
        return self.prepare_summary(sentences_text, Utils.get_ranking(scores, n))

    def calculate_sentences_scores(self, sentences):
        scores = list()
        tfs = self.calculate_tf_values(sentences)
        for sentence in sentences:
            if len(sentence) == 0:
                scores.append(0)
                break

            sentence_tf_idfs = list()
            for word in sentence:
                sentence_tf_idfs.append(tfs[word] * self.__get_word_idf(word))
            scores.append(sum(sentence_tf_idfs)/len(sentence))
        return scores

    def __get_word_idf(self, word):
        try:
            return self.idfs[word]
        except KeyError:
            return self.default_idf
