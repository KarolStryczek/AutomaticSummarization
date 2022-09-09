from summarizers.AbstractiveSummarizer import AbstractiveSummarizer
from util.PSCReader import PSCText, PSCSummary, read_all_PSC_files
from summarizers.SumBasicSummarizer import SumBasicSummarizer
from summarizers.TextRankSummarizer import TextRankSummarizer
from summarizers.LexRankSummarizer import LexRankSummarizer
from summarizers.TFIDFSummarizer import TFIDFSummarizer
from summarizers.AbstractSummarizer import AbstractSummarizer
from typing import List, Dict, Tuple
from datetime import datetime
from rouge import Rouge
import pandas as pd
import nltk


class RougeScore:
    """
        Utility class that stores ROUGE scores in evaluation process and contains methods that generates CSV records.
    """

    __rouge1 = 'rouge-1'
    __rouge2 = 'rouge-2'
    __rougeL = 'rouge-l'
    __recall = 'r'
    __precision = 'p'
    __f_score = 'f'

    def __init__(self, filename: str, psc_summary: PSCSummary, summarizer: AbstractSummarizer, rouge_score: Dict[str, Dict[str, float]]) -> None:
        self.filename = filename
        self.summary_type = psc_summary.summary_type
        self.author = psc_summary.author
        self.ratio = psc_summary.ratio
        self.summarizer = summarizer.__class__.__name__
        self.score = rouge_score

    def get_csv_record(self) -> Tuple[str, str, int, str, str, float, float, float, float, float, float, float, float, float]:
        """
            Creates CSV record with columns:
                - filename
                - summary type
                - summary length to text length ratio
                - author
                - used summarizer
                - 9 columns of ROUGE-1, ROUGE-2, ROUGE-L - recall, precision and F-score

            :return: Tuple with all necessary columns
        """
        return (self.filename, self.summary_type, self.ratio, self.author, self.summarizer) + \
               self.get_rouge_score_csv(self.score[self.__rouge1]) + \
               self.get_rouge_score_csv(self.score[self.__rouge2]) + \
               self.get_rouge_score_csv(self.score[self.__rougeL])

    def get_rouge_score_csv(self, rouge_score: Dict[str, float]) -> Tuple[float, float, float]:
        """
            Returns tuple of recall, precision and F-score for given ROUGE score.

            :param rouge_score: Dictionary with ROUGE scores
            :return: Tuple of ROUGE scores
        """
        return rouge_score[self.__recall], rouge_score[self.__precision], rouge_score[self.__f_score]

    def __str__(self):
        return str(self.get_csv_record())


class RougeEvaluator:
    """
        Class that enables ROUGE evaluation of text summarization algorithms on Polish Summaries Corpus.
        Contains necessary methods to load Polish Summaries Corpus, evaluate implemented algorithms and save results.
    """

    def __init__(self) -> None:
        self.extractive_summarizers = [SumBasicSummarizer(), TFIDFSummarizer(), TextRankSummarizer(), LexRankSummarizer()]
        self.abstractive_summarizer = AbstractiveSummarizer()
        self.rouge_evaluator = Rouge()
        self.results = self.load_results()

    @staticmethod
    def load_results() -> pd.DataFrame:
        """
            Loads previously saved CSV file with evaluation records (or empty file with no records, just columns).

            :return: Pandas dataframe with evaluation records
        """
        return pd.read_csv('util/results.csv')

    def save_results(self) -> None:
        """
            Saves calculated evalation results to result CSV file.

            :return: None
        """
        self.results.to_csv('util/results.csv', index=False)

    def evaluate_all(self) -> None:
        """
            Calculates recall, precision and F-Score of ROUGE-1, ROUGE-2 and ROUGE-L for all reference summaries
            from Polish Summaries Corpus using all implemented summarizers.
            Calculated scores are saved after each text processed (each text have multiple reference summaries).

            :return: None
        """
        for file in read_all_PSC_files():
            print(datetime.now().strftime("%H:%M:%S"), file.filename)
            if file.filename not in self.results['filename'].unique():
                scores = self.evaluate_extractive(file)
                scores += self.evaluate_abstractive(file)
                for score in scores:
                    score_df = pd.DataFrame([score.get_csv_record()], columns=self.results.columns)
                    self.results = pd.concat([self.results, score_df])
            self.save_results()

    def evaluate_extractive(self, psc_text: PSCText) -> List[RougeScore]:
        """
            Runs evaluation for all extractive summarizers.
            For each reference summary generates extractive summaries,
            consisting of the same number of sentences as reference summary.

            :param psc_text: Text from Polish Summaries Corpus with reference summaries
            :return: Scores for all reference summaries calculated with all summarizers
        """
        scores = list()
        ref_summaries = psc_text.get_all_summaries()
        for ref_summary in ref_summaries:
            n = len(nltk.sent_tokenize(ref_summary.text))
            for summarizer in self.extractive_summarizers:
                auto_summary = summarizer.summarize(psc_text.text, n)
                score = self.rouge_evaluator.get_scores(auto_summary, ref_summary.text)[0]
                scores.append(RougeScore(psc_text.filename, ref_summary, summarizer, score))
        return scores

    def evaluate_abstractive(self, psc_text: PSCText) -> List[RougeScore]:
        """
            Runs evaluation for an abstractive summarizer.
            For each reference summary generates extractive summaries,
            consisting of the same number of sentences as reference summary.

            :param psc_text: Text from Polish Summaries Corpus with reference summaries
            :return: Scores for all reference summaries calculated with abstractive summarizer
        """
        scores = list()
        ref_summaries = psc_text.get_all_summaries()
        for ratio in [5, 10, 20]:
            auto_summary = self.abstractive_summarizer.summarize(psc_text.text, ratio)
            for ref_summary in ref_summaries:
                if ref_summary.ratio == ratio:
                    score = self.rouge_evaluator.get_scores(auto_summary, ref_summary.text)[0]
                    scores.append(RougeScore(psc_text.filename, ref_summary, self.abstractive_summarizer, score))
        return scores
