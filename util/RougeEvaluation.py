from rouge import Rouge
from util.PSCReader import PSCText, PSCSummary, read_all_PSC_files
from summarizers.SumBasicSummarizer import SumBasicSummarizer
from summarizers.TFIDFSummarizer import TFIDFSummarizer
from summarizers.TextRankSummarizer import TextRankSummarizer
from summarizers.LexRankSummarizer import LexRankSummarizer
import nltk
import pandas as pd
from datetime import datetime


class RougeEvaluator:
    def __init__(self):
        self.summarizers = [SumBasicSummarizer(), TFIDFSummarizer(), TextRankSummarizer(), LexRankSummarizer()]
        self.rouge_evaluator = Rouge()
        self.results = self.load_results()

    @staticmethod
    def load_results():
        return pd.read_csv('util/results.csv')

    def save_results(self):
        self.results.to_csv('util/results.csv', index=False)

    def evaluate_all(self):
        for file in read_all_PSC_files():
            print(datetime.now().strftime("%H:%M:%S"), file.filename)
            if file.filename not in self.results['filename'].unique():
                scores = self.evaluate(file)
                for score in scores:
                    score_df = pd.DataFrame([score.get_csv_record()], columns=self.results.columns)
                    self.results = pd.concat([self.results, score_df])
            self.save_results()

    def evaluate(self, psc_text: PSCText):
        scores = list()
        ref_summaries = psc_text.get_all_summaries()
        for ref_summary in ref_summaries:
            n = len(nltk.sent_tokenize(ref_summary.text))
            for summarizer in self.summarizers:
                auto_summary = summarizer.summarize(psc_text.text, n)
                score = self.rouge_evaluator.get_scores(auto_summary, ref_summary.text)[0]
                scores.append(RougeScore(psc_text.filename, ref_summary, summarizer, score))
        return scores


class RougeScore:
    __rouge1 = 'rouge-1'
    __rouge2 = 'rouge-2'
    __rougeL = 'rouge-l'
    __recall = 'r'
    __precision = 'p'
    __f_score = 'f'

    def __init__(self, filename, psc_summary: PSCSummary, summarizer, rouge_score):
        self.filename = filename
        self.summary_type = psc_summary.summary_type
        self.author = psc_summary.author
        self.ratio = psc_summary.ratio
        self.summarizer = summarizer.__class__.__name__
        self.score = rouge_score

    def get_csv_record(self):
        return (self.filename, self.summary_type, self.ratio, self.author, self.summarizer) + \
               self.__get_rouge_score_csv(self.score[self.__rouge1]) + \
               self.__get_rouge_score_csv(self.score[self.__rouge2]) + \
               self.__get_rouge_score_csv(self.score[self.__rougeL])

    def __get_rouge_score_csv(self, rouge_score):
        return rouge_score[self.__recall], rouge_score[self.__precision], rouge_score[self.__f_score]

    def __str__(self):
        return str(self.get_csv_record())
