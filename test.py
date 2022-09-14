from summarizers.SumBasicSummarizer import SumBasicSummarizer
from summarizers.TextRankSummarizer import TextRankSummarizer
from summarizers.LexRankSummarizer import LexRankSummarizer
from summarizers.TFIDFSummarizer import TFIDFSummarizer
from summarizers.AbstractSummarizer import AbstractSummarizer
from util import Utils
import math
import numpy as np


class TestUtil:
    def test_calculate_idf_values(self):
        text1 = "Oko ucho nos"
        text2 = "Oko ucho"
        text3 = "Oko"
        res = Utils.calculate_idf_values([text1, text2, text3])
        assert len(res) == 3
        assert res['oko'] == 0
        assert res['ucho'] == math.log10(3/2)
        assert res['nos'] == math.log10(3)

    def test_read_and_clean_nkjp_xml_file(self):
        file_text = """
        <?xml version="1.0" encoding="UTF-8"?>
        <teiCorpus xmlns:xi="http://www.w3.org/2001/XInclude" xmlns="http://www.tei-c.org/ns/1.0">
         <xi:include href="header.xml"/>
         <TEI>
          <xi:include href="header.xml"/>
          <text xml:id="txt_text" xml:lang="pl">
           <body xml:id="txt_body">
            <div xml:id="NONE" decls="NONE">
             <ab n="NONE" xml:id="NONE">
                SOME TEI TEXT
             </ab>
            </div>
    
           </body>
          </text>
         </TEI>
        </teiCorpus>
        """
        assert Utils.read_and_clean_nkjp_xml_file(file_text) == "SOME TEI TEXT"

    def test_get_ranking(self):
        scores = [1, 3, 2, 6, 5, 4]
        assert Utils.get_ranking(scores, 3) == [3, 4, 5]

    def test_sort_dict_by_value(self):
        input_dict = {1: 2, 2: 1, 3: 3}
        assert Utils.sort_dict_by_value(input_dict) == {2: 1, 1: 2, 3: 3}

    def test_calculate_tf_values(self):
        sentence = ['oko', 'nos', 'oko', 'ucho', 'ucho']
        res = Utils.calculate_tf_values(sentence)
        assert len(res) == 3
        assert res['oko'] == 2
        assert res['ucho'] == 2
        assert res['nos'] == 1


class TestSumBasic:
    def test_prepare_summary(self):
        summarizer = SumBasicSummarizer()
        sentences = ["sentence1.", "sentence2.", "sentence3."]
        assert summarizer.prepare_summary(sentences, [1, 0]) == "sentence1.\nsentence2."

    def test_get_sentences(self):
        summarizer = SumBasicSummarizer()
        text = "Ala ma kota. Kot ma Alę."
        assert summarizer.get_sentences(text) == ["Ala ma kota.", "Kot ma Alę."]

    def test_calculate_tf_values(self):
        summarizer = SumBasicSummarizer()
        sentence1 = ["Ala", "ma", "kota"]
        sentence2 = ["Ala", "ma", "psa"]
        res = summarizer.calculate_tf_values([sentence1, sentence2])
        assert len(res) == 4
        assert res["Ala"] == 2/6
        assert res["ma"] == 2/6
        assert res["kota"] == 1/6
        assert res["psa"] == 1/6

    def test_calculate_score(self):
        summarizer = SumBasicSummarizer()
        sentence = ["Ala", "ma", "kota"]
        probabilities = {"Ala": 1, "ma": 2, "kota": 5}
        assert summarizer.calculate_score(sentence, probabilities) == 8/3

    def test_update_probabilities(self):
        summarizer = SumBasicSummarizer()
        sentence = ["Ala", "ma"]
        probabilities = {"Ala": 0.5, "ma": 0.5, "kota": 0.5}
        summarizer.update_probabilities(sentence, probabilities)
        assert probabilities["Ala"] == 0.25
        assert probabilities["ma"] == 0.25
        assert probabilities["kota"] == 0.5

    def test_summarize(self):
        summarizer = SumBasicSummarizer()
        text = "kot pies oko nos. kot pies oko. kot pies. kot."
        assert summarizer.summarize(text, 1) == "kot."
        assert summarizer.summarize(text, 3) == "kot pies oko.\nkot pies.\nkot."


class TestTFIDF:
    def test_prepare_summary(self):
        summarizer = TFIDFSummarizer({"Ala": 0, "ma": 0.5, "kota": 5})
        sentences = ["sentence1.", "sentence2.", "sentence3."]
        assert summarizer.prepare_summary(sentences, [1, 0]) == "sentence1.\nsentence2."

    def test_get_sentences(self):
        summarizer = TFIDFSummarizer({"Ala": 0, "ma": 0.5, "kota": 5})
        text = "Ala ma kota. Kot ma Alę."
        assert summarizer.get_sentences(text) == ["Ala ma kota.", "Kot ma Alę."]

    def test_calculate_tf_values(self):
        summarizer = TFIDFSummarizer({"Ala": 0, "ma": 0.5, "kota": 5})
        sentence1 = ["Ala", "ma", "kota"]
        sentence2 = ["Ala", "ma", "psa"]
        res = summarizer.calculate_tf_values([sentence1, sentence2])
        assert len(res) == 4
        assert res["Ala"] == 2/6
        assert res["ma"] == 2/6
        assert res["kota"] == 1/6
        assert res["psa"] == 1/6

    def test_calculate_score(self):
        summarizer = TFIDFSummarizer({"Ala": 0, "ma": 0.5, "kota": 5})
        sentence1 = ["Ala", "ma", "kota"]
        sentence2 = ["Ala", "ma", "psa"]
        sentence3 = ["Ala", "ma"]
        res = summarizer.calculate_scores([sentence1, sentence2, sentence3])
        assert len(res) == 3
        assert res[0] == 1/3 * (0.5 * 3/8 + 5 * 1/8)
        assert res[1] == 1/3 * (0.5 * 3/8 + 5 * 1/8)
        assert res[2] == 1/2 * (0.5 * 3/8)

    def test_summarize(self):
        summarizer = TFIDFSummarizer({"kot": 0, "pies": 3, "oko": 5, "nos": 100})
        text = "kot pies oko nos. kot pies oko. kot pies. kot."
        assert summarizer.summarize(text, 1) == "kot pies oko nos."
        assert summarizer.summarize(text, 3) == "kot pies oko nos.\nkot pies oko.\nkot pies."


class TestTextRank:
    def test_prepare_summary(self):
        summarizer = TextRankSummarizer()
        sentences = ["sentence1.", "sentence2.", "sentence3."]
        assert summarizer.prepare_summary(sentences, [1, 0]) == "sentence1.\nsentence2."

    def test_get_sentences(self):
        summarizer = TextRankSummarizer()
        text = "Ala ma kota. Kot ma Alę."
        assert summarizer.get_sentences(text) == ["Ala ma kota.", "Kot ma Alę."]

    def test_calculate_similarity(self):
        summarizer = TextRankSummarizer()
        sentence1 = ["oko", "nos", "ucho"]
        sentence2 = ["oko", "nos"]
        sentence3 = ["nos", "ucho"]
        assert summarizer.calculate_similarity(sentence1, sentence2) == 2 / math.log10(3*2)
        assert summarizer.calculate_similarity(sentence1, sentence3) == 2 / math.log10(3*2)
        assert summarizer.calculate_similarity(sentence2, sentence3) == 1 / math.log10(2*2)

    def test_create_similarity_matrix(self):
        summarizer = TextRankSummarizer()
        sentence1 = ["oko", "nos", "ucho"]
        sentence2 = ["oko", "nos"]
        sentence3 = ["nos", "ucho"]
        res = summarizer.create_similarity_matrix([sentence1, sentence2, sentence3])
        col_sums = np.sum(res, axis=0)
        assert res[0][0] == res[1][1] == res[2][2] == 0
        assert col_sums[0] == col_sums[1] == col_sums[2] == 1

    def test_summarize(self):
        summarizer = TextRankSummarizer()
        text = "kot pies oko nos. kot pies oko. kot pies. kot."
        assert summarizer.summarize(text, 1) == "kot pies."
        assert summarizer.summarize(text, 3) == "kot pies oko.\nkot pies.\nkot."


class TestLexRank:
    def test_prepare_summary(self):
        summarizer = LexRankSummarizer({"Ala": 0, "ma": 0.5, "kota": 5})
        sentences = ["sentence1.", "sentence2.", "sentence3."]
        assert summarizer.prepare_summary(sentences, [1, 0]) == "sentence1.\nsentence2."

    def test_get_sentences(self):
        summarizer = LexRankSummarizer({"Ala": 0, "ma": 0.5, "kota": 5})
        text = "Ala ma kota. Kot ma Alę."
        assert summarizer.get_sentences(text) == ["Ala ma kota.", "Kot ma Alę."]

    def test_create_similarity_matrix(self):
        summarizer = LexRankSummarizer({"oko": 0, "nos": 0.5, "ucho": 5})
        sentence1 = ["oko", "nos", "ucho"]
        sentence2 = ["oko", "nos"]
        sentence3 = ["nos", "ucho"]
        res = summarizer.create_similarity_matrix([sentence1, sentence2, sentence3])
        col_sums = np.sum(res, axis=0)
        assert res[0][0] == res[1][1] == res[2][2] == 0
        assert col_sums[0] == col_sums[1] == col_sums[2] == 1

    def test_summarize(self):
        summarizer = LexRankSummarizer({"kot": 0, "pies": 0.5, "oko": 5})
        text = "kot pies oko nos. kot pies oko. kot pies. kot."
        assert summarizer.summarize(text, 1) == "kot pies oko nos."
        assert summarizer.summarize(text, 3) == "kot pies oko nos.\nkot pies oko.\nkot pies."
