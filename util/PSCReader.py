from bs4 import BeautifulSoup
from typing import List
import os


class PSCSummary:
    """
        Utility class to store properties and content of reference summary from Polish Summaries Corpus.
    """
    def __init__(self, summary_xml) -> None:
        self.ratio = int(summary_xml.attrs['ratio'])
        self.summary_type = summary_xml.attrs['type']
        self.author = summary_xml.attrs['author']
        self.text = summary_xml.body.text


class PSCText:
    """
        Utility class to store properties, text and summaries of texts from Polish Summaries Corpus.
    """
    def __init__(self, filename, text_xml) -> None:
        bs = BeautifulSoup(text_xml, features='xml')
        self.filename = filename
        self.text = bs.body.text
        self.summaries = list()
        for summary in bs.summaries.find_all('summary'):
            self.summaries.append(PSCSummary(summary))

    def get_all_summaries(self) -> List[PSCSummary]:
        """
            Returns all summaries from given text.

            :return: List of summaries
        """
        return self.summaries


def read_all_PSC_files() -> List[PSCText]:
    """
        Utility function that loads and parses all Polish Summaries Corpus files.

        :return: List of Polish Summaries Corpus texts
    """
    psc_texts = list()
    for filepath in [f.path for f in os.scandir("data/PSC/")]:
        with open(filepath, encoding='utf-8') as f:
            psc_texts.append(PSCText(filepath, f.read()))
    return psc_texts
