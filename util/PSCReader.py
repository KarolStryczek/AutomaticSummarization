from bs4 import BeautifulSoup
import os


def read_all_PSC_files():
    psc_texts = list()
    for filepath in [f.path for f in os.scandir("data/PSC/")]:
        with open(filepath, encoding='utf-8') as f:
            psc_texts.append(PSCText(filepath, f.read()))
    return psc_texts


class PSCText:
    def __init__(self, filename, text_xml):
        bs = BeautifulSoup(text_xml, features='xml')
        self.filename = filename
        self.text = bs.body.text
        self.summaries = list()
        for summary in bs.summaries.find_all('summary'):
            self.summaries.append(PSCSummary(summary))

    def get_summaries(self, ratio=10, summary_type='extract'):
        return [s.text for s in self.summaries if s.ratio == ratio and s.summary_type == summary_type]

    def get_all_summaries(self):
        return self.summaries


class PSCSummary:
    def __init__(self, summary_xml):
        self.ratio = int(summary_xml.attrs['ratio'])
        self.summary_type = summary_xml.attrs['type']
        self.author = summary_xml.attrs['author']
        self.text = summary_xml.body.text
