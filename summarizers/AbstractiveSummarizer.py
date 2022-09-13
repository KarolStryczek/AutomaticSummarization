from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from .AbstractSummarizer import AbstractSummarizer
import nltk
import os


class AbstractiveSummarizer(AbstractSummarizer):
    """
        Implementation of abstractive summarization for Polish language.
        It is based on 3 steps:
            - Polish text translation into English using MBart-50 Many-To-Many model
            - Summarization using Google PEGASUS (CNN/DailyMail) model of English text
            - English summary translation to Polish
    """

    def __init__(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        self.__pl = "pl_PL"
        self.__en = "en_XX"
        self.__device = "cuda"

        self.translator_tokenizer = MBart50TokenizerFast.from_pretrained('facebook/mbart-large-50-many-to-many-mmt')
        self.translator_model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-50-many-to-many-mmt').to(self.__device)

        self.pegasus_tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-cnn_dailymail')
        self.pegasus_model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-cnn_dailymail').to(self.__device)

        self.translation_cache = dict()

    def summarize(self, text: str, size: int) -> str:
        """
            Summarization process uses machine translation models to translate Polish text to English
            in batches consisting of 15 sentences at most.
            PEGASUS model is then applied to generate abstractive summary that is later on translated again to Polish.
            Maximal size of generated summary is set to 200 tokens.

            :param text: Text to summarize
            :param size: Number of tokens to generate in summary (approximate)
            :return: Generated abstractive summary
        """
        words_count = len([token for token in nltk.word_tokenize(text) if token.isalpha()])
        summary_length = min(200, words_count*size//100)

        if self.translation_cache.get(hash(text)) is None:
            sentences = nltk.sent_tokenize(text)
            batches = [" ".join(sentences[i:i+15]) for i in range(0, len(sentences), 15)]
            text_en = " ".join([self.translate(batch, self.__pl, self.__en)[0] for batch in batches])
            self.translation_cache.clear()
            self.translation_cache[hash(text)] = text_en
        else:
            text_en = self.translation_cache.get(hash(text))

        summary_en = self.pegasus_summarize(text_en, summary_length)
        summary_pl = self.translate(summary_en, self.__en, self.__pl)[0]
        return summary_pl.replace("<n>", "\n")

    def pegasus_summarize(self, text: str, size: int) -> str:
        """
            Abstractive summarization using PEGASUS model.

            :param text: Text to summarize
            :param size: Number of tokens to generate in summary (approximate)
            :return: Generated abstractive summary
        """
        batch = self.pegasus_tokenizer(text, truncation=True, padding="longest", return_tensors="pt").to(self.__device)
        res = self.pegasus_model.generate(**batch, min_length=size-10, max_length=size+10)
        return self.pegasus_tokenizer.batch_decode(res, skip_special_tokens=True)

    def translate(self, text: str, source_language: str, target_language: str) -> str:
        """
            Machine translation using MBArt-50 Many-To-Many model.

            :param text: Text to translate
            :param source_language: Translation source language
            :param target_language: Translation target language
            :return: Translated text
        """
        self.translator_tokenizer.src_lang = source_language
        encoded_input = self.translator_tokenizer(text, return_tensors="pt").to(self.__device)
        generated_tokens = self.translator_model.generate(**encoded_input, forced_bos_token_id=self.translator_tokenizer.lang_code_to_id[target_language])
        return self.translator_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
