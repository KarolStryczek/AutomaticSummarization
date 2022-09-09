from abc import ABC, abstractmethod


class AbstractSummarizer(ABC):
    """
        Abstract super class for all summarizers.

        One method summarize() must be implemented by subclasses
    """

    @abstractmethod
    def summarize(self, text: str, size: int) -> str:
        """
            Method in subclasses should contain summarization algorithm implementation

            :param text: Text to summarize
            :param size: Size of output summary. Algorithms can use their own units to measure size (e.g. sentence/word)
            :return: Generated summary
        """
        pass
