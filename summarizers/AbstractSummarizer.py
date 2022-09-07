from abc import ABC, abstractmethod


class AbstractSummarizer(ABC):
    """
        Super class for all summarizers.

        One method summarize() must be implemented by subclasses
    """

    @abstractmethod
    def summarize(self, text, size):
        """
            Method in subclasses should contain summarization algorithm implementation
        """
        pass
