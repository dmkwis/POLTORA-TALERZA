from abc import ABC, abstractmethod
from typing import List

class AbstractMetric(ABC):
    def __init__(self, metric_name):
        self.metric_name = metric_name

    @staticmethod
    @abstractmethod
    def compute(prompt_text: List[str], generated_text: List[List[str]]) -> float:
        pass

    def get_name(self):
        return self.metric_name