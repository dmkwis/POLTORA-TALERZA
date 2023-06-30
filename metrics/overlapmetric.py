from .abstractmetric import AbstractMetric
from typing import List
from itertools import chain

class OverlapMetric(AbstractMetric):
    def __init__(self):
        super().__init__("Overlap metric")

    def compute(prompt_text: List[str], generated_text: List[List[str]]) -> float:
        assert len(prompt_text) != 0
        prompt_set = set(prompt_text)
        generated_set = set(chain(*generated_text))
        intersection_set = prompt_set.intersection(generated_set)
        return len(intersection_set) / len(prompt_set)
    
if __name__ == '__main__':
    assert OverlapMetric.compute(["p1"], [["p1", "p2"], ["p3"]]) == 1
    assert OverlapMetric.compute(["p1", "b1"], [["p1", "p2"], ["p3"]]) == 0.5

        