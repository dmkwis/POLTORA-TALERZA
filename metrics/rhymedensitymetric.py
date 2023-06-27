from abstractmetric import AbstractMetric
from typing import List, Optional
from itertools import chain
import os
import numpy as np
from functools import reduce

class RhymeDensityMetric(AbstractMetric):
    def __init__(self):
        super().__init__("Rhyme density metric")
    
    @staticmethod
    def generate_ipa(text: str, f_in: str, f_out: str) -> Optional[str]:
        with open(f_in, 'w') as source:    
            source.write(text)
        os.system(f'espeak -xq -f {f_in} > {f_out}')
        with open(f_out, 'r') as result:
            return result.read()[1:-1] #removing extra starting ' ' and extra ending '\n'
    
    @staticmethod
    def map_vowel(c):
        vowel_map = {
            '0':'o',
            'O':'o',
            'I':'i',
            'E':'e'
        }
        if c in vowel_map:
            return vowel_map[c]
        return c
    
    @staticmethod
    def is_vowel(c):
        return c in '3L5aAeEiI0VuUoO'

    @staticmethod
    def process_ipa(ipa: str):
        return "".join(filter(RhymeDensityMetric.is_vowel, filter(RhymeDensityMetric.map_vowel, ipa)))

    """Searches for max common contiguous substring of w1 in w2, where len(w1) << len(w2)"""
    @staticmethod
    def calculate_ipa_rhyme_length(w1: str, w2: str):
        n = len(w1)
        subwords = [w1[i:j + 1] for i in range(n) for j in range(i, n)]
        find_subwords = map(lambda x: (len(x), x in w2), subwords)
        found_subwords = filter(lambda x: x[1], find_subwords)
        found_lengths = list(map(lambda x: x[0], found_subwords))
        if len(found_lengths) > 0:
            return reduce(max, found_lengths)
        return 0

    @staticmethod
    def compute_ipa_representation(word: str,
            tmp_in_file="tmp_in.txt",
            tmp_out_file="tmp_out.txt"):
        ipa = RhymeDensityMetric.generate_ipa(word, tmp_in_file, tmp_out_file)
        os.system(f'rm {tmp_in_file} {tmp_out_file}')
        return RhymeDensityMetric.process_ipa(ipa)
    

    @staticmethod
    def compute_rhyme_length(w1: str, w2: str):
        tmp_in_file="tmp_in.txt",
        tmp_out_file="tmp_out.txt"
        ipa_w1 = RhymeDensityMetric.generate_ipa(w1, tmp_in_file, tmp_out_file)
        ipa_w2 = RhymeDensityMetric.generate_ipa(w2, tmp_in_file, tmp_out_file)
        os.system(f'rm {tmp_in_file} {tmp_out_file}')
        return RhymeDensityMetric.calculate_ipa_rhyme_length(ipa_w1, ipa_w2)

    

    def compute(prompt_text: List[str], generated_text: List[str]) -> float:
        assert prompt_text is None, "Metric used only on generated text"
        processed_ipas = []
        for word in generated_text:
            ipa = RhymeDensityMetric.compute_ipa_representation(word)
            processed_ipas.append(ipa)
        #print(processed_ipas)
        rhyme_lens = []
        for idx, word in enumerate(processed_ipas):
            left_words = "".join(processed_ipas[:idx])
            right_words = "".join(processed_ipas[idx+1:])
            rhyme_len = max(RhymeDensityMetric.calculate_ipa_rhyme_length(word, left_words),
                            RhymeDensityMetric.calculate_ipa_rhyme_length(word, right_words))
            rhyme_lens.append(rhyme_len)

        #print(rhyme_lens)

        return np.average(rhyme_lens)

if __name__ == '__main__':
    ran= '''i aint supposed to be
    always gone with these hoes
    and ever be the first
    i be fucked up with you'''
    print(ran.split())
    print(RhymeDensityMetric.compute(None, ran.split()))



        