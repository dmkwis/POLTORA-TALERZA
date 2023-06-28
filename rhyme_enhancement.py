import torch
import random
from copy import deepcopy
from typing import List, Tuple
from metrics.rhymedensitymetric import RhymeDensityMetric


class RhymeData:
    def __init__(self, words: List[List[str]], idxs: Tuple[int, int], tokenizer):
        self.words = deepcopy(words)
        n = len(self.words)
        self.idxs = idxs
        self.tokenizer = tokenizer

        self.src = self.words[self.idxs[0]][-1]
        self.tgt = self.words[self.idxs[1]][-1]

        # Mask first word
        self.words[self.idxs[0]][-1] = tokenizer.mask_token

        # Restore lines and sentence
        self.lines = [' '.join(self.words[i]) for i in range(n)]
        self.sentence = tokenizer.sep_token.join(self.lines)

        self.indexed_tokens = tokenizer.encode(self.sentence, add_special_tokens=True)

        self.prepare_tensors()

    def prepare_tensors(self):
        segments_ids = []
        mask_idxs = []
        separators_found = 0
        for i, token in enumerate(self.indexed_tokens):
            # Only 0 or 1
            segments_ids.append(separators_found // 2)
            if token == self.tokenizer.sep_token_id:
                separators_found += 1
            elif token == self.tokenizer.mask_token_id:
                mask_idxs.append(i)

        self.mask_idxs = mask_idxs
        self.segments_tensors = torch.tensor(segments_ids)


class RhymeEnhancer:
    def __init__(self):
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'modelForMaskedLM', 'bert-base-cased')

    def find_match(self, indexed_tokens: List[int], K: int, mask_idx: int, src_word: str, tgt_word: str,
                   segments_tensors: torch.Tensor) -> Tuple[str, float]:
        
        def rhyme_length(a: str, b: str) -> float:
            return RhymeDensityMetric.compute_rhyme_length(a, b)

        rl_original = rhyme_length(src_word, tgt_word)

        tokens_tensor = torch.tensor([indexed_tokens])
        predictions = self.bert(tokens_tensor, token_type_ids=segments_tensors).logits

        topk = torch.topk(predictions[0][mask_idx], K, dim=-1).indices

        predictions = self.tokenizer.convert_ids_to_tokens(topk)

        for pred in predictions:
            rl_new = rhyme_length(pred, tgt_word)
            if rl_new > rl_original:
                return pred, rl_new

        return src_word, rl_original

    @torch.no_grad()
    def enhance(self, sentence: str, K: int = 200, idxs: Tuple[int, int] = (0, 2)) -> str:
        # Enhances sentence which is 4 lines separated by \n by rhymes
        # Tests K best bert suggestion and chooses first one with higher rhyme metric than original
        # idxs represents which lines should rhyme

        lines = sentence.split('\n')
        n = len(lines)
        assert n > 2
        words = [line.split() for line in lines]

        rhyme_data_first = RhymeData(words, idxs, self.tokenizer)
        rhyme_data_second = RhymeData(words, tuple(reversed(idxs)), self.tokenizer)

        rhyme_data_first.prepare_tensors()
        rhyme_data_second.prepare_tensors()

        first_word, rl_first = self.find_match(
            rhyme_data_first.indexed_tokens,
            K,
            rhyme_data_first.mask_idxs[0],
            rhyme_data_first.src,
            rhyme_data_first.tgt,
            rhyme_data_first.segments_tensors,
        )

        second_word, rl_second = self.find_match(
            rhyme_data_second.indexed_tokens,
            K,
            rhyme_data_second.mask_idxs[0],
            rhyme_data_second.src,
            rhyme_data_second.tgt,
            rhyme_data_second.segments_tensors,
        )

        if rl_first > rl_second:
            words[idxs[0]][-1] = first_word
        else:
            words[idxs[1]][-1] = second_word

        lines = [' '.join(words[i]) for i in range(n)]
        sentence = '\n'.join(lines)
        return sentence

class ExampleParser:
    def __init__(self, file):
        self.f = file
        self.buffer = ""

    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            line = self.f.readline()
            if line == "":
                raise StopIteration
            elif line == "\n":
                to_return = self.buffer
                self.buffer = ""
                return to_return[:-1]
            else:
                self.buffer += line


if __name__ == '__main__':
    enhancer = RhymeEnhancer()
    with open("results.txt", 'r') as src:
        with open("enchanced.txt", 'w') as dst:
            example_parser = ExampleParser(src)
            for example in example_parser:
                print('Before:')
                print(example, "\n")

                example = enhancer.enhance(example, 50)
                print('After:')
                print(example, "\n")
