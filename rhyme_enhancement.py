import torch
import random
from copy import deepcopy
from typing import List, Tuple


class RhymeData:
    def __init__(self, words: List[List[str]], idxs: Tuple[int, int], tokenizer):
        self.words = deepcopy(words)
        self.idxs = idxs
        self.tokenizer = tokenizer

        self.src = self.words[self.idxs[0]][-1]
        self.tgt = self.words[self.idxs[1]][-1]

        # Mask first word
        self.words[self.idxs[0]][-1] = tokenizer.mask_token

        # Restore lines and sentence
        self.lines = [' '.join(self.words[i]) for i in range(4)]
        self.sentence = tokenizer.sep_token.join(self.lines)

        self.indexed_tokens = tokenizer.encode(self.sentence, add_special_tokens=True)

    def prepare_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        segments_ids = []
        mask_idxs = []
        separators_found = 0
        for i, token in enumerate(self.indexed_tokens):
            segments_ids.append(separators_found)
            if token == self.tokenizer.sep_token_id:
                separators_found += 1
            elif token == self.tokenizer.mask_token_id:
                mask_idxs.append(i)

        return mask_idxs, torch.tensor(segments_ids)


class RhymeEnhancer:
    def __init__(self):
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'modelForMaskedLM', 'bert-base-cased')

    def find_match(self, indexed_tokens: List[int], K: int, mask_idx: int, src_word: str, tgt_word: str,
                   segments_tensors: torch.Tensor) -> Tuple[str, float]:
        # TODO replace with actual rhyme metric
        def rhyme_length(a: str, b: str) -> float:
            print(f'Calculating rhyme length of word \'{a}\' and \'{b}\'')
            return random.random()

        # src_word = self.tokenizer.convert_ids_to_tokens(indexed_tokens[mask_idx])
        rl_original = rhyme_length(src_word, tgt_word)

        tokens_tensor = torch.tensor([indexed_tokens])
        print(tokens_tensor)
        # predictions = self.bert(tokens_tensor, token_type_ids=segments_tensors).logits
        predictions = self.bert(tokens_tensor).logits

        topk = torch.topk(predictions[0][mask_idx], K, dim=-1).indices
        print(topk)

        predictions = self.tokenizer.convert_ids_to_tokens(topk)

        for pred in predictions:
            rl_new = rhyme_length(pred, tgt_word)
            # if rl_new > rl_original:
            #     return pred, rl_new

        return src_word, rl_original

    @torch.no_grad()
    def enhance(self, sentence: str, K: int = 200, rhyme: str = 'abab') -> str:
        assert rhyme in ('abab', 'abba')
        lines = sentence.split('\n')
        assert len(lines) == 4
        words = [line.split() for line in lines]

        rhyme_data_first = RhymeData(words, (0, 2), self.tokenizer)
        # rhyme_data_second = RhymeData(words, (2, 0), self.tokenizer)

        indexed_tokens = self.tokenizer.encode(rhyme_data_first.sentence, add_special_tokens=True)
        print(indexed_tokens)
        # debug print
        print()
        for token in indexed_tokens:
            print(token, self.tokenizer.convert_ids_to_tokens(token))
        print()

        mask_idxs, segments_tensors = rhyme_data_first.prepare_tensors()

        first_word, rl_first = self.find_match(indexed_tokens, K, mask_idxs[0], rhyme_data_first.src,
                                               rhyme_data_first.tgt, segments_tensors)
        # self.find_match(indexed_tokens, K, separator_idxs[1] - 1, separator_idxs[0] - 1)

        return first_word


if __name__ == '__main__':
    enhancer = RhymeEnhancer()

    example = '''i aint supposed to be
always gone with these hoes
and ever be the first
i be fucked up with you'''
    print('Before:')
    print(example)

    example = enhancer.enhance(example, 50)
    print('After:')
    print(example)
