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

        self.prepare_tensors()

    def prepare_tensors(self):
        segments_ids = []
        mask_idxs = []
        separators_found = 0
        for i, token in enumerate(self.indexed_tokens):
            segments_ids.append(separators_found)
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
        # TODO replace with actual rhyme metric
        def rhyme_length(a: str, b: str) -> float:
            print(f'Calculating rhyme length of word \'{a}\' and \'{b}\'')
            return random.random()

        rl_original = rhyme_length(src_word, tgt_word)

        tokens_tensor = torch.tensor([indexed_tokens])
        # predictions = self.bert(tokens_tensor, token_type_ids=segments_tensors).logits
        predictions = self.bert(tokens_tensor).logits

        topk = torch.topk(predictions[0][mask_idx], K, dim=-1).indices

        predictions = self.tokenizer.convert_ids_to_tokens(topk)

        for pred in predictions:
            rl_new = rhyme_length(pred, tgt_word)
            if rl_new > rl_original:
                return pred, rl_new

        return src_word, rl_original

    @torch.no_grad()
    def enhance(self, sentence: str, K: int = 200, rhyme: str = 'abab') -> str:
        assert rhyme in ('abab', 'abba')

        lines = sentence.split('\n')
        assert len(lines) == 4
        words = [line.split() for line in lines]

        rhyme_data_first = RhymeData(words, (0, 2) if rhyme == 'abab' else (1, 3), self.tokenizer)
        rhyme_data_second = RhymeData(words, (2, 0) if rhyme == 'abab' else (3, 1), self.tokenizer)

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
            words[0 if rhyme == 'abab' else 1][-1] = first_word
        else:
            words[2 if rhyme == 'abab' else 3][-1] = second_word

        lines = [' '.join(words[i]) for i in range(4)]
        sentence = '\n'.join(lines)
        return sentence


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
