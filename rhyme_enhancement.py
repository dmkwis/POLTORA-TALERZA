import torch
import random
from copy import deepcopy
from typing import List, Tuple


class RhymeEnhancer:
    def __init__(self):
        self.tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')
        self.bert = torch.hub.load('huggingface/pytorch-transformers', 'modelForMaskedLM', 'bert-base-cased')

    def find_match(self, indexed_tokens: List[int], K: int, mask_idx: int, src_word: str, tgt_word: str, segments_tensors: torch.Tensor) -> Tuple[str, float]:
        # TODO replace with actual rhyme metric
        def rhyme_length(a: str, b: str) -> float:
            print(f'Calculating rhyme length of word \'{a}\' and \'{b}\'')
            return random.random()

        # src_word = self.tokenizer.convert_ids_to_tokens(indexed_tokens[mask_idx])
        rl_original = rhyme_length(src_word, tgt_word)

        indexed_tokens[mask_idx] = self.tokenizer.mask_token_id
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
        words_first = deepcopy(words)
        # words_second = deepcopy(words)

        src_first = words_first[0][-1]
        tgt_first = words_first[2][-1]
        words_first[0][-1] = self.tokenizer.mask_token

        # words_second[2][-1] = self.tokenizer.mask_token

        lines_first = [' '.join(words_first[i]) for i in range(4)]
        print('Lines changed', lines_first)
        sentence_first = self.tokenizer.sep_token.join(lines_first)
        print('Sentence changed', sentence_first)

        # lines_second = [' '.join(words_second[i]) for i in range(4)]
        # print('Lines changed', lines_second)
        # sentence_second = '\n'.join(lines_second)
        # print('Sentence changed', sentence_second)

        # segment_ids = [[i] * len(line.split()) for i, line in enumerate(lines)]
        # print(segment_ids)

        indexed_tokens = self.tokenizer.encode(sentence_first, add_special_tokens=True)
        print(indexed_tokens)

        # debug print
        print()
        for token in indexed_tokens:
            print(token, self.tokenizer.convert_ids_to_tokens(token))
        print()

        segments_ids = []
        mask_idxs = []
        separators_found = 0
        for i, token in enumerate(indexed_tokens):
            segments_ids.append(separators_found)
            if token == self.tokenizer.sep_token_id:
                separators_found += 1
            elif token == self.tokenizer.mask_token_id:
                mask_idxs.append(i)

        print(segments_ids)
        print(mask_idxs)

        segments_tensors = torch.tensor([segments_ids])

        first_word, rl_first = self.find_match(indexed_tokens, K, mask_idxs[0], src_first, tgt_first, segments_tensors)
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
