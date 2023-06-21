from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import random
import math
import numpy as np


data_files = {
    'pretrain_x': 'pretrain_x.txt',
    'pretrain_y': 'pretrain_y.txt',
    'finetune_x': 'finetune_x.txt',
    'finetune_y': 'finetune_y.txt',
}

START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'
NEWLINE_TOKEN = '\n'

# global variables
w2i = None
i2w = None


# take the sequence of outputs from the transformer and convert it into a sentence
# assumes w2i was already computed
def translate_output(transformer_outputs: List[torch.Tensor]) -> str:
    global w2i
    global i2w
    if w2i is None or i2w is None:
        fill_w2i()
    result = ''
    for output in transformer_outputs:
        i = np.argmax(output)
        word = i2w[i.item()]
        if word != START_TOKEN and word != END_TOKEN and word != PAD_TOKEN:
            result += word + ' '
    return result


class LyricsDataset(Dataset):
    def __init__(self, x: List[List[str]], y: List[List[str]]) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.block_size = max([len(v) + 2 for v in x]) # + 2 for start, end tokens
        self.block_size = max(self.block_size, max([len(v) + 2 for v in y]))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = [START_TOKEN] + self.x[index] + [END_TOKEN]
        y = [START_TOKEN] + self.y[index] + [END_TOKEN]

        x.extend([PAD_TOKEN] * (self.block_size - len(x)))
        y.extend([PAD_TOKEN] * (self.block_size - len(y) + 1))

        x = torch.tensor([w2i[w] for w in x], dtype=torch.long)
        y = torch.tensor([w2i[w] for w in y], dtype=torch.long)

        return x, y


class LyricsDatasetProvider:
    def __init__(self, train_frac: float = 0.8) -> None:
        assert train_frac >= 0.0 and train_frac <= 1.0        
        self.train_frac = train_frac
        global w2i

        if w2i is None:
            fill_w2i()

    def get_dataset(self, name: str, training: bool = True):
        assert name in ['pretrain', 'finetune']
        set_x, set_y = get_data(name)
        num = math.floor(self.train_frac * len(set_y))

        if training:
            x = set_x[:num]
            y = set_y[:num]
        else:
            x = set_x[num:]
            y = set_y[num:]

        return LyricsDataset(x, y)


def fill_w2i():
    # w2i
    global w2i
    pretrain_x, pretrain_y = get_data('pretrain')
    finetune_x, finetune_y = get_data('finetune')
    words = get_all_words([pretrain_x, pretrain_y, finetune_x, finetune_y])
    w2i = get_word_to_int(words)

    # free up the memory
    del pretrain_x
    del pretrain_y
    del finetune_x
    del finetune_y

    # i2w
    global i2w
    i2w = {}
    for key, val in w2i.items():
        i2w[val] = key


def get_all_words(data: List[List[List[str]]]) -> List[str]:
    words = []
    for part in data:
        for verse in part:
            words.extend(verse)

    words = sorted(list(set(words))) # sort to always get the same output
    # include start, end, pad tokens as words
    # include newline as we want the model to use it to separate lines in verse
    words = [PAD_TOKEN, START_TOKEN, END_TOKEN, NEWLINE_TOKEN] + words
    return words


def get_word_to_int(words) -> Dict[str, int]:
    word_to_int = {}
    for i, w in enumerate(words):
        word_to_int[w] = i    
    return word_to_int


def get_data(name: str) -> Tuple[List[List[str]], List[List[str]]]:
    with open(f"data/{name}_x.txt") as file:
        x = file.read()
    with open(f"data/{name}_y.txt") as file:
        y = file.read()

    x = x.split('\n\n')
    # treat newline as a separate word
    x = [v.replace('\n', ' \n ') for v in x]
    # each verse in x is a list of words, including '\n's
    x = [v.split(' ') for v in x]

    y = y.split('\n\n')
    y = [v.replace('\n', ' \n ') for v in y]
    y = [v.split(' ') for v in y]

    return x, y


if __name__ == '__main__':

    fill_w2i()
    a = torch.rand(len(w2i))
    b = torch.rand(len(w2i))
    output = translate_output([a, b])
    print(output)

    dataset_provider = LyricsDatasetProvider()

    print('pretrain train')    
    dataset = dataset_provider.get_dataset('pretrain', training=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # check iterate over dataloader
    for x, y in dataloader:
        pass

    print('pretrain test')    
    dataset = dataset_provider.get_dataset('pretrain', training=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # check iterate over dataloader
    for x, y in dataloader:
        pass

    print('finetune train')    
    dataset = dataset_provider.get_dataset('finetune', training=True)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # check iterate over dataloader
    for x, y in dataloader:
        pass

    print('finetune test')    
    dataset = dataset_provider.get_dataset('finetune', training=False)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # check iterate over dataloader
    for x, y in dataloader:
        pass
