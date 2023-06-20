from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch
import random
import math


data_files = {
    'pretrain_x': 'pretrain_x.txt',
    'pretrain_y': 'pretrain_y.txt',
    'finetune_x': 'finetune_x.txt',
    'finetune_y': 'finetune_y.txt',
}

START_TOKEN = '<START>'
END_TOKEN = '<END>'
PAD_TOKEN = '<PAD>'


class LyricsDataset(Dataset):
    def __init__(self, word_to_int: Dict[str, int], x: List[List[str]], y: List[List[str]]) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.w2i = word_to_int
        self.block_size = max([len(v) + 2 for v in x]) # + 2 for start, end tokens
        self.block_size = max(self.block_size, max([len(v) + 2 for v in y]))
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = [START_TOKEN] + self.x[index] + [END_TOKEN]
        y = [START_TOKEN] + self.y[index] + [END_TOKEN]

        x.extend([PAD_TOKEN] * (self.block_size - len(x)))
        y.extend([PAD_TOKEN] * (self.block_size - len(y) + 1))

        x = torch.tensor([self.w2i[w] for w in x], dtype=torch.long)
        y = torch.tensor([self.w2i[w] for w in y], dtype=torch.long)

        return x, y


class LyricsDatasetProvider:
    def __init__(self, shuffle: bool = True, pretrain_frac: float = 1.0, train_frac: float = 0.8) -> None:
        # choose pretrain frac below 1.0 to only use part of dataset for pretraining
        assert pretrain_frac >= 0.0 and pretrain_frac <= 1.0
        assert train_frac >= 0.0 and train_frac <= 1.0
        
        self.train_frac = train_frac

        print('loading pretrain')
        self.pretrain_x, self.pretrain_y = get_data('pretrain')
        print('loading finetune')
        self.finetune_x, self.finetune_y = get_data('finetune')

        if shuffle:
            a = zip(self.pretrain_x, self.pretrain_y)
            random.shuffle(a)
            b = list(zip(*a))
            self.pretrain_x, self.pretrain_y = list(b[0]), list(b[1])

            a = zip(self.finetune_x, self.finetune_y)
            random.shuffle(a)
            b = list(zip(*a))
            self.finetune_x, self.finetune_y = list(b[0]), list(b[1])

        # optionally shrink the pretrain dataset
        if pretrain_frac < 1.0:
            num_pretrain = math.floor(pretrain_frac * len(self.pretrain_y))
            a = zip(self.pretrain_x, self.pretrain_y)
            a = a[:num_pretrain]
            b = list(zip(*a))
            self.pretrain_x, self.pretrain_y = list(b[0]), list(b[1])

        self.words = get_all_words([
            self.pretrain_x,
            self.pretrain_y,
            self.finetune_x,
            self.finetune_y,
        ])
        self. w2i = get_word_to_int(self.words)

    def get_pretrain_train(self):
        num = math.floor(self.train_frac * len(self.pretrain_y))
        x = self.pretrain_x[:num]
        y = self.pretrain_y[:num]
        return LyricsDataset(self.w2i, x, y)
    
    def get_pretrain_test(self):
        num = math.floor(self.train_frac * len(self.pretrain_y))
        x = self.pretrain_x[num:]
        y = self.pretrain_y[num:]
        return LyricsDataset(self.w2i, x, y)

    def get_finetune_train(self):
        num = math.floor(self.train_frac * len(self.finetune_y))
        x = self.finetune_x[:num]
        y = self.finetune_y[:num]
        return LyricsDataset(self.w2i, x, y)

    def get_finetune_test(self):
        num = math.floor(self.train_frac * len(self.finetune_y))
        x = self.finetune_x[num:]
        y = self.finetune_y[num:]
        return LyricsDataset(self.w2i, x, y)


def get_all_words(data: List[List[List[str]]]) -> List[str]:
    words = []

    for part in data:
        for verse in part:
            words.extend(verse)
            words = list(set(words))

    words = sorted(list(set(words))) # sort to always get the same output

    # include start, end, pad tokens as words
    # include newline as we want the model to use it to separate lines in verse
    words = [PAD_TOKEN, START_TOKEN, END_TOKEN, '\n'] + words

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

    dataset_provider = LyricsDatasetProvider()

    print('pretrain train')    
    dataset = dataset_provider.get_pretrain_train()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # check iterate over dataloader
    for x, y in dataloader:
        pass

    print('pretrain test')    
    dataset = dataset_provider.get_pretrain_test()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # check iterate over dataloader
    for x, y in dataloader:
        pass

    print('finetune train')    
    dataset = dataset_provider.get_finetune_train()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # check iterate over dataloader
    for x, y in dataloader:
        pass

    print('finetune test')    
    dataset = dataset_provider.get_finetune_test()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # check iterate over dataloader
    for x, y in dataloader:
        pass
