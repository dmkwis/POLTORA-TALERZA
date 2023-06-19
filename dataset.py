from typing import Dict, List, Tuple
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch


data_files = {
    'train_x': 'lyrics_train_x.txt',
    'train_y': 'lyrics_train_y.txt',
    'test_x': 'lyrics_test_x.txt',
    'test_y': 'lyrics_test_y.txt',
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


def get_all_words() -> List[str]:
    words = []

    for _, filename in data_files.items():
        with open(f'data/{filename}') as file:
            s = file.read()
            s = s.replace('\n\n', ' ')
            s = s.replace('\n', ' ')
            s_words = s.split(' ')
            words.extend(s_words)

    words = sorted(list(set(words))) # sort to always get the same output

    # include start, end, pad tokens as words
    # include newline as we want the model to use it to separate lines in verse
    words = [PAD_TOKEN, START_TOKEN, END_TOKEN, '\n'] + words

    return words


def get_word_to_int() -> Dict[str, int]:
    words = get_all_words()
    word_to_int = {}
    
    for i, w in enumerate(words):
        word_to_int[w] = i
    
    return word_to_int


def get_data(train: bool = True) -> Tuple[List[List[str]], List[List[str]]]:
    file_x, file_y = ('train_x', 'train_y') if train else ('test_x', 'test_y')

    with open(f"data/{data_files[file_x]}") as file:
        x = file.read()
    with open(f"data/{data_files[file_y]}") as file:
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
    w2i = get_word_to_int()
    x, y = get_data(train = True)
    dataset = LyricsDataset(w2i, x, y)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # check iterate over dataloader
    for x, y in dataloader:
        pass
