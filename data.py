import pandas as pd
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from typing import List, Tuple
import string
import random
import math


def read_data():
    df_artists = pd.read_csv('data/artists-data.csv')
    df_lyrics = pd.read_csv('data/lyrics-data.csv')
    df_lyrics.rename(columns={"ALink": "Link"}, inplace=True)
    merged_dfs = df_lyrics.merge(df_artists, how='inner', on='Link')
    return merged_dfs


def get_songs_of_language(df_lyrics: pd.DataFrame, language: str):
    return df_lyrics[df_lyrics['language'] == language]


def get_rap_songs(df: pd.DataFrame):
    df.dropna(inplace=True)
    df = df[
        (df['Genres'].str.contains('Hip Hop')) |
        (df['Genres'].str.contains('Rap'))
    ]
    return df['Lyric'].tolist()


def line_meaningless(line: str): # used to check if line denotes song fragment like verse / chorus
    if line.startswith('[') and line.endswith(']'):
        return True
    if 'Verse' in line or 'Chorus' in line:
        return True
    if line.endswith(':'):
        return True


def get_song_verses(song: str):

    # output verses list
    verses = []

    # separate fragments divided by \n\n
    fragments = song.split('\n\n')

    for fragment in fragments:
        lines = fragment.split('\n')

        # filter out lines denoting fragment name: [Chorus] etc...
        lines = [l for l in lines if not line_meaningless(l)]

        # group 4 lines to form a verse
        if len(lines) < 4:
            continue

        for i in range(0, len(lines), 4):
            if i + 4 <= len(lines):
                verse = lines[i: i + 4]
            else:
                # if can't fit 4 lines then take the last 4
                verse = lines[len(lines) - 4: len(lines)]
            # skip broken verses
            if '' in verse:
                continue
            verses.append('\n'.join(verse))
    
    return verses


def get_verses(songs: List[str]):
    verses = []
    for song in songs:
        verses.extend(get_song_verses(song))
    return verses


def create_base_dataset(verses: List[str]):
    Y = verses
    X = []
    Ynew = []

    stop_words = set(stopwords.words('english'))

    for i, y in enumerate(Y):

        if i % 1000 == 0:
            print(f'base dataset progress: {i} / {len(Y)}')

        # remove punctuation and convert to lowercase
        y = y.translate(str.maketrans('', '', string.punctuation))
        y = y.lower()

        # tokenize
        x = y.split('\n')
        x = [word_tokenize(l) for l in x]

        # remove stop words and numbers (keep alpha)
        x = [[w for w in l if not w in stop_words] for l in x]
        x = [[w for w in l if w.isalpha()] for l in x]

        # convert back to str
        x = [' '.join(l) for l in x]
        # skip useless content words
        if '' in x:
            continue
        x = '\n'.join(x)

        X.append(x)
        Ynew.append(y)
    
    return X, Ynew


def verse_to_wordlist(verse: str): # list of words which can be put back to verse by investigating (line, pos) index
    lines = verse.split('\n')
    wordlist = []
    for i, l in enumerate(lines):
        ws = l.split(' ')
        words = [(w, i, j) for j, w in enumerate(ws)]
        wordlist.extend(words)
    return wordlist


def wordlist_to_verse(words: List[Tuple[str, int, int]]): # reconstruct verse from wordlist
    verse = [[], [], [], []]
    for w, i, j in words:
        verse[i].append((w, j))

    for i in range(len(verse)):
        verse[i] = sorted(verse[i], key=lambda x: x[1])
        verse[i] = [w for w, _ in verse[i]]
    
    verse = [' '.join(l) for l in verse]
    verse = '\n'.join(verse)
    return verse


def noise_synonyms(verse: str):
    words = verse_to_wordlist(verse)
    random.shuffle(words)
    to_alter = math.ceil(1 / 5 * len(words))
    altered = 0

    for k in range(len(words)):
        w, i, j = words[k]
        syns = wordnet.synsets(w)
        syns = [s.lemmas()[0].name() for s in syns]
        syns = [s for s in syns if s != w and s.isalpha() and s.islower()]
        
        if len(syns) == 0:
            continue

        synonym = syns[random.randint(0, len(syns) - 1)]
        words[k] = synonym, i, j
        altered += 1
        if altered >= to_alter:
            break

    return wordlist_to_verse(words)


def noise_drop(verse: str):
    words = verse_to_wordlist(verse)
    num = len(words)
    random.shuffle(words)
    firstmet = [1337, 1337, 1337, 1337]
    for _, i, j in words:
        firstmet[i] = min(firstmet[i], j)
    
    # gotta leave at least 1 word per line
    remain_words = [(w, i, j) for w, i, j in words if firstmet[i] == j]
    words_to_change = [(w, i, j) for w, i, j in words if firstmet[i] != j]

    to_delete = math.ceil(1 / 5 * num)
    to_leave = len(words_to_change) - to_delete

    words = words_to_change[:to_leave]
    words.extend(remain_words)

    return wordlist_to_verse(words)


def noise_shuffle(verse: str):
    lines = verse.split('\n')
    lines = [l.split(' ') for l in lines]
    for l in lines:
        random.shuffle(l)
    lines = [' '.join(l) for l in lines]
    verse = '\n'.join(lines)
    return verse


def create_noised_samples(X: List[str]):
    Xnew = []

    # iterate dataset and introduce one type of noise, each with probability 1 / 3 
    for i, x in enumerate(X):

        # print progress per each 1000 samples
        if i % 1000 == 0:
            print(f'noising progress: {i} / {len(X)}', i)

        action = random.randint(0, 2)

        # shuffle
        if action == 0:
            xnew = noise_shuffle(x)
        
        # drop
        if action == 1:
            xnew = noise_drop(x)

        # synonym
        if action == 2:
            xnew = noise_synonyms(x)

        Xnew.append(xnew)

    return Xnew


if __name__ == '__main__':
    nltk.download('stopwords')
    nltk.download('wordnet')

    data = read_data()
    data = get_songs_of_language(data, 'en')
    data = get_rap_songs(data)
    verses = get_verses(data)

    X, Y = create_base_dataset(verses)
    X = create_noised_samples(X)

    D = list(zip(X, Y))
    random.shuffle(D)

    num_train_samples = math.floor(4 / 5 * len(D))
    train_pairs = D[:num_train_samples]
    test_pairs = D[num_train_samples:]

    t = list(zip(*train_pairs))
    train_x, train_y = list(t[0]), list(t[1])

    t = list(zip(*test_pairs))
    test_x, test_y = list(t[0]), list(t[1])

    with open('data/lyrics_train_x.txt', 'w') as file:
        s = '\n\n'.join(train_x)
        file.write(s)

    with open('data/lyrics_train_y.txt', 'w') as file:
        s = '\n\n'.join(train_y)
        file.write(s)

    with open('data/lyrics_test_x.txt', 'w') as file:
        s = '\n\n'.join(test_x)
        file.write(s)

    with open('data/lyrics_test_y.txt', 'w') as file:
        s = '\n\n'.join(test_y)
        file.write(s)
