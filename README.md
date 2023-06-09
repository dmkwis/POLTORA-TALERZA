# Artificial rapper

Transformer-based rap lyrics generator. Highly based on the [Rapformer paper](https://arxiv.org/abs/2004.03965).

## Data preprocessing

We use the [dataset from Kaggle](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres). To create the train and test datasets perform the following:

1. Download the files `artists-data.csv` and `lyrics-data.csv` from Kaggle and place them in `data/` folder in the project root.

2. Run the following to generate files which will contain train and test examples:
```bash
python3 data.py
```
This will create 4 files: `lyrics_[train|test]_[x|y].txt`.