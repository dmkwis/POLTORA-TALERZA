# Artificial rapper

Transformer-based rap lyrics generator. Highly based on the [Rapformer paper](https://arxiv.org/abs/2004.03965).

## Data preprocessing

We use the [dataset from Kaggle](https://www.kaggle.com/datasets/neisse/scrapped-lyrics-from-6-genres). To create the train and test datasets perform the following:

1. Download the files `artists-data.csv` and `lyrics-data.csv` from Kaggle and place them in `data/` folder in the project root.

2. Run the following to generate files which will contain train and test examples:
```bash
python3 data.py
```
This will create 4 files: `pretrain_[x|y].txt` and `finetune_[x|y]` in the `data/` folder. After this step the datasets are available by using `LyricsDatasetProvider` and `LyricsDataset` from `dataset.py`.

After that run the following to reduce dataset sizes and the number of distinct tokens.
```bash
python3 filter.py
```
This will override the `pretrain_[x|y]` and `finetune_[x|y]` files.

## Training

### Wandb
To use Wandb
```bash
pip install wandb

wandb login
```
and paste your API key.

### Training script
Running `train.py` utilizes params, to check available run: 
```bash
python train.py --help
```

and to train model with chosen params run:
```bash
python train.py [params]
```
alternatively use training script to easily change previously used parameters
```bash
./run_train.sh
```

## Inference
Run `infer.py` with appropriate parameters. This will generate `results.txt` file.

## Rhyme enhancement
Run `rhyme_enhancement.py`. This assumes that there exists the file named `results.txt` which contains generated examples. It prints the rhyme-enhanced examples into the screen.
