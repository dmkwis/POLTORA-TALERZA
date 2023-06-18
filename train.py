import time
import random
import argparse
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl

from dataset import generate_dataset, get_vocab_size
from model.config import ModelConfig
from model.transformer import TransformerEncoderDecoder


class TrainingModule(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        config = ModelConfig(
            self.hparams.vocab_size,
            self.hparams.hidden_size,
            self.hparams.num_layers,
            self.hparams.num_heads,
            self.hparams.ff_dim,
            self.hparams.dropout,
            self.hparams.max_seq_length,
        )
        self.transformer = TransformerEncoderDecoder(config)

    def training_step(self, batch, batch_idx: int):
        x, y = batch

        output = self.transformer(x, y)

        return output

    def validation_step(self, batch, batch_idx: int):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.transformer.parameters(),
            lr=self.hparams.lr,
        )

        return optimizer


def parse_arguments() -> Dict[str, int]:
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--hidden_size', type=int, default=2,
                        help='size of word embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='number of heads per layer')
    parser.add_argument('--ff_dim', type=int, default=2,
                        help='size of embeddings in feed forward layer')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')

    # Training params
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='use gpu')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')

    args = parser.parse_args()
    args = vars(args)
    return args


def set_seed(seed: int):
    print('Setting seed to:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = parse_arguments()

    if args['seed'] is not None:
        set_seed(args['seed'])

    # TODO
    args['vocab_size'] = get_vocab_size()
    args['max_seq_length'] = 568
    print('Using args:', args)

    model = TrainingModule(**args)

    if args['gpu']:
        trainer = pl.Trainer(
            max_epochs=args['epochs'],
            accelerator='gpu',
            devices=1,
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args['epochs'],
        )

    # train_dataloader, test_dataloader = generate_dataset()
    train_dataloader = generate_dataset(args['batch_size'])
    print('Generated dataset')

    trainer.fit(
        model,
        train_dataloader,
        # test_dataloader,
    )
