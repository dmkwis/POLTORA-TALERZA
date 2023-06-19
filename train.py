import time

import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import dataset
from parser import parse_arguments
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
        self.loss_fn = nn.CrossEntropyLoss()

    def step(self, batch, batch_idx: int, mode='train'):
        x, y = batch

        y_input = y[:, :-1]
        y_expected = y[:, 1:]

        # Returns [batch_size, max_seq_len, vocab_size] representing logits for each word in vocab at each position
        output = self.transformer(
            x,
            y_input,
            tgt_mask=nn.Transformer.generate_square_subsequent_mask(y_input.shape[-1], self.device),
            # src_key_padding_mask=None,
            # tgt_key_padding_mask=None,
        )

        output_flat = output.view(-1, output.shape[-1])
        loss = self.loss_fn(output_flat, y_expected.reshape(-1))
        self.log('loss_' + mode, loss, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx: int):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx: int):
        return self.step(batch, batch_idx, 'valid')

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.transformer.parameters(),
            lr=self.hparams.lr,
        )

        return optimizer


def set_seed(seed: int, gpu: bool):
    pl.seed_everything(seed)
    if gpu:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    run_id = str(time.strftime("%H:%M:%S_%d_%m", time.localtime()))
    print('Starting run:', run_id)
    args = parse_arguments()

    if args['seed'] is not None:
        set_seed(args['seed'], args['gpu'])

    w2i = dataset.get_word_to_int()
    train_x, train_y = dataset.get_data(train=True)
    train_dataset = dataset.LyricsDataset(w2i, train_x, train_y)
    train_dataloader = dataset.DataLoader(
        train_dataset,
        batch_size=args['batch_size'],
        num_workers=args['workers'],
        shuffle=True,
    )

    test_x, test_y = dataset.get_data(train=False)
    test_dataset = dataset.LyricsDataset(w2i, test_x, test_y)
    test_dataloader = dataset.DataLoader(
        test_dataset,
        batch_size=args['batch_size'],
        num_workers=args['workers'],
    )

    args['vocab_size'] = len(w2i)
    args['max_seq_length'] = max(train_dataset.block_size, test_dataset.block_size)
    print('Using args:', args)

    model = TrainingModule(**args)

    if args['wandb']:
        logger = WandbLogger(
            project='poltora-talerza',
            name='Train_run_' + run_id,
            log_model='all',
        )
        logger.watch(model)
    else:
        logger = None

    trainer_kwargs = {
        'max_epochs': args['epochs'],
        'logger': logger,
    }

    if args['gpu']:
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = 1

    if args['save']:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath='checkpoints/run_' + run_id,
            filename='{epoch}-{loss_train:.2f}-{loss_valid:.2f}',
            save_last=True,
            every_n_epochs=1,
        )
        trainer_kwargs['callbacks'] = checkpoint_callback

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(
        model,
        train_dataloader,
        test_dataloader,
    )
