import torch.nn as nn
from dataset import w2i, PAD_TOKEN

from model.pos_enc import PositionalEncoding
from model.config import ModelConfig


class TransformerEncoderDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super(TransformerEncoderDecoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, w2i[PAD_TOKEN])
        self.hidden_size = config.hidden_size
        self.pos_encoder = PositionalEncoding(config.hidden_size, config.max_seq_length)
        self.pos_decoder = PositionalEncoding(config.hidden_size, config.max_seq_length)

        self.transformer = nn.Transformer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation='relu',
            batch_first=True,
        )

        self.fc = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, src, tgt, **kwargs):
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)

        src_encoded = self.pos_encoder(src_embedded)
        tgt_encoded = self.pos_decoder(tgt_embedded)

        output = self.transformer(src_encoded, tgt_encoded, **kwargs)
        output = self.fc(output)

        return output
