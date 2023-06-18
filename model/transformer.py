import torch.nn as nn
from pos_enc import PositionalEncoding

@dataclass
class TransformerConfig:
    input_size: int
    hidden_size: int
    output_size: int
    num_layers: int
    max_len: int
    nhead: int
    causal_attention: bool  # Whether to use causal self-attention or not

import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    input_size: int
    hidden_size: int
    output_size: int
    num_layers: int
    max_len: int
    nhead: int
    dropout: float
    ff_dim: int 

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerEncoderDecoder, self).__init__()
        self.hidden_size = config.hidden_size
        self.pos_encoder = PositionalEncoding(config.hidden_size, config.max_len)
        self.pos_decoder = PositionalEncoding(config.hidden_size, config.max_len)
        self.transformer = nn.Transformer(d_model=config.hidden_size,
                                          nhead=config.nhead,
                                          num_encoder_layers=config.num_layers,
                                          num_decoder_layers=config.num_layers,
                                          dim_feedforward=config.ff_dim,
                                          dropout=config.dropout,
                                          activation='relu')
        self.fc = nn.Linear(config.hidden_size, config.output_size)

    def forward(self, src, tgt):
        src = self.pos_encoder(src)
        tgt = self.pos_decoder(tgt)

        output = self.transformer(src, tgt)
        output = self.fc(output)

        return output