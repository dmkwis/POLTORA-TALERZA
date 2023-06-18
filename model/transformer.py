import torch.nn as nn
from dataclasses import dataclass
from pos_enc import PositionalEncoding

@dataclass
class TransformerConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    max_len: int
    nhead: int
    dropout: float
    ff_dim: int 

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, config: TransformerConfig):
        super(TransformerEncoderDecoder, self).__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
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
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, src, tgt):
        src_embedded = self.embedding(src)
        tgt_embedded = self.embedding(tgt)

        src_encoded = self.pos_encoder(src_embedded)
        tgt_encoded = self.pos_decoder(tgt_embedded)

        output = self.transformer(src_encoded, tgt_encoded)
        output = self.fc(output)

        return output