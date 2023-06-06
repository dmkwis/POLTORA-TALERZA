import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from dataclasses import dataclass
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, dropout_rate, max_length):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)

        position = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * -(np.log(10000.0) / hidden_size))
        pe = torch.zeros(max_length, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

@dataclass
class BERTConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    ff_dim: int
    dropout: float
    max_seq_length: int


class BERT(nn.Module):
    def __init__(self, config: BERTConfig):
        super(BERT, self).__init__()

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_encoding = PositionalEncoding(config.hidden_size, config.dropout, config.max_seq_length)

        encoder_layers = TransformerEncoderLayer(config.hidden_size, config.num_heads, config.ff_dim, config.dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.num_layers)

        self.fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, attention_mask):
        # Embedding and positional encoding
        embedded = self.embedding(input_ids)
        encoded = self.positional_encoding(embedded)

        # Transformer encoder with attention mask
        encoded = self.transformer_encoder(encoded, src_key_padding_mask=attention_mask)

        # Pooling
        pooled = encoded.mean(dim=1)  # Take the mean across the sequence length

        # Fully connected layer and activation
        pooled = self.fc(pooled)
        pooled = self.activation(pooled)

        # Apply dropout
        pooled = self.dropout(pooled)

        return pooled
