import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from dataclasses import dataclass
from pos_enc import PositionalEncoding

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
        self.positional_encoding = PositionalEncoding(config.hidden_size, config.max_seq_length)

        encoder_layers = TransformerEncoderLayer(config.hidden_size, config.num_heads, config.ff_dim, config.dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, config.num_layers)
        
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids, attention_mask):
        # Embedding and positional encoding
        embedded = self.embedding(input_ids)
        encoded = self.positional_encoding(embedded)

        # Transformer encoder with attention mask
        encoded = self.transformer_encoder(encoded, src_key_padding_mask=attention_mask)

        # Pooling
        pooled = encoded.mean(dim=1)  # Take the mean across the sequence length

        # Computing logits
        logits = self.fc(pooled)
        logits = self.activation(logits)

        return logits
