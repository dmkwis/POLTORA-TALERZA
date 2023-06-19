from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int
    hidden_size: int
    num_layers: int
    num_heads: int
    ff_dim: int
    dropout: float
    max_seq_length: int
