import torch
from torch.utils.data import DataLoader, Dataset
from BERT import BERT, BERTConfig

# Define a dummy dataset for testing
class DummyDataset(Dataset):
    def __init__(self, vocab_size, max_seq_length):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

    def __len__(self):
        return 10  # Number of samples in the dataset

    def __getitem__(self, index):
        input_tokens = torch.randint(0, self.vocab_size, (self.max_seq_length,))
        attention_mask = torch.randint(0, 2, (self.max_seq_length,)).bool() # Random attention mask
        return input_tokens, attention_mask

# Instantiate the BERT model
config = BERTConfig(vocab_size=10000, hidden_size=256, num_heads=4, num_layers=6, max_seq_length=128, ff_dim=64, dropout=0.1)
model = BERT(config)

# Create a DataLoader with the dummy dataset
dataset = DummyDataset(config.vocab_size, config.max_seq_length)
dataloader = DataLoader(dataset, batch_size=7, shuffle=True)

# Pass a random example input through the model
input_tokens, attention_mask = next(iter(dataloader))
attention_mask = attention_mask.view(-1, config.max_seq_length)
logits = model(input_tokens, attention_mask)

# Print the shapes of the input and output tensors
print("Input tokens shape:", input_tokens.shape)
print("Attention mask shape:", attention_mask.shape)
print("Logits shape:", logits.shape)
