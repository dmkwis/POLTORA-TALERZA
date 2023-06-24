import argparse
from dataset import w2i, i2w, PAD_TOKEN, START_TOKEN, END_TOKEN
import dataset
import torch
from model.transformer import TransformerEncoderDecoder
from model.config import ModelConfig
from train import TrainingModule


# take the sequence of outputs from the transformer and convert it into a sentence
def translate_output(transformer_outputs: torch.tensor) -> str:
    result = ''
    _, n, l = transformer_outputs.shape
    for i1 in range(n):
        i = torch.argmax(transformer_outputs[0, i1, :])
        word = i2w[i.item()]
        print(i1, word)
        if word != START_TOKEN and word != END_TOKEN and word != PAD_TOKEN:
            result += word + ' '
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()


    PATH = args.path

    dataset.fill_w2i()

    module = TrainingModule.load_from_checkpoint(checkpoint_path=PATH, map_location=torch.device('cpu'))

    model = module.transformer

    model.eval()

    provider = dataset.LyricsDatasetProvider()
    data = provider.get_dataset('finetune', training=False)
    dataloader = dataset.DataLoader(data, batch_size=1)

    for x, y in dataloader:
        w = model(
            x, 
            y,
            src_key_padding_mask=(x == w2i[PAD_TOKEN]),
            tgt_key_padding_mask=(y == w2i[PAD_TOKEN]),
        )
        print(w)
        print(translate_output(w))
        a = input()
