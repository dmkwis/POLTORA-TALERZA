import argparse
from dataset import w2i, i2w, PAD_TOKEN, START_TOKEN, END_TOKEN, NEWLINE_TOKEN
import dataset
import torch
from model.transformer import TransformerEncoderDecoder
from model.config import ModelConfig
from train import TrainingModule


# take the sequence of outputs from the transformer and convert it into a sentence
def translate_output(transformer_outputs: torch.tensor) -> str:
    result = ''
    for token in transformer_outputs:
        word = i2w[token.item()]
        if word == END_TOKEN:
            break
        if word == START_TOKEN or word == PAD_TOKEN:
            continue
        if word == NEWLINE_TOKEN:
            result += word
        else:
            result += word + ' '
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--savefile', type=str, required=True)
    parser.add_argument('--num_examples', type=int, required=True)
    args = parser.parse_args()


    PATH = args.path

    dataset.fill_w2i()

    module = TrainingModule.load_from_checkpoint(checkpoint_path=PATH, map_location=torch.device('cpu'))

    model = module.transformer

    model.eval()

    provider = dataset.LyricsDatasetProvider()
    data = provider.get_dataset('finetune', training=False)
    dataloader = dataset.DataLoader(data, batch_size=1, shuffle=True)

    start_token = torch.tensor([[w2i[START_TOKEN]]])

    with torch.no_grad():
        with open(args.savefile, 'w') as f:
            i = 0
            for x, y in dataloader:
                if i == args.num_examples:
                    break
                print(i)
                i += 1
                output_sequence = start_token
                for _ in range(len(y[0])):
                    output = model(
                        x, 
                        output_sequence,
                        src_key_padding_mask=(x == w2i[PAD_TOKEN]),
                        tgt_key_padding_mask=None
                    )
                    last_token = torch.argmax(output[:, -1, :], dim=1, keepdim=True)
                    output_sequence = torch.cat((output_sequence, last_token), dim=1)
                f.write(translate_output(output_sequence[0]))
                if i != args.num_examples:
                    f.write("\n\n")
                else:
                    f.write("\n")