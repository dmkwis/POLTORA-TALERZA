import argparse
from dataset import w2i, i2w, PAD_TOKEN, START_TOKEN, END_TOKEN, NEWLINE_TOKEN
import dataset
import torch
from model.transformer import TransformerEncoderDecoder
from model.config import ModelConfig
from train import TrainingModule
from fileparser import FileParser

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
    parser.add_argument('--manual', action='store_true', default=False)
    parser.add_argument('--preset', type=str, default=False)
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


    if args.preset:
        parser = FileParser(open(args.preset, 'r'))

    with torch.no_grad():
        with open(args.savefile, 'w') as f:
            i = 0
            for x, y in dataloader:
                if i == args.num_examples:
                    break
                print(i)
                i += 1
                output_sequence = start_token

                if args.manual or args.preset:
                    # getting content words from the input
                    print('Input 4 lines of content words:\n')
                    words = [START_TOKEN]
                    if args.preset:
                        try:
                            lines = next(parser).splitlines()
                        except:
                            break
                    for i1 in range(4):
                        if args.preset:
                            line = lines[i1]
                        else:
                            line = input()
                        line_words = line.split(' ')
                        words.extend(line_words)
                        if i1 < 3:
                            words.append(NEWLINE_TOKEN)
                    words.append(END_TOKEN)
                    print(words)

                    # reject too long sequences
                    if len(words) > len(x[0]):
                        print('Error: Too long sequence')
                        continue

                    while len(words) < len(x[0]):
                        words.append(PAD_TOKEN)

                    # reject examples with unknown words
                    all_in_vocab = True
                    for w in words:
                        if w not in w2i:
                            print('Error: unknown word:', w)
                            all_in_vocab = False
                            break
                    if not all_in_vocab:
                        continue

                    # tokenize
                    words = [w2i[w] for w in words]

                    # make batched tensor
                    x = torch.tensor([words])
                    

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