import argparse
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_examples', type=int, required=True)
    parser.add_argument('--words_path', type=str, default="data/finetune_x.txt")
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    words = set()
    with open(args.words_path, 'r') as f:
        for line in f:
            for word in line.split():
                words.add(word)
    with open(args.save_path, 'w') as f:
        for i in range(args.num_examples):
            for j in range(4):
                hm = random.randint(1, 2)
                sample = random.sample(sorted(words), hm)
                line = " ".join(sample) + "\n"
                f.write(line)
            f.write("\n")


        