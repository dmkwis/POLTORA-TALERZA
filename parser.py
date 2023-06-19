import argparse
from typing import Dict, Union


def parse_arguments() -> Dict[str, Union[int, str, float, bool]]:
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--hidden_size', type=int, default=2,
                        help='size of word embeddings')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--num_heads', type=int, default=2,
                        help='number of heads per layer')
    parser.add_argument('--ff_dim', type=int, default=2,
                        help='size of embeddings in feed forward layer')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout applied to layers (0 = no dropout)')

    # Training params
    parser.add_argument('--epochs', type=int, default=2,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='batch size')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--seed', type=int, default=None,
                        help='random seed')
    parser.add_argument('--gpu', action='store_true', default=False,
                        help='use gpu')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='use Wandb')
    parser.add_argument('--save', action='store_true', default=False,
                        help='whether to save models in checkpoints directory')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of workers for dataset')

    args = parser.parse_args()
    args = vars(args)
    return args
