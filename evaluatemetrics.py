from metrics.overlapmetric import OverlapMetric
from metrics.rhymedensitymetric import RhymeDensityMetric
import argparse
from fileparser import FileParser
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--rap_path', type=str, required=True)
    args = parser.parse_args()
    metrics = [OverlapMetric, RhymeDensityMetric]
    prompts = []
    generated_texts = []

    for metric in metrics:
        values = []
        with open(args.prompt_path, 'r') as pf:
            with open(args.rap_path, 'r') as rf:
                p_parser = FileParser(pf)
                r_parser = FileParser(rf)
                for prompt, text in zip(p_parser, r_parser):
                        values.append(metric.compute(prompt, text))
        print("MEAN:", np.mean(values), "STD:", np.std(values))
        
        