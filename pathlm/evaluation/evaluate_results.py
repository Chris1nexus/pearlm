import argparse
import os
import pickle

from pathlm.evaluation.eval_metrics import evaluate_rec_quality_from_results
from pathlm.evaluation.eval_utils import get_set, get_result_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2-large', help='which model to evaluate')
    parser.add_argument('--dataset', type=str, default='ml1m', help='which dataset evaluate')
    parser.add_argument('--sample_size', default='1000', type=str,
                        help='')
    parser.add_argument('--n_hop', default='3', type=str, help='')
    parser.add_argument('--decoding_strategy', default='gcd', type=str, help='')
    args = parser.parse_args()

    if args.model in ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl']:
        args.model = f'end-to-end@{args.dataset}@{args.model}@{args.sample_size}@{args.n_hop}@{args.decoding_strategy}'
    result_dir =  get_result_dir(args.dataset, args.model)
    test_set = get_set(args.dataset, set_str='test')

    with open(os.path.join(result_dir, 'topk_items.pkl'), 'rb') as f:
        topk_items = pickle.load(f)

    evaluate_rec_quality_from_results(args.dataset, args.model, test_set)