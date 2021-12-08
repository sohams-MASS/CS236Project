import argparse

import sys
import os
import yaml
import torch
import numpy as np
import pickle as pkl
from dataset import NIPS2015Dataset
from Baseline import RNN

import matplotlib
if os.environ.get('DISPLAY', '') == '':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

SAMPLE_SEQ_LEN = 1000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory of saving checkpoints')
    parser.add_argument('--data_dir', type=str, default='data/', help='Directory of papers.csv')
    parser.add_argument('--log_dir', type=str, default='logs/', help='Directory of putting logs')
    parser.add_argument('--gpu', action='store_true', help="Turn on GPU mode")

    args = parser.parse_args()
    return args

def plot_log_p(filename, dataset, rnn):
    with open(filename + '.pkl', 'rb') as f:
        lls = []
        data = pkl.load(f)
        for i, str in data.items():
            str_np = np.asarray([dataset.char2idx[c] for c in str])
            lls.append(rnn.compute_prob(str_np))

    with open(filename + '_raw.pkl', 'wb') as f:
        pkl.dump(lls, f, protocol=pkl.HIGHEST_PROTOCOL)

    plt.figure()
    plt.hist(lls)
    plt.xlabel('Log-likelihood')
    plt.xlim([-800, -50])
    plt.ylabel('Counts')
    plt.title(filename)
    plt.savefig(filename + '.png', bbox_inches='tight')
    plt.show()
    plt.close()
    print("# Figure written to %s.png." % filename)


def main():
    args = parse_args()
    config = parse_config(args)
    np.random.seed(config.seed)
    if args.gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(config.seed)

    dataset = get_dataset(expert_dataset);

    rnn = RNN(
        vocab_size=dataset.seq_len,
        embedding_dim=config.embedding_dim,
        num_lstm_units=config.num_lstm_units,
        num_lstm_layers=config.num_lstm_layers,
        dataset=dataset,
        device=device
    )

    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'checkpoint.pth'), map_location=device)
    rnn.load_state_dict(checkpoint['rnn'])
    print("# RNN weights restored.")

    return 0

if __name__ == '__main__':
    sys.exit(main())
