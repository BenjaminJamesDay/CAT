from __future__ import division

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
import glob
from torch.autograd import Variable

from utils import load_data, accuracy
from models import CCModel

runs = 50

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no_cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--cuda_device', type=str, default='3', help='CUDA device to use.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=time.time(), help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=50, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

if args.cuda:
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    return loss_val.item()


def compute_test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    return str(acc_test.item())


t = time.strftime("%Y-%m-%d-%H-%M")

for run in range(runs):
    # Create new model and optimizer
    model = CCModel(classes=int(labels.max()) + 1, ins=features.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    model.cuda()
    # Reset all model counters / monitors
    loss_values = []
    bad_counter = 0
    best = float("inf")
    best_epoch = 0
    
    saved_model = 'best_model' + t + '.pkl'
    # Train the model
    for epoch in range(args.epochs):
        loss_values.append(train(epoch))

        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), saved_model)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    # Restore the best from this set of models model
    model.load_state_dict(torch.load(saved_model))

    # Test the model & log result
    result = compute_test()

    f = open("results" + t + ".txt", "a+")
    f.write(result + "\n")
    f.close()
