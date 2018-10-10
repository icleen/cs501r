from __future__ import unicode_literals, print_function, division
from io import open
import numpy as np
import matplotlib.pyplot as plt
import time
import re
import json
import sys
import os

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from model import RNN, GRUNet
from utils import *

# from tqdm import tqdm
# import pdb
# import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(inp, target):
    hidden = decoder.init_hidden()
    decoder_optimizer.zero_grad()
    loss = 0
    for c in range(chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c].unsqueeze(0))

    loss.backward()
    decoder_optimizer.step()
    return loss.item() / chunk_len

def evaluate(prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()

    prime_input = char_tensor(prime_str)
    # prime_input = onehot(prime_input)

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]

    predicted = ""
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        # print('output size: {}'.format(output.size()))

        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        inp = top_i

        # Pick the char with the highest probability
        # _, ind = output.view(-1).max(0)
        # inp = ind

        predicted += all_characters[inp]

    return predicted

n_epochs = 2000
print_every = 100
plot_every = 10
hidden_size = 100
n_layers = 1
lr = 0.005

# decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder = GRUNet(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    loss = train(*random_training_set())
    loss_avg += loss

    if epoch % print_every == 0:
        print('[%s (%d %d%%) %.4f]' % ((time.time() - start), epoch, epoch / n_epochs * 100, loss))
        print('predicted string:\n{}\n:The End'.format(evaluate('Wh', 100)))

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0
