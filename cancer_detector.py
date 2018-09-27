import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch.nn.parameter import Parameter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
import json
import sys
import gc
import os

from cancer_dataset import CancerDataset
from cancer_model import UNet


def main():

  config_path = 'config.json'
  if len(sys.argv) > 1:
    config_path = sys.argv[1]
  with open(config_path) as f:
      config = json.load(f)

  trainset = CancerDataset(config['training_set_path'], download=False)
  train_loader = DataLoader(trainset, batch_size=5, shuffle=True, pin_memory=True)

  valset = CancerDataset(config['validation_set_path'], download=False, train=False)
  val_loader = DataLoader(valset, batch_size=1, pin_memory=True)

  model = UNet(config['network']['input_size'], config['network']['output_size'])
  model.cuda()

  objective = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=config['network']['learning_rate'])

  epochs = 100
  losses = []
  valosses = []
  doloop = config['looper']
  lowest_loss = float('inf')
  gc.collect()

  for epoch in range(epochs):

    if doloop:
      loop = tqdm(total=len(train_loader), position=0)

    for i, (x, y) in enumerate(train_loader):
      x, y = x.cuda(async=True), y.cuda(async=True)

      optimizer.zero_grad()

      preds = model(x)
      loss = objective(preds, y)
      losses.append(loss.item())

      loss.backward()
      optimizer.step()
      preds = None
      loss = None

      if doloop:
        loop.set_description(
          'epoch: {}, iter: {}, loss: {:.6f}, memory: {}'.format(
            epoch+1, i, losses[-1], torch.cuda.memory_allocated(0) / 1e9 ) )
        loop.update(1)
      elif i % 30 == 0:
        print('epoch: {}, iter: {}, loss: {:.6f}, memory: {}'.format(
          epoch+1, i, losses[-1], torch.cuda.memory_allocated(0) / 1e9 ) )

      if i % 30 == 0:
        valosses.append( ( len(losses),
          np.mean([objective(model(x.cuda()), y.cuda()).item() for x, y in val_loader]) ) )

        if valosses[-1][-1] < lowest_loss:
          lowest_loss = valosses[-1][-1]
          print("Saving Best")
          dirname = os.path.dirname(config['model_save_path'])
          if len(dirname) > 0 and not os.path.exists(dirname):
            os.makedirs(dirname)
          torch.save(model.state_dict(), os.path.join(config['model_save_path']))

      gc.collect()

    if doloop:
      loop.close()


  a, b = zip(*valosses)
  plt.plot(losses, label='train')
  plt.plot(a, b, label='avg val per 30')
  plt.legend()
  plt.xlabel('iterations')
  plt.ylabel('loss')
  plt.savefig('cancer_fig.png')


if __name__ == '__main__':
  main()
