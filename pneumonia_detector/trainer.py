import sys
import os
from os.path import join as opjoin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import json
import gc
import numpy as np

from utils.dataset import PneuDataset
from model import PneuNet


class Trainer(object):
  """docstring for Trainer."""
  def __init__(self, config):
    super(Trainer, self).__init__()
    with open(config, 'r') as f:
      self.config = json.load(f)

    self.iterations = self.config['train']['iterations']
    self.lr = self.config['train']['learning_rate']
    self.write_interval = self.config['train']['write_interval']

    trainset = PneuDataset(self.config['data']['train'], self.config['data']['image_path'])
    self.trainloader = DataLoader(trainset, batch_size=1, pin_memory=True)
    valset = PneuDataset(self.config['data']['valid'], self.config['data']['image_path'])
    self.valloader = DataLoader(valset, batch_size=1, pin_memory=True)

    self.model = PneuNet(self.config['model']['img_shape'], self.config['model']['classes'])
    if torch.cuda.is_available():
      self.model.cuda()
      self.dtype = torch.cuda.FloatTensor
      print("Using GPU")
    else:
      self.dtype = torch.FloatTensor
      print("No GPU detected")

    self.objective = nn.CrossEntropyLoss()
    self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    self.losses = []
    self.vallosses = []

    # print(torch.cuda.memory_allocated(0) / 1e9)

  def train(self, itr):
    # interval = len(self.trainloader) / self.write_interval
    while itr < self.iterations:
      for j, (x, y) in enumerate(self.trainloader):
        if torch.cuda.is_available():
          x, y = x.cuda(async=True), y.cuda(async=True)

        self.optimizer.zero_grad()

        preds = self.model(x)

        loss = self.objective(preds, y)
        self.losses.append(loss.cpu().item())
        loss.backward()

        self.optimizer.step()

        preds = None
        gc.collect()

        itr += 1
        if itr % self.write_interval == 0:
          valloss = np.mean(self.validate())
          print( 'iter: {}, trainloss: {}, valloss: {}'.format( itr,
            np.mean(self.losses[-self.write_interval:]),
            valloss ) )
          self.vallosses.append(valloss)
          self.write_out(itr)

        # print(torch.cuda.memory_allocated(0) / 1e9)

  def validate(self):
    if torch.cuda.is_available():
      return [self.objective(self.model(x.cuda(async=True)), y.cuda(async=True)).cpu().item() for (x, y) in self.valloader]
    return [self.objective(self.model(x), y).cpu().item() for (x, y) in self.valloader]

  def write_out(self, itr):
    train_info = {}
    train_info['iter'] = itr
    train_info['losses'] = self.losses
    train_info['valloss'] = self.vallosses
    train_info['optimizer'] = self.optimizer
    torch.save( train_info, opjoin(self.config['model']['trainer_save_path']) )

    torch.save( self.model.state_dict(),
      str(self.config['model']['model_save_path'].split('.pt')[0] + '_' + str(itr) + '.pt') )

  def run(self, cont=False):
    # check to see if we should continue from an existing checkpoint
    # otherwise start from scratch
    if cont:
      self.train(100)
    else:
      self.train(0)


if __name__ == '__main__':

  if len(sys.argv) < 2:
    print('Usage: ' + sys.argv[0] + ' config')
    exit(0)

  config = sys.argv[1]
  trainer = Trainer(config)

  trainer.run()
