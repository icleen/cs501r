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
      config = json.load(f)

    self.iterations = config['train']['iterations']

    self.model = PneuNet(config['model']['img_shape'],
                        config['model']['classes'])

    self.objective = nn.CrossEntropyLoss()

    lr = config['train']['learning_rate']
    self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    self.losses = []
    self.vallosses = []

    if torch.cuda.is_available():
      self.model.cuda()
      self.dtype = torch.cuda.FloatTensor
      print("Using GPU")
    else:
      self.dtype = torch.FloatTensor
      print("No GPU detected")

    batch_size = config['train']['batch_size']
    trainset = PneuDataset(config['data']['train'],
                           config['data']['image_path'],
                           config['model']['img_shape'][1])
    self.trainloader = DataLoader(trainset,
                                  batch_size=batch_size,
                                  pin_memory=True)
    valset = PneuDataset(config['data']['valid'],
                         config['data']['image_path'],
                         config['model']['img_shape'][1])
    self.valloader = DataLoader(valset, batch_size=1, pin_memory=True)

    self.write_interval = config['model']['write_interval']
    self.train_info_path = config['model']['train_info_path']
    self.model_path = config['model']['model_save_path'].split('.pt')[0]
    self.graph_path = config['model']['graph_save_path'].split('.png')[0]

    # print(torch.cuda.memory_allocated(0) / 1e9)

  def train(self, itr):
    while itr < self.iterations:
      for j, (x, y) in enumerate(self.trainloader):
        if torch.cuda.is_available():
          x, y = x.cuda(async=True), y.cuda(async=True)

        preds = self.model(x)
        loss = self.objective(preds, y)
        self.losses.append(loss.cpu().item())

        self.optimizer.zero_grad()
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

        gc.collect()

  def validate(self):
    if torch.cuda.is_available():
      return [self.objective(self.model(x.cuda(async=True)),
              y.cuda(async=True)).cpu().item()
              for (x, y) in self.valloader]

    return [self.objective(self.model(x), y).cpu().item()
              for (x, y) in self.valloader]

  def read_in(self, itr=None):
    train_info = {}
    train_info = torch.load(self.train_info_path)
    if itr is None:
      itr = train_info['itr']
    self.losses = train_info['losses'][:itr]
    self.vallosses = train_info['valloss'][:itr]
    self.optimizer = train_info['optimizer'][:itr]

    self.model.load_state_dict(torch.load(
      str(self.model_path + '_' + str(itr) + '.pt') ))

    self.epochs += itr
    return itr

  def write_out(self, itr):
    train_info = {}
    train_info['iter'] = itr
    train_info['losses'] = self.losses
    train_info['valloss'] = self.vallosses
    train_info['optimizer'] = self.optimizer
    torch.save( train_info, self.train_info_path )

    torch.save( self.model.state_dict(),
      str(self.model_path + '_' + str(itr) + '.pt') )

  def run(self, cont=False):
    # check to see if we should continue from an existing checkpoint
    # otherwise start from scratch
    if cont:
      itr = self.read_in()
      print('continuing')
      self.train(itr)
    else:
      self.train()

def main():
  if len(sys.argv) < 2:
    print('Usage: ' + sys.argv[0] + ' config')
    exit(0)

  cont = False
  if len(sys.argv) > 2:
    info = sys.argv[2]
    if info == 'cont':
      cont = True

  config = sys.argv[1]
  trainer = Trainer(config)
  trainer.run(cont=cont)

if __name__ == '__main__':
  main()
