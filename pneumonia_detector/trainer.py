import sys
import os
from os.path import join as opjoin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import json

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

    trainset = PneuDataset(self.config['data']['train'])
    self.trainloader = DataLoader(trainset, batch_size=10, pin_memory=True)
    valset = PneuDataset(self.config['data']['valid'])
    self.valloader = DataLoader(valset, batch_size=10, pin_memory=True)

    self.model = PneuNet(self.config['model']['img_shape'])
    if torch.cuda.is_available():
      self.net.cuda()
      self.dtype = torch.cuda.FloatTensor
      print("Using GPU")
    else:
      self.dtype = torch.FloatTensor
      print("No GPU detected")

    self.objective = nn.MSELoss()
    self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    self.losses = []

  def train(self, itr):
    while itr < self.iterations:
      for j, (x, y) in enumerate(self.trainloader):
        if torch.cuda.is_available():
          x, y = x.cuda(async=True), y.cuda(async=True)

        self.optimizer.zero_grad()

        preds = self.model(x)

        loss = self.objective(preds, y)
        self.losses.append(loss.item())
        loss.backward()

        self.optimizer.step()

        gc.collect()

        itr += 1
        if itr % self.write_interval:
          valloss = np.mean(self.validate())
          print( 'iter: {}, valloss: {}, trainloss: {}'.format( i, valloss,
            np.mean(self.losses[-self.write_interval:]) ) )
          self.write_out(itr, valloss)

  def validate(self):
    return [self.objective(self.model(x), y).item() for (x, y) in self.valloader]

  def write_out(self, itr, valloss):
    train_info = {}
    train_info['iter'] = itr
    train_info['losses'] = self.losses
    train_info['valloss'] = valloss
    train_info['optimizer'] = self.optimizer
    torch.save( train_info, opjoin(self.config['trainer_save_path']) )

    torch.save( self.model.state_dict(),
      str(self.config['model_save_path'].split('.pt')[0] + '_' + itr + '.pt') )

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
