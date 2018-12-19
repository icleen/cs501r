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
import matplotlib.pyplot as plt

from utils.dataset import PneuDataset
from yolo_model import PneuYoloNet
from loss import YoloLoss


class Trainer(object):
  """docstring for Trainer."""
  def __init__(self, config, dev='0'):
    super(Trainer, self).__init__()
    with open(config, 'r') as f:
      config = json.load(f)

    self.iterations = config['train']['iterations']

    # self.model = PneuNet(config['model']['img_shape'],
    #                     config['model']['classes'])
    # self.objective = nn.CrossEntropyLoss()

    self.model = PneuYoloNet(config['model']['img_shape'])
    self.objective = YoloLoss(config['model']['img_shape'][1])

    lr = config['train']['learning_rate']
    self.optimizer = optim.SGD(self.model.parameters(), lr=lr)

    self.losses = []
    self.vallosses = []

    if torch.cuda.is_available():
      torch.cuda.set_device(int(dev))
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
    self.valloader = DataLoader(valset, batch_size=batch_size, pin_memory=True)

    self.write_interval = config['model']['write_interval']
    self.train_info_path = config['model']['train_info_path']
    self.model_path = config['model']['model_save_path'].split('.pt')[0]
    self.graph_path = config['model']['graph_save_path'].split('.png')[0]

    # print(torch.cuda.memory_allocated(0) / 1e9)

  def train(self, itr):
    while itr < self.iterations:
      for j, (img, x, y, w, h, label) in enumerate(self.trainloader):
        if torch.cuda.is_available():
          img = img.cuda(async=True)
          # x, y = x.cuda(async=True), y.cuda(async=True)
          # w, h, label = w.cuda(async=True), h.cuda(async=True), label.cuda(async=True)

        print(itr)
        preds = self.model(img)
        loss = self.objective(preds, (x, y, w, h, label))
        self.losses.append(loss.cpu().item())
        print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        preds = None
        gc.collect()

        itr += 1
        if itr % self.write_interval == 0:
          print('validating...')
          valloss = np.mean(self.validate())
          print( 'iter: {}, trainloss: {}, valloss: {}'.format( itr,
            np.mean(self.losses[-self.write_interval:]),
            valloss ) )
          self.vallosses.append(valloss)
          self.write_out(itr)

        gc.collect()

  def validate(self):
    if torch.cuda.is_available():
      return [self.objective(self.model(img.cuda(async=True)), (x, y, w, h, label)).item()
              for (img, x, y, w, h, label) in self.valloader]
 #     return [self.objective(
 #            self.model(img.cuda(async=True)),
#(x.cuda(async=True), y.cuda(async=True), w.cuda(async=True), h.cuda(async=True), label.cuda(async=True))
 #           ).cpu().item()
 #             for (img, x, y, w, h, label) in self.valloader]
    

    return [self.objective(self.model(img), (x, y, w, h, label)).item()
              for (img, x, y, w, h, label) in self.valloader]

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
      
    plt.plot(self.losses, label='train loss')
    plt.plot(self.vallosses, label='validation loss')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig(str(self.graph_path + '_loss.png'))
    plt.clf()


  def run(self, cont=False):
    # check to see if we should continue from an existing checkpoint
    # otherwise start from scratch
    if cont:
      itr = self.read_in()
      print('continuing')
      self.train(itr)
    else:
      self.train(0)

def main():
  if len(sys.argv) < 2:
    print('Usage: ' + sys.argv[0] + ' config')
    exit(0)

  cont = False
  dev = '0'
  if len(sys.argv) > 2:
    info = sys.argv[2]
    if info == 'cont':
      cont = True
    elif info == '1':
      dev = info
    elif info == '2':
      dev = info
    elif info == '3':
      dev = info
  elif len(sys.argv) > 3:
    dev = sys.argv[3]

  config = sys.argv[1]
  trainer = Trainer(config, dev=dev)
  trainer.run(cont=cont)

if __name__ == '__main__':
  main()
