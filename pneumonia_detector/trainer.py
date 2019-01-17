import sys
import os
from os.path import join as opjoin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

import json
import gc
import random
import numpy as np
import matplotlib.pyplot as plt

from utils.dataset import PneuDataset, PneuYoloDataset
from yolo_model import PneuYoloNet
from loss import YoloLoss

from models import *
from utils.utils import *
from utils.parse_config import *


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

    # self.model = PneuYoloNet(config['model']['img_shape'])
    # self.objective = YoloLoss(config['model']['img_shape'][1])

    self.model = Darknet(config['model']['yolo_path'],
                        img_size=config['model']['img_shape'][1])
    self.model.apply(weights_init_normal)

    lr = config['train']['learning_rate']
    self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)

    self.losses = []
    self.vallosses = []

    self.dtype = torch.FloatTensor
    if torch.cuda.is_available():
      torch.cuda.set_device(int(dev))
      self.model.cuda()
      self.dtype = torch.cuda.FloatTensor
      print("Using GPU")
    else:
      print("No GPU detected")

    batch_size = config['train']['batch_size']
    trainset = PneuYoloDataset(config['data']['train'],
                           config['data']['image_path'],
                           config['model']['img_shape'][1])
    self.trainloader = DataLoader(trainset,
                                  batch_size=batch_size,
                                  pin_memory=True)
    self.valset = PneuYoloDataset(config['data']['valid'],
                         config['data']['image_path'],
                         config['model']['img_shape'][1])
    self.valloader = DataLoader(self.valset, batch_size=batch_size, pin_memory=True)

    self.write_interval = config['model']['write_interval']
    self.train_info_path = config['model']['train_info_path']
    self.model_path = config['model']['model_save_path'].split('.pt')[0]
    self.graph_path = config['model']['graph_save_path'].split('.png')[0]

    # print(torch.cuda.memory_allocated(0) / 1e9)

  def train(self, itr):
    self.model.train()
    while itr < self.iterations:
      for j, (_, imgs, targets) in enumerate(self.trainloader):
        # if torch.cuda.is_available():
        #   img = img.cuda(async=True)
        #   targets = targets.cuda(async=True)
          # x, y = x.cuda(async=True), y.cuda(async=True)
          # w, h, label = w.cuda(async=True), h.cuda(async=True), label.cuda(async=True)
        imgs = Variable(imgs.type(self.dtype))
        targets = Variable(targets.type(self.dtype), requires_grad=False)
        #print(itr)
        # import pdb; pdb.set_trace()
        # preds = self.model(img)
        # loss = self.objective(preds, targets)
        loss = self.model(imgs, targets)
        self.losses.append(loss.cpu().item())
        #print(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        preds = None
        gc.collect()

        itr += 1
        if itr % self.write_interval == 0:
          self.model.eval()
          print('validating...')
          valloss = np.mean(self.validate())
          print( 'iter: {}, trainloss: {}, valloss: {}'.format( itr,
            np.mean(self.losses[-self.write_interval:]),
            valloss ) )
          self.vallosses.append(valloss)
          self.write_out(itr)
          self.model.train()

        gc.collect()

  def validate(self):
    if torch.cuda.is_available():
      return [self.objective(self.model(img.cuda(async=True)), (x, y, w, h, label)).item()
              for (img, x, y, w, h, label) in self.valloader]
    return [self.objective(self.model(img), (x, y, w, h, label)).item()
              for (img, x, y, w, h, label) in self.valloader]

  def sample_img(self):
    if torch.cuda.is_available():
      i = random.uniform(0,len(self.valset)-1)
      img, x, y, w, h, label = self.valset[i]
      inp = img.cuda(async=True).unsqueeze(0)
      img = self.valset.get_image(i)
      preds = self.model(inp)
      pc, px, py, pw, ph = preds
      pc, px, py, pw, ph = pc.cpu(), px.cpu(), py.cpu(), pw.cpu(), ph.cpu()

      x, y, w, h = x.item(), y.item(), w.item(), h.item()
      xl = x - (w/2)
      yl = y - (h/2)
      xr = x + (w/2)
      yr = y + (h/2)

      torch.argmax(pc)
      pxl =

      draw = ImageDraw.Draw(img)
      draw.rectangle(xl, yl, xr, yr)
      draw.rectangle(pxl, pyl, pxr, pyr)


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
