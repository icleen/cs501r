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

from celeb_dataset import CelebaDataset
from generator import Generator
from descriminator import Descriminator
from trainer import Trainer
from loss import DescriminatorLoss, GeneratorLoss


class GANTrainer(Trainer):
  """docstring for Trainer."""
  def __init__(self, config):
    super(Trainer, self).__init__()
    with open(config, 'r') as f:
      config = json.load(f)
    config = config

    self.iterations = config['train']['iterations']
    self.critic_iters = config['train']['critic_iters']
    self.batch_size = config['train']['batch_size']
    self.lr = config['train']['learning_rate']
    self.write_interval = config['train']['write_interval']

    trainset = CelebaDataset()
    self.trainloader = DataLoader(trainset, batch_size=self.batch_size, pin_memory=True)

    self.z_size = config['model']['z_size']
    self.generator = Generator(self.z_size)
    self.descriminator = Descriminator()

    lam = config['model']['lambda']
    self.g_loss = GeneratorLoss(lam=lam)
    self.d_loss = DescriminatorLoss(lam=lam)

    self.g_optim = optim.SGD(self.generator.parameters(), lr=self.lr)
    self.d_optim = optim.SGD(self.descriminator.parameters(), lr=self.lr)

    self.dlosses = []
    self.glosses = []

    if torch.cuda.is_available():
      self.model.cuda()
      self.dtype = torch.cuda.FloatTensor
      print("Using GPU")
    else:
      self.dtype = torch.FloatTensor
      print("No GPU detected")

  def train(self, itr):
    # interval = len(self.trainloader) / self.write_interval
    while itr < self.iterations:
      for j, true_img in enumerate(self.trainloader):
        if torch.cuda.is_available():
          true_img = true_img.cuda(async=True)

        """train discriminator"""
        #because you want to be able to backprop through the params in discriminator
        for p in self.descriminator.parameters():
          p.requires_grad = True

        for p in self.generator.parameters():
          p.requires_grad = False

        for n in range(self.critic_iters):
          self.d_optim.zero_grad()
          eps = np.random.uniform()

          z = self.random_z() # generate noise tensor z
          gen_img = self.generator(z)

          hat_img = eps*true_img + (1-eps)*gen_img

          # calculate disc loss: you will need autograd.grad
          predg = self.descriminator(gen_img)
          predt = self.descriminator(true_img)
          predh = self.descriminator(hat_img)

          dloss = self.d_loss(predg, predt, predh)
          dloss.backward()
          self.d_optim.step()

          predg = None
          predt = None
          predh = None

        """train generator"""
        for p in self.descriminator.parameters():
          p.requires_grad = False

        for p in self.generator.parameters():
          p.requires_grad = True

        self.g_optim.zero_grad()

        z = self.random_z() # generate noise tensor z
        gen_img = self.generator(z)
        pred = self.descriminator(gen_img)
        # calculate loss for gen
        gloss = self.g_loss(pred)
        # call gloss.backward() and gen_optim.step()
        gloss.backward()
        self.g_optim.step()

        pred = None
        gc.collect()

        self.dlosses.append(dloss.cpu().item())
        self.glosses.append(gloss.cpu().item())

        itr += 1
        if itr % self.write_interval == 0:
          print( 'iter: {}, dloss: {}, gloss: {}'.format( itr, dloss, gloss ) )
          self.write_out(itr)

        # print(torch.cuda.memory_allocated(0) / 1e9)

  def run(self, cont=False):
    # check to see if we should continue from an existing checkpoint
    # otherwise start from scratch
    if cont:
      self.train(100)
    else:
      self.train(0)

  def random_z():
    z = torch.zeros((self.batch_size, self.z_size), dtype=torch.float)
    z.normal_() # generate noise tensor z
    if torch.cuda.is_available():
      z.cuda(async=True)
    return z

  def write_out(self, itr):
    train_info = {}
    train_info['iter'] = itr
    train_info['dlosses'] = self.dlosses
    train_info['glosses'] = self.glosses
    train_info['optimizer'] = self.optimizer
    torch.save( train_info, opjoin(self.config['model']['trainer_save_path']) )

    torch.save( self.model.state_dict(),
      str(self.config['model']['model_save_path'].split('.pt')[0] + '_' + str(itr) + '.pt') )



if __name__ == '__main__':

  if len(sys.argv) < 2:
    print('Usage: ' + sys.argv[0] + ' config')
    exit(0)

  config = sys.argv[1]
  trainer = Trainer(config)
  trainer.run()
