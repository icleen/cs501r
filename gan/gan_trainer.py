import sys
import os
from os.path import join as opjoin

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import json
import gc
import numpy as np

from celeb_dataset import CelebaDataset
from generator import Generator
from descriminator import Descriminator
from loss import DescriminatorLoss, GeneratorLoss


def is_connected(y,x):
    def recur_is_connected(fn,x):
        if hasattr(fn,'variable'):
            if fn.variable is x:
                return True
        if hasattr(fn,'next_functions'):
            for f in fn.next_functions:
                if recur_is_connected(f[0],x):
                    return True
        return False
    return recur_is_connected(y.grad_fn,x)


class GANTrainer():
  """docstring for Trainer."""
  def __init__(self, config):
    with open(config, 'r') as f:
      config = json.load(f)
    self.config = config

    self.iterations = config['train']['iterations']
    self.critic_iters = config['train']['critic_iters']
    self.batch_size = config['train']['batch_size']
    self.lr = config['train']['learning_rate']
    img_size = config['model']['image_size']

    trainset = CelebaDataset(config['data']['image_path'], size=img_size)
    self.trainloader = DataLoader(trainset, batch_size=self.batch_size, pin_memory=True)

    self.z_size = config['train']['z_size']
    self.generator = Generator(self.z_size)
    self.descriminator = Descriminator()

    lam = config['train']['lambda']
    self.g_loss = GeneratorLoss()
    self.d_loss = DescriminatorLoss(lam=lam)

    betas = (config['train']['beta1'], config['train']['beta2'])
    self.g_optim = optim.SGD(self.generator.parameters(), lr=self.lr, betas=betas)
    self.d_optim = optim.SGD(self.descriminator.parameters(), lr=self.lr, betas=betas)

    self.dlosses = []
    self.glosses = []

    if torch.cuda.is_available():
      self.generator.cuda()
      self.descriminator.cuda()
      self.device = torch.device("cuda")
      print("Using GPU")
    else:
      self.device = torch.device("cpu")
      print("No GPU detected")

    self.write_interval = config['model']['write_interval']
    self.train_info_path = self.config['model']['trainer_save_path']
    self.generator_path = self.config['model']['generator_save_path'].split('.pt')[0]
    self.descriminator_path = self.config['model']['descriminator_save_path'].split('.pt')[0]
    self.img_path = self.config['model']['image_save_path'].split('.png')[0]


  def train(self, itr=0):
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

          gen_img = self.generate_img()

          hat_img = eps*true_img + (1-eps)*gen_img
          hat_img.requires_grad_()

          # calculate disc loss: you will need autograd.grad
          predg = self.descriminator(gen_img)
          predt = self.descriminator(true_img)
          predh = self.descriminator(hat_img)

          dloss = self.d_loss(predg, predt, predh, hat_img)
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

        gen_img = self.generate_img()
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

        if itr % self.write_interval == 0:
          print('iter: {}, dloss: {}, gloss: {}'.format( itr, dloss, gloss ))
          self.write_out(itr)

        itr += 1
        # print(torch.cuda.memory_allocated(0) / 1e9)

  def run(self, cont=False):
    # check to see if we should continue from an existing checkpoint
    # otherwise start from scratch
    if cont:
      itr = self.read_in()
      self.train(itr)
    else:
      self.train()

  def random_z(self):
    z = torch.zeros((self.batch_size, self.z_size), dtype=torch.float, device=self.device)
    z.normal_() # generate noise tensor z
    return z

  def generate_img(self):
    z = self.random_z() # generate noise tensor z
    return self.generator(z)

  def read_in(self, itr=None):
    train_info = {}
    train_info = torch.load(self.train_info_path)
    if itr is None:
      itr = train_info['iter']
    self.dlosses = train_info['dlosses']
    self.glosses = train_info['glosses']
    self.g_optim = train_info['g_optimizer']
    self.d_optim = train_info['d_optimizer']

    self.generator.load_state_dict(torch.load(
      str(self.generator_path + '_' + str(itr) + '.pt')))

    self.descriminator.load_state_dict(torch.load(
      str(self.descriminator_path + '_' + str(itr) + '.pt')))

    self.iterations += itr
    return itr

  def write_out(self, itr):
    train_info = {}
    train_info['iter'] = itr
    train_info['dlosses'] = self.dlosses
    train_info['glosses'] = self.glosses
    train_info['g_optimizer'] = self.g_optim
    train_info['d_optimizer'] = self.d_optim
    torch.save( train_info, self.train_info_path )

    torch.save( self.generator.state_dict(),
      str(self.generator_path + '_' + str(itr) + '.pt') )

    torch.save( self.descriminator.state_dict(),
      str(self.descriminator_path + '_' + str(itr) + '.pt') )

    gen_img = self.generate_img()
    gen_img = gen_img[0]
    save_image(gen_img, str(self.img_path + '_' + str(itr) + '.png'))


if __name__ == '__main__':

  if len(sys.argv) < 2:
    print('Usage: ' + sys.argv[0] + ' config')
    exit(0)

  cont = False
  if len(sys.argv) > 2:
    info = sys.argv[2]
    if info == 'cont':
      cont = True

  config = sys.argv[1]
  trainer = GANTrainer(config)
  trainer.run(cont=cont)
