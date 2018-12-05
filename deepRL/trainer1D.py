import sys
import os
from os.path import join as opjoin
import json
import gc
import numpy as np
from queue import Queue
from threading import Thread

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.distributions import Categorical

from env_factory import *
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset import RLDataset, ExperienceDataset
from model import Policy1D, Value1D
from loss import PPOLoss
from rollout_factory import RolloutFactory


def multinomial_likelihood(dist, idx):
  return dist[range(dist.shape[0]), idx.long()[:, 0]].unsqueeze(1)


def _prepare_numpy(ndarray, device):
    return torch.from_numpy(ndarray).float().unsqueeze(0).to(device)


def _prepare_tensor_batch(tensor, device):
    return tensor.detach().float().to(device)


class Trainer1D():
  """docstring for Trainer1D."""
  def __init__(self, config):
    with open(config, 'r') as f:
      config = json.load(f)

    self.epochs = config['train']['epochs']
    self.value_epochs = config['train']['value_epochs']
    self.policy_epochs = config['train']['policy_epochs']
    self.policy_batch_size = config['train']['policy_batch_size']

    state_size = config['model']['state_size']
    action_size = config['model']['action_size']
    self.action_size = action_size
    self.policy_net = Policy1D(state_size, action_size)

    self.value_loss = nn.MSELoss()

    epsilon = config['train']['epsilon']
    self.ppoloss = PPOLoss(epsilon)
    self.ppo_low_bnd = 1 - epsilon
    self.ppo_up_bnd = 1 + epsilon

    betas = (config['train']['betas1'], config['train']['betas2'])
    weight_decay = config['train']['weight_decay']
    lr = config['train']['lr']
    # params = chain(self.policy_net.parameters(), self.value_net.parameters())
    self.optim = optim.Adam(self.policy_net.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    self.plosses = []
    self.vlosses = []
    self.avg_rewards = []
    self.stand_time = []

    if torch.cuda.is_available():
      self.policy_net.cuda()
      self.value_net.cuda()
      self.device = torch.device("cuda")
      print("Using GPU")
    else:
      self.device = torch.device("cpu")
      print("No GPU detected")


    env = gym.make(config['model']['gym'])
    env_samples = config['train']['env_samples']
    episode_length = config['train']['episode_length']
    gamma = config['train']['gamma']
    self.rollFact = RolloutFactory(env, config['model']['gym'],
                                self.policy_net, env_samples,
                                episode_length, gamma,
                                cutearly=config['train']['cutearly'])

    self.write_interval = config['model']['write_interval']
    self.train_info_path = config['model']['trainer_save_path']
    self.policy_path = config['model']['policy_save_path'].split('.pt')[0]
    self.value_path = config['model']['value_save_path'].split('.pt')[0]
    self.graph_path = config['model']['graph_save_path'].split('.png')[0]

  def train(self, itr=0):

    # loop = tqdm(total=self.epochs, position=0, leave=False)

    for i in range(self.epochs):
      avg_r = 0
      avg_policy_loss = 0
      avg_val_loss = 0

      rollouts = self.rollFact.get_rollouts()
      for r1 in rollouts:
        for r2 in r1:
          avg_r += r2[-2]
      avg_r /= len(rollouts)

      # Update the policy
      experience_dataset = ExperienceDataset(rollouts)
      data_loader = DataLoader(experience_dataset,
                              batch_size=self.policy_batch_size,
                              shuffle=True,
                              pin_memory=True)
      for _ in range(self.policy_epochs):
        avg_policy_loss = 0
        avg_val_loss = 0
        for state, aprob, action, reward, value in data_loader:
          state = _prepare_tensor_batch(state, self.device)
          aprob = _prepare_tensor_batch(aprob, self.device)
          action = _prepare_tensor_batch(action, self.device)
          value = _prepare_tensor_batch(value, self.device).unsqueeze(1)

          # Calculate the ratio term
          pdist, pval = self.policy_net(state, False)
          clik = multinomial_likelihood(pdist, action)
          olik = multinomial_likelihood(aprob, action)
          ratio = (clik / olik)

          # Calculate the value loss
          val_loss = self.value_loss(pval, value)

          # Calculate the policy loss
          advantage = value - pval.detach()
          lhs = ratio * advantage
          rhs = torch.clamp(ratio, self.ppo_low_bnd, self.ppo_up_bnd) * advantage
          policy_loss = -torch.mean(torch.min(lhs, rhs))

          # For logging
          avg_val_loss += val_loss.item()
          avg_policy_loss += policy_loss.item()

          # Backpropagate
          self.optim.zero_grad()
          loss = policy_loss + val_loss
          loss.backward()
          self.optim.step()

      # Log info
      avg_val_loss /= len(data_loader)
      avg_val_loss /= self.policy_epochs
      avg_policy_loss /= len(data_loader)
      avg_policy_loss /= self.policy_epochs
      self.vlosses.append(avg_val_loss)
      self.plosses.append(avg_policy_loss)
      self.avg_rewards.append(avg_r)

      # loop.set_description(
      #   'avg reward: % 6.2f, value loss: % 6.2f, policy loss: % 6.2f' \
      #   % (avg_r, avg_val_loss, avg_policy_loss))
      # loop.update(1)

      if (itr+i) % self.write_interval == 0:
        print(
          'avg reward: % 6.2f, value loss: % 6.2f, policy loss: % 6.2f' \
          % (avg_r, avg_val_loss, avg_policy_loss) )
        self.write_out(itr+i)


  def read_in(self, itr=None):
    train_info = {}
    train_info = torch.load(self.train_info_path)
    if itr is None:
      itr = train_info['iter']
    self.plosses = train_info['plosses']
    self.vlosses = train_info['vlosses']
    self.avg_rewards = train_info['avg_reward']
    self.optim = train_info['optimizer']

    self.policy_net.load_state_dict(torch.load(
      str(self.policy_path + '_' + str(itr) + '.pt') ))

    self.epochs += itr
    return itr

  def write_out(self, itr):
    train_info = {}
    train_info['iter'] = itr
    train_info['plosses'] = self.plosses
    train_info['vlosses'] = self.vlosses
    train_info['avg_reward'] = self.avg_rewards
    train_info['optimizer'] = self.optim
    # train_info['policy_optimizer'] = self.policy_optim
    # train_info['value_optimizer'] = self.value_optim
    torch.save( train_info, self.train_info_path )

    torch.save( self.policy_net.state_dict(),
      str(self.policy_path + '_' + str(itr) + '.pt') )

    if itr > 2:
      plt.plot(self.vlosses[2:], label='value loss')
      plt.plot(self.plosses[2:], label='policy loss')
      plt.legend()
      plt.xlabel('epochs')
      plt.ylabel('loss')
      plt.savefig(str(self.graph_path + '_loss.png'))
      plt.clf()

      plt.plot(self.avg_rewards[2:], label='rewards')
      plt.legend()
      plt.xlabel('epochs')
      plt.ylabel('rewards')
      plt.savefig(str(self.graph_path + '_reward.png'))
      plt.clf()


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
  trainer = Trainer1D(config)
  trainer.run(cont=cont)

if __name__ == '__main__':
  main()
