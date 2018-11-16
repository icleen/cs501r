import sys
import os
from os.path import join as opjoin
import json
import gc
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.distributions import Categorical

import gym

from dataset import RLDataset
from model import Policy1D, Value1D


class RLTrainer():
  """docstring for RLTrainer."""
  def __init__(self, config):
    with open(config, 'r') as f:
      config = json.load(f)

    self.episode_length = config['train']['episode_length']

    self.env = gym.make(config['model']['gym'])

    state_size = config['model']['state_size']
    action_size = config['model']['action_size']
    self.policy_net = Policy1D(state_size, action_size)
    self.value_net = Value1D(state_size)

    self.plosses = []
    self.vlosses = []
    self.stand_time = []

    if torch.cuda.is_available():
      self.policy_net.cuda()
      self.value_net.cuda()
      self.device = torch.device("cuda")
      print("Using GPU")
    else:
      self.device = torch.device("cpu")
      print("No GPU detected")

    self.write_interval = config['model']['write_interval']
    self.train_info_path = config['model']['trainer_save_path']
    self.policy_path = config['model']['policy_save_path'].split('.pt')[0]
    self.value_path = config['model']['value_save_path'].split('.pt')[0]

  def test(self):
    env = self.env
    state = env.reset()
    # env.render()
    it = 0
    value = 0.0
    done = False
    while not done and it < self.episode_length:
      state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
      dist = self.policy_net(state)
      action = dist.sample().squeeze().cpu().numpy()
      state, reward, done, _ = env.step(action)
      value += reward
      # env.render()
      it += 1
    print('Reward: {}, steps: {}'.format(value, it))
    return value

  def visuals(self, itr):
    print('iter: {}, avg stand time: {:.2f}, vloss: {:.3f}, ploss: {:.3f}'.format(
      itr, self.stand_time[-1], self.vlosses[-1], self.plosses[-1] ))

    plt.plot(self.vlosses[2:], label='value loss')
    plt.plot(self.plosses[2:], label='policy loss')
    plt.plot(self.stand_time[2:], label='stand time')
    # plt.scatter(trainitrs[np.argmin(trainloss)], np.min(trainloss), label='lowest loss')
    # plt.scatter(valitrs[np.argmax(valloss)], np.max(valloss), label='highest IOU')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('loss_figure.png')

  def run(self):
    itr = self.read_in()
    self.test()
    self.visuals(itr)

  def read_in(self, itr=None):
    train_info = {}
    train_info = torch.load(self.train_info_path)
    if itr is None:
      itr = train_info['iter']
    self.plosses = train_info['plosses']
    self.vlosses = train_info['vlosses']
    self.stand_time = train_info['stand_time']
    self.policy_optim = train_info['policy_optimizer']
    self.value_optim = train_info['value_optimizer']

    self.policy_net.load_state_dict(torch.load(
      str(self.policy_path + '_' + str(itr) + '.pt') ))

    self.value_net.load_state_dict(torch.load(
      str(self.value_path + '_' + str(itr) + '.pt') ))

    return itr


if __name__ == '__main__':

  if len(sys.argv) < 2:
    print('Usage: ' + sys.argv[0] + ' config')
    exit(0)

  config = sys.argv[1]
  trainer = RLTrainer(config)
  trainer.run()
