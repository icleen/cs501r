import sys
import os
from os.path import join as opjoin
import json
import gc
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.distributions import Categorical

import gym
import imageio
from itertools import chain
import matplotlib.pyplot as plt

from dataset import RLDataset
from model import Policy1D, Value1D
from loss import PPOLoss


class RLTester():
  """docstring for RLTester."""
  def __init__(self, config):
    with open(config, 'r') as f:
      config = json.load(f)

    self.env = gym.make(config['model']['gym'])
    self.mtcar = (config['model']['gym'] == 'MountainCar-v0')

    state_size = config['model']['state_size']
    action_size = config['model']['action_size']
    hidden_size = config['model']['hidden_size']
    layer_size = config['model']['hidden_layers']
    logheat = config['model']['logheat']
    self.policy_net = Policy1D(state_size, action_size,
                                hidden_size=hidden_size,
                                layers=layer_size,
                                logheat=logheat)

    if torch.cuda.is_available():
      self.policy_net.cuda()
      self.value_net.cuda()
      self.device = torch.device("cuda")
      print("Using GPU")
    else:
      self.device = torch.device("cpu")
      print("No GPU detected")

    self.visual = True
    self.cut = config['train']['cutearly']
    self.write_interval = config['model']['write_interval']
    self.train_info_path = config['model']['trainer_save_path']
    self.policy_path = config['model']['policy_save_path'].split('.pt')[0]

  def test(self):
    try:
      env = self.env
      state = env.reset()
      reward = 0
      for i in range(1000):
        if self.visual:
          env.render()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        adist, action = self.policy_net(state)
        adist, action = adist[0], action[0]
        state, rew, done, _ = env.step(action.item())
        reward += rew
        if i % 50 == 0:
          print(state)
          print(adist)
        if self.mtcar:
          reward = state[0] + 0.5
          if state[0] >= 0.5:
            reward += 1
        if self.cut and done:
          break
      print('Final reward: {}'.format(reward))
    except Exception as e:
      return

  def run(self, itr=None):
    self.read_in(itr)
    self.test()

  def read_in(self, itr=None):
    train_info = {}
    train_info = torch.load(self.train_info_path)
    if itr is None:
      itr = train_info['iter']

    path = str(self.policy_path + '_' + str(itr) + '.pt')
    self.policy_net.load_state_dict(torch.load( path ))
    print('Loaded:', path)


if __name__ == '__main__':

  if len(sys.argv) < 2:
    print('Usage: ' + sys.argv[0] + ' config')
    exit(0)

  itr = None
  if len(sys.argv) > 2:
    itr = int(sys.argv[2])
    # if info == 'cont':
    #   cont = True

  config = sys.argv[1]
  tester = RLTester(config)
  tester.run(itr=itr)
