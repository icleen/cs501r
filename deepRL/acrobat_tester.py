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


class RLTrainer():
  """docstring for RLTrainer."""
  def __init__(self, config):
    with open(config, 'r') as f:
      config = json.load(f)

    self.env = gym.make(config['model']['gym'])

    state_size = config['model']['state_size']
    action_size = config['model']['action_size']
    hidden_size = config['model']['hidden_size']
    layer_size = config['model']['hidden_layers']
    self.action_size = action_size
    self.policy_net = Policy1D(state_size, action_size,
                              hidden_size=hidden_size,
                              layers=layer_size)
    self.value_net = Value1D(state_size,
                            hidden_size=hidden_size,
                            layers=layer_size)

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
    self.gif_path = config['model']['gif_save_path'].split('.gif')[0]
    self.graph_path = config['model']['graph_save_path']

  def test(self):
    try:
      env = self.env
      state = env.reset()
      reward = 0
      for _ in range(1000):
        env.render()
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state).squeeze()
        probs_np = probs.cpu().detach().numpy()
        action_one_hot = np.random.multinomial(1, probs_np)
        action = np.argmax(action_one_hot)
        state, rew, _, _ = env.step(action)
        reward += rew
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

    self.policy_net.load_state_dict(torch.load(
      str(self.policy_path + '_' + str(itr) + '.pt') ))

    self.value_net.load_state_dict(torch.load(
      str(self.value_path + '_' + str(itr) + '.pt') ))

    print('Loaded: {}, {}'.format(
      str(self.policy_path + '_' + str(itr) + '.pt'),
      str(self.value_path + '_' + str(itr) + '.pt') ))


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
  trainer = RLTrainer(config)
  trainer.run(itr=itr)
