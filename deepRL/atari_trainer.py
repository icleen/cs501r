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
from model import Policy2D
from loss import PPOLoss
from rollout_factory import RolloutFactory2D


class RLTrainer():
  """docstring for RLTrainer."""
  def __init__(self, config):
    with open(config, 'r') as f:
      config = json.load(f)

    self.epochs = config['train']['epochs']
    self.env_samples = config['train']['env_samples']
    self.episode_length = config['train']['episode_length']
    self.gamma = config['train']['gamma']
    self.policy_epochs = config['train']['policy_epochs']
    self.batch_size = config['train']['batch_size']
    self.policy_batch_size = config['train']['policy_batch_size']
    epsilon = config['train']['epsilon']

    self.env = gym.make(config['model']['gym'])

    state_size = config['model']['state_size']
    action_size = config['model']['action_size']
    hidden_size = config['model']['hidden_size']
    layer_size = config['model']['conv_layers']
    logheat = config['model']['logheat']
    self.action_size = action_size
    self.policy_net = Policy2D(state_size, action_size,
                              hidden_size=hidden_size, layers=layer_size,
                              logheat=logheat)

    self.value_loss = nn.MSELoss()
    self.ppoloss = PPOLoss(epsilon)

    betas = (config['train']['betas1'], config['train']['betas2'])
    weight_decay = config['train']['weight_decay']
    lr = config['train']['lr']
    self.optim = optim.Adam(self.policy_net.parameters(),
                          lr=lr, betas=betas, weight_decay=weight_decay)

    self.plosses = []
    self.vlosses = []
    self.avg_reward = []

    if torch.cuda.is_available():
      self.policy_net.cuda()
      self.device = torch.device("cuda")
      print("Using GPU")
    else:
      self.device = torch.device("cpu")
      print("No GPU detected")

    self.rollFact = RolloutFactory2D(self.env, config['model']['gym'],
                                self.policy_net, self.env_samples,
                                self.episode_length, self.gamma)

    self.write_interval = config['model']['write_interval']
    self.train_info_path = config['model']['trainer_save_path']
    self.policy_path = config['model']['policy_save_path'].split('.pt')[0]
    self.graph_path = config['model']['graph_save_path'].split('.png')[0]


  def train(self, itr=0):
    for i in range(self.epochs):
      # generate rollouts
      rollouts = self.rollFact.get_rollouts()

      # Learn a policy
      vlosses = []
      plosses = []
      dataset = RLDataset(rollouts)
      dataloader = DataLoader(dataset, batch_size=self.policy_batch_size,
                              shuffle=True, pin_memory=True)
      for _ in range(self.policy_epochs):
        # train policy network
        for state, aprob, action, reward, value in dataloader:
          state, aprob = state.to(self.device), aprob.to(self.device)
          action, value = action.to(self.device), value.to(self.device)

          pdist, pval = self.policy_net(state)
          clik = self.multinomial_likelihood(pdist, action)
          olik = self.multinomial_likelihood(aprob, action)
          ratio = (clik / olik)

          vloss = self.value_loss(pval, value)
          vlosses.append(vloss.cpu().item())

          advantage = value - pval.detach()
          ploss = self.ppoloss(ratio, advantage)
          plosses.append(ploss.cpu().item())

          pdist = None
          pval = None
          clik = None
          olik = None
          ratio = None
          advantage = None
          gc.collect()

          self.optim.zero_grad()
          loss = ploss + vloss
          loss.backward()
          self.optim.step()
      self.vlosses.append(np.mean(vlosses))
      self.plosses.append(np.mean(plosses))

      if (itr+i) % self.write_interval == 0:
        self.avg_reward = self.rollFact.avg_reward
        print('iter: {}, avg reward: {}, vloss: {}, ploss: {}, avg_len: {}'.format(
          itr+i, self.avg_reward[-1], vloss, ploss, len(rollouts[-1]) ))
        self.write_out(itr+i)

      # print(torch.cuda.memory_allocated(0) / 1e9)

  def multinomial_likelihood(self, dist, idx):
    return dist[range(dist.shape[0]), idx.long()[:, 0]].unsqueeze(1)

  def test(self):
    env = self.env
    state = env.reset()
    for _ in range(1000):
      env.render()
      probs = self.policy_net(state)
      probs_np = probs.cpu().detach().numpy()
      action_one_hot = np.random.multinomial(1, probs_np)
      action = np.argmax(action_one_hot)
      state, reward, done, info = env.step(action)

  def run(self, cont=False):
    # check to see if we should continue from an existing checkpoint
    # otherwise start from scratch
    if cont:
      itr = self.read_in()
      print('continuing')
      self.train(itr)
    else:
      self.train()

  def read_in(self, itr=None):
    train_info = {}
    train_info = torch.load(self.train_info_path)
    if itr is None:
      itr = train_info['iter']
    self.plosses = train_info['plosses']
    self.vlosses = train_info['vlosses']
    self.avg_reward = train_info['avg_reward']
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
    train_info['avg_reward'] = self.avg_reward
    train_info['optimizer'] = self.optim
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

      plt.plot(self.avg_reward[2:], label='rewards')
      plt.legend()
      plt.xlabel('epochs')
      plt.ylabel('rewards')
      plt.savefig(str(self.graph_path + '_reward.png'))
      plt.clf()


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
  trainer = RLTrainer(config)
  trainer.run(cont=cont)
