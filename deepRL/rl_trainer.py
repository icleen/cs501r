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

from dataset import RLDataset
from model import Policy1D, Value1D
from loss import PPOLoss


class RLTrainer():
  """docstring for RLTrainer."""
  def __init__(self, config):
    with open(config, 'r') as f:
      config = json.load(f)

    self.epochs = config['train']['epochs']
    self.env_samples = config['train']['env_samples']
    self.episode_length = config['train']['episode_length']
    self.gamma = config['train']['gamma']
    self.value_epochs = config['train']['value_epochs']
    self.policy_epochs = config['train']['policy_epochs']
    self.batch_size = config['train']['batch_size']
    self.policy_batch_size = config['train']['policy_batch_size']
    epsilon = config['train']['epsilon']

    self.env = gym.make(config['model']['gym'])

    state_size = config['model']['state_size']
    action_size = config['model']['action_size']
    self.action_size = action_size
    self.policy_net = Policy1D(state_size, action_size)
    self.value_net = Value1D(state_size)

    self.value_loss = nn.MSELoss()
    self.ppoloss = PPOLoss(epsilon)

    policy_lr = config['train']['policy_lr']
    value_lr = config['train']['value_lr']
    policy_decay = config['train']['policy_decay']
    value_decay = config['train']['value_decay']
    self.policy_optim = optim.Adam(self.policy_net.parameters(), lr=policy_lr, weight_decay=policy_decay)
    self.value_optim = optim.Adam(self.value_net.parameters(), lr=value_lr, weight_decay=value_decay)

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


  def train(self, itr=0):
    for i in range(self.epochs):
      # generate rollouts
      rollouts = self.get_rollouts()

      # Approximate the value function
      vloss = []
      dataset = RLDataset(rollouts)
      dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)
      for _ in range(self.value_epochs):
        for it in dataloader:
          state, _, _, value = it
          state, value = state.to(self.device), value.to(self.device)
          pval = self.value_net(state)
          loss = self.value_loss(pval, value)
          vloss.append(loss.cpu().item())

          self.value_optim.zero_grad()
          loss.backward()
          self.value_optim.step()
          gc.collect()

          # input('{}, {}, {}'.format(pval, value, loss.cpu().item()))
      vloss = np.mean(vloss)
      self.vlosses.append(vloss)


      rollouts = self.calculate_advantages(rollouts)
      # Learn a policy
      ploss = []
      dataset = RLDataset(rollouts)
      dataloader = DataLoader(dataset, batch_size=self.policy_batch_size, shuffle=True, pin_memory=True)
      # dataloader = DataLoader(dataset, batch_size=5, shuffle=True, pin_memory=True)
      for _ in range(self.policy_epochs):
        # train policy network
        for it in dataloader:
          state, action, aprob, advantage = it
          state, action = state.to(self.device), action.to(self.device)
          aprob, advantage = aprob.to(self.device), advantage.to(self.device)
          pdist = self.policy_net(state)
          # ratio = pdist.log_prob(action)/aprob
          # ratio = (pdist.log_prob(action) - aprob).exp()
          ratio = pdist[torch.arange(action.size(0), dtype=torch.long), action]/aprob
          loss = self.ppoloss(ratio, advantage)
          ploss.append(loss.cpu().item())

          self.policy_optim.zero_grad()
          loss.backward()
          self.policy_optim.step()
          gc.collect()
      ploss = np.mean(ploss)
      self.plosses.append(ploss)

      if (itr+i) % self.write_interval == 0:
        print('iter: {}, avg stand time: {}, vloss: {}, ploss: {}'.format(
          itr+i, self.stand_time[-1], vloss, ploss ))
        self.write_out(itr+i)

      # print(torch.cuda.memory_allocated(0) / 1e9)

  def get_rollouts(self):
    env = self.env
    rollouts = []
    standing_len = 0.0
    for p in self.policy_net.parameters():
      p.requires_grad = False
    for _ in range(self.env_samples):
      # don't forget to reset the environment at the beginning of each episode!
      # rollout for a certain number of steps!
      rollout = []
      state = env.reset()
      done = False
      while not done and len(rollout) < self.episode_length:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state).squeeze()
        dist = Categorical(probs)
        action = dist.sample().squeeze()
        # input(action)
        tp = [state.squeeze().cpu(), action.cpu(), dist.log_prob(action).squeeze().cpu()]
        # dist = self.policy_net(state)
        # action = dist.sample().squeeze()
        # tp = [state.squeeze(), action.to(self.device), dist.log_prob(action).squeeze()]
        action = action.cpu().numpy()
        # next_state, reward, done, info
        state, reward, done, _ = env.step(action)
        tp.append(torch.FloatTensor([reward]))
        rollout.append(tp)
      rollouts.append(rollout)
      standing_len += len(rollout)
      gc.collect()

    for p in self.policy_net.parameters():
      p.requires_grad = True
    self.stand_time.append(standing_len / self.env_samples)
    # print('avg standing time:', self.stand_time[-1])
    rollouts = self.calculate_values(rollouts)
    gc.collect()
    return rollouts

  def calculate_values(self, rollouts):
    for k in range(len(rollouts)):
      for i in range(len(rollouts[k])):
        gamma = 1.0
        value = 0.0
        for j in range(i,len(rollouts[k])):
          value += rollouts[k][j][-1] * gamma
          gamma *= self.gamma
          # print('value: {}, gamma: {}'.format(value, gamma))
        rollouts[k][i][-1] = value
    return rollouts

  def calculate_advantages(self, rollouts):
    for p in self.value_net.parameters():
      p.requires_grad = False
    for i in range(len(rollouts)):
      for j in range(len(rollouts[i])):
        state, _, _, value = rollouts[i][j]
        state, value = state.to(self.device), value.to(self.device)
        pval = self.value_net(state)
        rollouts[i][j][-1] = (pval - value).cpu()
    for p in self.value_net.parameters():
      p.requires_grad = True
    return rollouts

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
    self.stand_time = train_info['stand_time']
    self.policy_optim = train_info['policy_optimizer']
    self.value_optim = train_info['value_optimizer']

    self.policy_net.load_state_dict(torch.load(
      str(self.policy_path + '_' + str(itr) + '.pt') ))

    self.value_net.load_state_dict(torch.load(
      str(self.value_path + '_' + str(itr) + '.pt') ))

    self.epochs += itr
    return itr

  def write_out(self, itr):
    train_info = {}
    train_info['iter'] = itr
    train_info['plosses'] = self.plosses
    train_info['vlosses'] = self.vlosses
    train_info['stand_time'] = self.stand_time
    train_info['policy_optimizer'] = self.policy_optim
    train_info['value_optimizer'] = self.value_optim
    torch.save( train_info, self.train_info_path )

    torch.save( self.policy_net.state_dict(),
      str(self.policy_path + '_' + str(itr) + '.pt') )

    torch.save( self.value_net.state_dict(),
      str(self.value_path + '_' + str(itr) + '.pt') )


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
