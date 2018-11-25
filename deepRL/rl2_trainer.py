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

    betas = (config['train']['betas1'], config['train']['betas2'])
    weight_decay = config['train']['weight_decay']
    lr = config['train']['lr']
    params = chain(self.policy_net.parameters(), self.value_net.parameters())
    self.optim = optim.Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)

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
    self.gif_path = config['model']['gif_save_path'].split('.gif')[0]
    self.graph_path = config['model']['graph_save_path']


  def train(self, itr=0):
    for i in range(self.epochs):
      # generate rollouts
      rollouts = self.get_rollouts()

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

          pdist = self.policy_net(state)
          clik = self.multinomial_likelihood(pdist, action)
          olik = self.multinomial_likelihood(aprob, action)
          ratio = (clik / olik)

          pval = self.value_net(state)
          vloss = self.value_loss(pval, value)
          vlosses.append(vloss.cpu().item())

          advantage = value - pval.detach()
          ploss = self.ppoloss(ratio, advantage)
          plosses.append(ploss.cpu().item())

          self.optim.zero_grad()
          loss = ploss + vloss
          loss.backward()
          self.optim.step()
          gc.collect()
      self.vlosses.append(np.mean(vlosses))
      self.plosses.append(np.mean(plosses))

      if (itr+i) % self.write_interval == 0:
        print('iter: {}, avg stand time: {}, vloss: {}, ploss: {}'.format(
          itr+i, self.stand_time[-1], vloss, ploss ))
        self.write_out(itr+i)
        self.make_gif(itr, rollouts[0])

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
        probs_np = probs.cpu().detach().numpy()
        action_one_hot = np.random.multinomial(1, probs_np)
        action = np.argmax(action_one_hot)
        tp = [state.squeeze().cpu(), probs.cpu(), torch.LongTensor([action]).cpu()]
        # next_state, reward, done, info
        state, reward, done, _ = env.step(action)
        tp.append(torch.FloatTensor([reward]).cpu())
        rollout.append(tp)
      value = 0.0
      for i in reversed(range(len(rollout))):
        value = rollout[i][-1] + self.gamma * value
        rollout[i].append(value.cpu())
      rollouts.append(rollout)
      standing_len += len(rollout)
      gc.collect()

    for p in self.policy_net.parameters():
      p.requires_grad = True
    self.stand_time.append(standing_len / self.env_samples)
    # print('avg standing time:', self.stand_time[-1])

    return rollouts

  def multinomial_likelihood(self, dist, idx):
    return dist[range(dist.shape[0]), idx.long()[:, 0]].unsqueeze(1)

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
    self.optim = train_info['optimizer']
    # self.policy_optim = train_info['policy_optimizer']
    # self.value_optim = train_info['value_optimizer']

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
    train_info['optimizer'] = self.optim
    # train_info['policy_optimizer'] = self.policy_optim
    # train_info['value_optimizer'] = self.value_optim
    torch.save( train_info, self.train_info_path )

    torch.save( self.policy_net.state_dict(),
      str(self.policy_path + '_' + str(itr) + '.pt') )

    torch.save( self.value_net.state_dict(),
      str(self.value_path + '_' + str(itr) + '.pt') )

    if itr > 2:
      plt.plot(self.vlosses[2:], label='value loss')
      plt.plot(self.plosses[2:], label='policy loss')
      plt.plot(self.stand_time[2:], label='stand time')
      plt.legend()
      plt.xlabel('epochs')
      plt.ylabel('loss')
      plt.savefig(self.graph_path)


  def make_gif(self, itr, rollout):
    pass
    # Make gifs
    # with imageio.get_writer(str(self.gif_path + '_' + str(itr) + '.gif'),
    #                         mode='I', duration=1 / 30) as writer:
    #   for x in rollout:
    #     x = x[0].numpy()
    #     input(x)
    #     writer.append_data((x * 255).astype(np.uint8))


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
