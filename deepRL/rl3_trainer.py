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

from dataset import RLDataset
from model import Policy1D, Value1D
from loss import PPOLoss


def _prepare_numpy(ndarray, device):
    return torch.from_numpy(ndarray).float().unsqueeze(0).to(device)

def _prepare_tensor_batch(tensor, device):
    return tensor.detach().float().to(device)

def _calculate_returns(trajectory, gamma):
  current_return = 0
  for i in reversed(range(len(trajectory))):
    state, action_dist, action, reward = trajectory[i]
    current_return = reward + gamma * current_return
    trajectory[i] = (state, action_dist, action, reward, current_return)

def _run_envs(env, embedding_net, policy, experience_queue, reward_queue, num_rollouts, max_episode_length,
              gamma, device):
    for _ in range(num_rollouts):
      current_rollout = []
      s = env.reset()
      episode_reward = 0
      for _ in range(max_episode_length):
        input_state = _prepare_numpy(s, device)
        if embedding_net:
          input_state = embedding_net(input_state)

        action_dist = policy(input_state).squeeze().cpu().detach().numpy()
        action_one_hot = np.random.multinomial(1, action_dist)
        action = np.array([np.argmax(action_one_hot)], dtype=np.uint8)
        s_prime, r, t = env.step(action)

        if type(r) != float:
          print('run envs:', r, type(r))
        current_rollout.append((s, action_dist, action, r))
        episode_reward += r
        if t:
            break
        s = s_prime
      _calculate_returns(current_rollout, gamma)
      experience_queue.put(current_rollout)
      reward_queue.put(episode_reward)


class ExperienceDataset(Dataset):
    def __init__(self, experience):
        super(ExperienceDataset, self).__init__()
        self._exp = []
        for x in experience:
            self._exp.extend(x)
        self._length = len(self._exp)

    def __getitem__(self, index):
        return self._exp[index]

    def __len__(self):
        return self._length


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

    # self.env = gym.make(config['model']['gym'])
    env_factory = CartPoleEnvironmentFactory()
    self.env_threads=1
    self.envs = [env_factory.new() for _ in range(self.env_threads)]
    rollouts_per_thread = self.env_samples // self.env_threads
    remainder = self.env_samples % self.env_threads
    self.rollout_nums = ([rollouts_per_thread + 1] * remainder) + ([rollouts_per_thread] * (self.env_threads - remainder))

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


  def train(self, itr=0):
    for i in range(self.epochs):
      # generate rollouts
      experience_queue = Queue()
      reward_queue = Queue()
      threads = [Thread(target=_run_envs,
                        args=(self.envs[i],
                        None,
                        self.policy_net,
                        experience_queue,
                        reward_queue,
                        self.rollout_nums[i],
                        self.episode_length,
                        self.gamma,
                        self.device)) for i in range(self.env_threads)]
      for x in threads:
        x.start()
      for x in threads:
        x.join()

      # Collect the experience
      rollouts = list(experience_queue.queue)
      # avg_r = sum(reward_queue.queue) / reward_queue.qsize()
      # loop.set_description('avg reward: % 6.2f' % (avg_r))
      self.stand_time.append(sum(reward_queue.queue) / reward_queue.qsize())

      # Update the policy
      experience_dataset = ExperienceDataset(rollouts)
      dataloader = DataLoader(experience_dataset,
                              batch_size=self.policy_batch_size,
                              shuffle=True,
                              pin_memory=True)

      # Learn a policy
      vlosses = []
      plosses = []
      # dataset = RLDataset(rollouts)
      # dataloader = DataLoader(dataset, batch_size=self.policy_batch_size,
      #                         shuffle=True, pin_memory=True)
      for _ in range(self.policy_epochs):
        # train policy network
        for state, aprob, action, reward, value in dataloader:
          state = _prepare_tensor_batch(state, self.device)
          aprob = _prepare_tensor_batch(aprob, self.device)
          # state, aprob = state.to(self.device), aprob.to(self.device)
          action = _prepare_tensor_batch(action, self.device)
          value = _prepare_tensor_batch(value, self.device).unsqueeze(1)
          # action, value = action.to(self.device), value.to(self.device)

          pdist = self.policy_net(state)
          clik = self.multinomial_likelihood(pdist, action)
          olik = self.multinomial_likelihood(aprob, action)
          ratio = (clik / olik)

          pval = self.value_net(state)
          # print('pred: {}, actual: {}'.format(pval.size(), value.size()))
          # input('waiting...')
          vloss = self.value_loss(pval, value)
          vlosses.append(vloss.cpu().item())

          advantage = pval - value
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

      # print(torch.cuda.memory_allocated(0) / 1e9)

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
