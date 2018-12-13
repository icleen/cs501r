import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.nn.parameter import Parameter
from torch.distributions import Categorical

import pdb
import numpy as np

class Policy1D(nn.Module):
  """docstring for Policy1D."""
  def __init__(self, state_size=4, action_size=2, hidden_size=100,
                    layers=2, logheat=1.0, random=0.0):
    super(Policy1D, self).__init__()

    self.logheat = logheat
    self.random = random
    self.action_size = action_size

    modules = []
    modules.append(nn.Linear(state_size, hidden_size))
    modules.append(nn.ReLU())
    for n in range(layers):
      modules.append(nn.Linear(hidden_size, hidden_size))
      modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_size, action_size))
    self.policy = nn.Sequential(*modules)

    modules = []
    modules.append(nn.Linear(state_size, hidden_size))
    modules.append(nn.ReLU())
    for n in range(layers):
      modules.append(nn.Linear(hidden_size, hidden_size))
      modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_size, 1))
    self.value = nn.Sequential(*modules)

    self.softmax = nn.Softmax(dim=1)

  def forward(self, x, get_action=True):
    p, v = self.policy(x), self.value(x)
    p = self.softmax(p /  self.logheat)

    if not get_action:
      return p, v

    batch_size = x.shape[0]
    actions = np.empty((batch_size, 1), dtype=np.uint8)
    probs_np = p.cpu().detach().numpy()
    for i in range(batch_size):
      if np.random.uniform() < self.random:
        actions[i, 0] = np.random.randint(0, self.action_size)
        continue
      action_one_hot = np.random.multinomial(1, probs_np[i])
      action_idx = np.argmax(action_one_hot)
      actions[i, 0] = action_idx
    return p, actions


class Value1D(nn.Module):
  """docstring for Value1D."""
  def __init__(self, state_size=4, hidden_size=100, layers=2):
    super(Value1D, self).__init__()

    modules = []
    modules.append(nn.Linear(state_size, hidden_size))
    modules.append(nn.ReLU())
    for n in range(layers):
      modules.append(nn.Linear(hidden_size, hidden_size))
      modules.append(nn.ReLU())
    modules.append(nn.Linear(hidden_size, 1))
    self.layer = nn.Sequential(*modules)

  def forward(self, x):
    x = self.layer(x)
    return x


class Policy2D(nn.Module):
  """docstring for Policy2D."""
  def __init__(self, state_size=[3, 84, 84], action_size=6, hidden_size=64,
                    layers=3, logheat=1.0):
    super(Policy2D, self).__init__()

    self.logheat = logheat

    modules = []
    modules.append(nn.Conv2d(state_size[0], hidden_size,
                  kernel_size=3, stride=1, padding=1))
    modules.append(nn.ReLU())
    hidden_prev = hidden_size
    maxpool = nn.MaxPool2d(kernel_size=2)
    down = 0
    for n in range(layers):
      if n % 2 == 0:
        down += 1
        modules.append(maxpool)
        hidden_size *= 2
      modules.append(nn.Conv2d(hidden_prev, hidden_size,
                    kernel_size=3, stride=1, padding=1))
      modules.append(nn.ReLU())
      hidden_prev = hidden_size

    self.net = nn.Sequential(*modules)
    x = int(state_size[1] / pow(2, down))
    y = int(state_size[2] / pow(2, down))
    self.modsize = x * y * hidden_size
    self.policy = nn.Sequential(
      nn.Linear(self.modsize, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, action_size)
    )
    self.softmax = nn.Softmax(dim=1)
    self.value = nn.Sequential(
      nn.Linear(self.modsize, hidden_size),
      nn.ReLU(),
      nn.Linear(hidden_size, 1)
    )


  def sample(self, probs):
    probs = probs.squeeze().cpu()
    probs_np = probs.detach().numpy()
    action_one_hot = np.random.multinomial(1, probs_np)
    action = np.argmax(action_one_hot)
    return action

  def forward(self, x):
    x = self.net(x)
    x = x.view(-1, self.modsize)
    p, v = self.policy(x), self.value(x)
    p = self.softmax(p /  self.logheat)
    return p, v



if __name__ == '__main__':
  policy = Policy1D(4, 2)
  value = Value1D(4)

  import gym
  env = gym.make('SpaceInvaders-v0')
  state = env.reset()
  state = np.rollaxis(state,2,0)
  action_size = env.action_space
  print('state: {}, action space: {}'.format(state.shape, action_size))
  policy = Policy2D(state.shape, 6)
  state = torch.FloatTensor(state).unsqueeze(0)
  p, v = policy(state)
  print(p.size(), v.size())
