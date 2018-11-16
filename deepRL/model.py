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
  def __init__(self, state_size=4, action_size=2):
    super(Policy1D, self).__init__()

    self.layer = nn.Sequential(
      nn.Linear(state_size, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU(),
      # nn.Linear(100, 100),
      # nn.ReLU(),
      # nn.Linear(100, 100),
      # nn.ReLU(),
      nn.Linear(100, action_size),
      nn.Softmax(dim=1)
    )

  def forward(self, x):
    x = self.layer(x)
    # return Categorical(x)
    return x


class Value1D(nn.Module):
  """docstring for Value1D."""
  def __init__(self, state_size=4):
    super(Value1D, self).__init__()

    self.layer = nn.Sequential(
      nn.Linear(state_size, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU(),
      nn.Linear(100, 100),
      nn.ReLU(),
      # nn.Linear(100, 100),
      # nn.ReLU(),
      # nn.Linear(100, 100),
      # nn.ReLU(),
      nn.Linear(100, 1)
    )

  def forward(self, x):
    x = self.layer(x)
    return x


if __name__ == '__main__':
  policy = Policy1D(4, 2)
  value = Value1D(4)
