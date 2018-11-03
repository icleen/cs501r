import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.nn.parameter import Parameter

import pdb
import numpy as np

class Descriminator(nn.Module):
  """docstring for Descriminator."""
  def __init__(self, in_channels=3, img_shape=64):
    super(Descriminator, self).__init__()
    self.in_channels = in_channels

    self.layer = nn.Sequential(
      nn.Conv2d(in_channels, 128, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
      nn.ReLU(),
      nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
      nn.ReLU()
      # nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
      # nn.ReLU(),
      # nn.Conv2d(1024, 1024, kernel_size=2, stride=1, padding=0),
      # nn.ReLU(),
      # nn.Conv2d(1024, 1, kernel_size=4, stride=1, padding=0)
    )

    out_shape = int(img_shape / pow(2, 4))
    self.linear_size = out_shape*out_shape*1024
    self.fc = nn.Linear(self.linear_size, 1)
    # self.fc = nn.Linear(1024, 1)

  def forward(self, x):
    x = self.layer(x)
    # print(x.size())
    # print(x.size(1)*x.size(2)*x.size(3))
    # print(self.linear_size)
    x = x.view(-1, self.linear_size)
    # x = x.squeeze(2).squeeze(2)
    # print(x.size())
    #
    x = self.fc(x)
    return x

if __name__ == '__main__':
  x = torch.zeros((3, 64, 64), dtype=torch.float)
  x.normal_()
  x.unsqueeze_(0)
  print(x.size())

  descriminator = Descriminator()
  out = descriminator(x)
  print(out.size())
