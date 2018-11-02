import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.nn.parameter import Parameter

import pdb
import numpy as np

class Generator(nn.Module):
  """docstring for Generator."""
  def __init__(self, z_size=100):
    super(Generator, self).__init__()
    self.z_size = z_size

    self.init_size = 4
    self.init_channels = 1024
    self.fc = nn.Linear(z_size, self.init_channels*self.init_size*self.init_size)

    channel0 = self.init_channels
    channel1 = 512
    channel2 = 256
    channel3 = 128
    channel4 = 3
    self.layer = nn.Sequential(
      nn.ConvTranspose2d(
        channel0, channel1, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.Conv2d(channel1, channel1, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),

      nn.ConvTranspose2d(
        channel1, channel2, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.Conv2d(channel2, channel2, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),

      nn.ConvTranspose2d(channel2, channel3, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.Conv2d(channel3, channel3, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),

      nn.ConvTranspose2d( channel3, channel4, kernel_size=3, stride=2, padding=1, output_padding=1),
      nn.Conv2d(channel4, channel4, kernel_size=3, stride=1, padding=1),
      nn.Sigmoid(),
    )

  def forward(self, x=None):
    if x is None:
      x = torch.zeros((self.z_size), dtype=torch.float)
      x.normal_()
      x.unsqueeze_(0) # become a pretend batch
      if torch.cuda.is_available():
        x.cuda(async=True)

    x = self.fc(x)
    x = x.view(-1, self.init_channels, self.init_size, self.init_size)
    x = self.layer(x)
    return x


if __name__ == '__main__':
  net = Generator()
  out = net.forward()
  print(out.size())

  z = torch.zeros((10, 100), dtype=torch.float)
  z.normal_()
  # z.unsqueeze_(0) # become a pretend batch
  out2 = net.forward(z)
  print(out2.size())
