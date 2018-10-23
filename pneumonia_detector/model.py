import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.nn.parameter import Parameter
import torchvision.models as models

import pdb
import numpy as np


# class Residual(nn.Module):
#
#   def __init__(self, in_channels, out_channels, kernel_size,
#                stride=1, padding=0, dilation=1, groups=1, bias=True):
#     self.__dict__.update(locals())
#     super(Conv2d, self).__init__()
#
#     self.weight = Parameter( torch.Tensor(out_channels, in_channels,
#                                           *kernel_size) )
#     self.bias = Parameter(torch.Tensor(out_channels))
#
#     self.weight.data.uniform_(-1, 1)
#     self.bias.data.uniform_(0, 0)
#
#
#   def forward(self, x):
#     return F.conv2d(x, self.weight, self.bias, self.stride, self.padding,
#                     self.dilation, self.groups)
#
#   def extra_repr(self):
#     return 'hello repr!'


class PneuNet(nn.Module):

  def __init__(self, img_shape, classes=2):
    super(PneuNet, self).__init__()

    in_channels = img_shape[0]
    assert img_shape[1] == img_shape[2]
    width = img_shape[1]
    maxpools = 2
    fcc_shape = int(width / pow(2, maxpools))
    self.fcc_shape = fcc_shape * fcc_shape * 256

    self.net = nn.Sequential(
      nn.Conv2d(in_channels, 64, (3,3), padding=1, stride=1),
      nn.ReLU(),
      nn.Conv2d(64, 64, (3,3), padding=1, stride=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(64, 128, (3,3), padding=1, stride=1),
      nn.ReLU(),
      nn.Conv2d(128, 128, (3,3), padding=1, stride=1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(128, 256, (3,3), padding=1, stride=1),
      nn.ReLU(),
      nn.Conv2d(256, 256, (3,3), padding=1, stride=1),
      nn.ReLU()
    )

    self.resnet = models.resnet50()

    self.fc1 = nn.Linear(self.fcc_shape, classes)


  def forward(self, x):
    out = self.resnet(x)
    print(out.size())
    x = self.net(x)
    x = x.view(-1, self.fcc_shape)
    x = self.fc1(x)
    return x
