import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from torch.nn.parameter import Parameter
from tqdm import tqdm
import pdb

# assert torch.cuda.is_available()
# You need to request a GPU from Runtime > Change Runtime Type


class UNet(nn.Module):

  def __init__(self, input_size, output_size):
    super(UNet, self).__init__()

    self.layer1 = nn.Sequential(
      nn.Conv2d(input_size[0], 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU()
    )

    self.layer2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU()
    )

    self.layer3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU()
    )

    self.layer4 = nn.Sequential(
      nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU()
    )

    self.midlayer = nn.Sequential(
      nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
    )

    self.uplayer4 = nn.Sequential(
      nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
    )

    self.uplayer3 = nn.Sequential(
      nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
    )

    self.uplayer2 = nn.Sequential(
      nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
    )

    self.uplayer1 = nn.Sequential(
      nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(64, 2, kernel_size=3, stride=1, padding=1)
    )

    self.maxpool = nn.MaxPool2d(kernel_size=2)




  def forward(self, x):
    x1 = self.layer1( x )
    # print(x1.size())
    x2 = self.layer2( self.maxpool(x1) )
    # print(x2.size())
    x3 = self.layer3( self.maxpool(x2) )
    # print(x3.size())
    x4 = self.layer4( self.maxpool(x3) )
    # print(x4.size())

    xout = self.midlayer( self.maxpool(x4) )
    # print(xout.size())

    xout = self.uplayer4( torch.cat((xout, x4), 1) )
    # print(xout.size())
    xout = self.uplayer3( torch.cat((xout, x3), 1) )
    # print(xout.size())
    xout = self.uplayer2( torch.cat((xout, x2), 1) )
    # print(xout.size())
    xout = self.uplayer1( torch.cat((xout, x1), 1) )
    # print(xout.size())
    return xout

if __name__ == '__main__':
  model = UNet((3, 512, 512))
  model.cuda()
