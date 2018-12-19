import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.nn.parameter import Parameter
from resnet import resnet50

import pdb
import numpy as np

# https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/models.py
# https://github.com/ayooshkathuria/pytorch-yolo-v3/blob/master/darknet.py

class PneuYoloNet(nn.Module):

  def __init__(self, img_shape, num_classes=1, bb_num=1):
    super(PneuYoloNet, self).__init__()
    in_channels = img_shape[0]
    assert img_shape[1] == img_shape[2]
    width = img_shape[1]
    self.bb_num = bb_num

    # avg_pool_size = 7
    # maxpoolsr = 5
    # fccr_shape = int(width / pow(2, maxpoolsr)) - (avg_pool_size-1)
    # self.fccr_shape = fccr_shape * fccr_shape * 2048
    self.resnet = resnet50(in_channels=in_channels)
    resout_channels = 2048
    """the number of out_channels is the number of class + number of
    things you want to predict for each bounding box * the number of bounding boxes.
    In our case, we only have one class, so we just predict the x,y,w,h offsets
    and confidence for each bounding box."""
    bb_preds = 5
    out_channels = bb_preds * bb_num
    """this conv layer just reshapes the activation to be the
    right shape defined above"""
    self.conv = nn.Conv2d(resout_channels, out_channels,
                          kernel_size=1, stride=1, padding=0)
                          
    self.sig = nn.Sigmoid()
    self.relu = nn.ReLU()


  def forward(self, x):
    out = self.resnet(x)
    out = self.conv(out)
    out.transpose(1,2)
    pc = out[:,0,:,:]
    px = self.sig(out[:,1,:,:]) - 0.5
    py = self.sig(out[:,2,:,:]) - 0.5
    pw = self.relu(out[:,3,:,:])
    ph = self.relu(out[:,4,:,:])
    return (pc, px, py, pw, ph)
