import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from torch.nn.parameter import Parameter

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pdb
import json
import sys
import gc
import os

from cancer_dataset import CancerDataset
from cancer_model import UNet


def main():

  config_path = 'config.json'
  if len(sys.argv) > 1:
    config_path = sys.argv[1]
  with open(config_path) as f:
      config = json.load(f)

  model = UNet(config['network']['input_size'], config['network']['output_size'])
  model.load_state_dict(torch.load(config['model_save_path']))
  model.cuda()

  valset = CancerDataset(config['validation_set_path'], download=False, train=False)
  val_loader = DataLoader(valset, batch_size=1, pin_memory=True)

  valacc = []
  doloop = True
  gc.collect()

  if doloop:
    loop = tqdm(total=len(val_loader), position=0)

  for i, (x, y) in enumerate(val_loader):
    x, y = x.cuda(async=True), y.cuda(async=True)

    preds = model(x)
    preds = torch.argmax(preds, 1)

    acc = torch.mean((trainy == preds).type(torch.FloatTensor))
    valacc.append(acc)

    if doloop:
      loop.set_description(
        'iter: {}, loss: {:.6f}, memory: {}'.format(
          i, losses[-1], torch.cuda.memory_allocated(0) / 1e9 ) )
      loop.update(1)

    preds = None
    gc.collect()

  if doloop:
    loop.close()

  print('Mean Acc: {}'.format(np.mean(valacc)))


if __name__ == '__main__':
  main()
