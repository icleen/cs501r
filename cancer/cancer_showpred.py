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

def interOverUnion(preds, y):
  preds = torch.argmax(preds, 1)
  # value to avoid division by 0
  # intersection will be 0 if model predicts no cancer or the target says there's none
  # union will only be 0 if model predicts no cancer in the image and the target agrees
  inter = ((preds * y).sum()).cpu().numpy()
  union = ((preds + y).sum()).cpu().numpy() - inter
  if union == 0:
    return 1.0
  return inter / union

def main():

  config_path = 'configs/test_config.json'
  if len(sys.argv) > 1:
    config_path = sys.argv[1]
  with open(config_path) as f:
    config = json.load(f)

  valset = CancerDataset(config['validation_set_path'], download=False, train=False)
  trainx, trainy = valset[172]

  trainx = trainx.unsqueeze(0).cuda(async=True)
  trainy = trainy.unsqueeze(0).cuda(async=True)

  model = UNet(config['network']['input_size'], config['network']['output_size'])
  model.cuda()
  model.load_state_dict(torch.load(config['model_save_path']))

  config = None

  gc.collect()
  preds = model(trainx)
  iou = interOverUnion(preds, trainy)
  print('iou: {}'.format(iou))

  preds = torch.argmax(preds, 1)
  preds = preds.squeeze(0).squeeze(0).cpu().numpy()
  # print(preds.shape)
  plt.imshow(preds)
  plt.savefig('cancer_val_pred.png')
  temp = trainx.cpu().squeeze(0).numpy()
  # print(temp.shape)
  temp = temp.transpose(1,2,0)
  # print(temp.shape)
  plt.imshow(temp)
  plt.savefig('cancer_val_org.png')

  temp = trainy.cpu().squeeze(0).numpy()
  # print(temp.shape)
  plt.imshow(temp)
  plt.savefig('cancer_val_gt.png')

  preds = None
  gc.collect()



if __name__ == '__main__':
  main()
