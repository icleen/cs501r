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

def clear_log(log_path):
  with open(log_path, 'w') as f:
    f.write('')

def save_log(tipe, itr, val, log_path):
  with open(log_path, 'a') as f:
    f.write('{}, {}, {}\n'.format(tipe, itr, val))

def interOverUnion(preds, y):
  preds = torch.argmax(preds, 1)
  # value to avoid division by 0
  # intersection will be 0 if model predicts no cancer or the target says there's none
  # union will only be 0 if model predicts no cancer in the image and the target agrees
  inter = ((preds * y).sum()).cpu().numpy()
  union = ((preds + y).sum()).cpu().numpy() - inter
  if union == 0:
    return 1.0
  # print(inter)
  # print(union)
  # print(inter / float(union))
  return inter / union

def val_epoch(itr, model, val_loader, highest_IOU, log_path, model_path):
  IOU = np.mean([interOverUnion(model(x.cuda()), y.cuda()) for x, y in val_loader])
  # print(IOU)
  print('iter: {}, IOU: {:.6f}, memory: {}'.format(
    itr, IOU, torch.cuda.memory_allocated(0) / 1e9 ) )
  save_log('valid', itr, IOU, log_path)
  if IOU > highest_IOU:
    highest_IOU = IOU
    print("Saving Best")
    dirname = os.path.dirname(model_path)
    if len(dirname) > 0 and not os.path.exists(dirname):
      os.makedirs(dirname)
    torch.save(model.state_dict(), os.path.join(model_path))
  gc.collect()

  return highest_IOU

def main():

  config_path = 'config.json'
  if len(sys.argv) > 1:
    config_path = sys.argv[1]
  with open(config_path) as f:
      config = json.load(f)

  trainset = CancerDataset(config['training_set_path'], download=False)
  train_loader = DataLoader(trainset, batch_size=config['batch_size'],
                            shuffle=True, pin_memory=True)

  valset = CancerDataset(config['validation_set_path'], download=False, train=False)
  val_loader = DataLoader(valset, batch_size=1, pin_memory=True)

  model = UNet(config['network']['input_size'], config['network']['output_size'])
  model.cuda()

  objective = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=config['network']['learning_rate'])

  itr = 0
  epochs = config['epochs']
  doloop = config['looper']
  highest_IOU = float('-inf')
  model_path = config['model_save_path']
  log_path = config['log_path']
  clear_log(log_path)
  config = None
  gc.collect()

  for epoch in range(epochs):

    if doloop:
      loop = tqdm(total=len(train_loader), position=0)

    for i, (x, y) in enumerate(train_loader):
      gc.collect()
      x, y = x.cuda(async=True), y.cuda(async=True)

      optimizer.zero_grad()

      preds = model(x)
      loss = objective(preds, y)

      loss.backward()
      optimizer.step()
      preds = None
      loss = loss.item()
      gc.collect()

      if doloop:
        loop.set_description(
          'iter: {}, loss: {:.6f}, memory: {}'.format(
            itr, loss, torch.cuda.memory_allocated(0) / 1e9 ) )
        loop.update(1)
      elif i % 20 == 0:
        print('iter: {}, loss: {:.6f}, memory: {}'.format(
          itr, loss, torch.cuda.memory_allocated(0) / 1e9 ) )

        save_log('train', itr, loss, log_path)

      loss = None

      # validate every 40 batches and save the model if the validation loss is
      # lower than the lowest loss seen so far
      if i % 50 == 0:
        highest_IOU = val_epoch(itr, model, val_loader, highest_IOU, log_path, model_path)

      gc.collect()
      itr += 1

    if doloop:
      loop.close()

  highest_IOU = val_epoch(itr, model, val_loader, highest_IOU, log_path, model_path)



if __name__ == '__main__':
  main()
