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

  config_path = 'teste_config.json'
  if len(sys.argv) > 1:
    config_path = sys.argv[1]
  with open(config_path) as f:
      config = json.load(f)

  trainset = CancerDataset(config['training_set_path'], download=False)
  trainx, trainy = trainset[-1]
  trainx = trainx.unsqueeze(0).cuda(async=True)
  trainy = trainy.unsqueeze(0).cuda(async=True)
  # print(torch.sum(trainx))
  # print(trainx.size())
  # print(torch.sum(trainy))
  # print(trainy.size())

  model = UNet(config['network']['input_size'], config['network']['output_size'])
  model.cuda()

  objective = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=config['network']['learning_rate'])

  epochs = 1000
  losses = []
  valosses = []
  lowest_loss = float('inf')
  gc.collect()

  for epoch in range(epochs):

    optimizer.zero_grad()

    preds = model(trainx)
    loss = objective(preds, trainy)
    losses.append(loss.item())

    loss.backward()
    optimizer.step()
    preds = None
    loss = None

    if epoch % 100 == 0:
      print('epoch: {}, loss: {:.6f}, memory: {}'.format(
        epoch+1, losses[-1], torch.cuda.memory_allocated(0) / 1e9 ) )

      if losses[-1] < lowest_loss:
        lowest_loss = losses[-1]
        print("Saving Best")
        dirname = os.path.dirname(config['model_save_path'])
        if len(dirname) > 0 and not os.path.exists(dirname):
          os.makedirs(dirname)
        torch.save(model.state_dict(), os.path.join(config['model_save_path']))

    gc.collect()


  plt.plot(losses, label='train')
  plt.legend()
  plt.xlabel('iterations')
  plt.ylabel('loss')
  plt.savefig('cancer_fig.png')

  model.load_state_dict(torch.load(config['model_save_path']))
  gc.collect()

  preds = model(trainx)
  preds = torch.argmax(preds, 1)
  print('pred sum: {}, gt sum: {}'.format(torch.sum(preds), torch.sum(trainy)))

  acc = torch.mean((trainy == preds).type(torch.FloatTensor))
  print('acc: {}'.format(acc))

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
