import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, utils, datasets
from torch.nn.parameter import Parameter
from tqdm import tqdm
import pdb

from cancer_dataset import CancerDataset
from cancer_model import UNet


def main():

  trainset = CancerDataset()
  trainloader = DataLoader(trainset)

  x, y = trainset[0]

  model = UNet(x.size())
  model.cuda()


  x, y = dataset[0]
  model = ConvNet(x.size(), y.size())
  model = model.cuda()

  # objective = nn.CrossEntropyLoss()
  # code up own CrossEntropy
  objective = CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-4)

  epochs = 1
  losses = []
  valosses = []

  for epoch in range(epochs):

    loop = tqdm(total=len(train_loader), position=0)

    for i, (x, y) in enumerate(train_loader):
      x, y = x.cuda(async=True), y.cuda(async=True)

      optimizer.zero_grad()

      preds = model(x)
      loss = objective(preds, y)
      losses.append(loss.item())
      loss.backward()

      optimizer.step()

      loop.set_description('loss: {:.4f}'.format(loss.item()))
      loop.update(1)

      if i % 10 == 0:
        valosses.append( ( len(losses),
          np.mean([objective(model(x.cuda()), y.cuda()).item() for x, y in val_loader]) ) )

    loop.close()

  valosses.append( ( len(losses),
    np.mean([objective(model(x.cuda()), y.cuda()).item() for x, y in val_loader]) ) )



if __name__ == '__main__':
  main()
