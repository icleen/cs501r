from dataset import *
from model import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter
from torchvision import transforms, utils, datasets

from tqdm import tqdm
import pdb
import numpy as np
import matplotlib.pyplot as plt



epochs = 100
learning_rate = 1e-4


trainset = PneuDataset()
trainloader = DataLoader(trainset, batch_size=10, pin_memory=True)

inst = trainset[0]
x = inst['img'].unsqueeze(0)
net = PneuNet(x.shape)

objective = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

losses = []
for epoch in range(epochs):

  loop = tqdm(total=len(trainloader), position=0)

  for i, (x, y) in enumerate(trainloader):
    optimizer.zero_grad()

    preds = model(x)
    loss = objective(preds, y)
    losses.append(loss.item())
    loss.backward()

    optimizer.step()

    loop.set_description('loss: {:.4f}'.format(loss.item()))
    loop.update(1)

  print('epoch: {}, loss: {}'.format())
