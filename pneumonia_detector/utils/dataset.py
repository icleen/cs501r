import torch
from torch.utils.data import Dataset
import csv
import os
import numpy as np
import pydicom


class PneuDataset(Dataset):

  def __init__(self, csv_file='data/stage_1_train_labels.csv', img_path='data/train_images'):
    super(PneuDataset, self).__init__()

    with open(csv_file, 'r') as f:
      reader = csv.reader(f, delimiter=',', quotechar='|')
      self.data = [inst for inst in reader]

    self.csv_file = csv_file
    self.img_path = img_path



  def __len__(self):
    return  len(self.data)

  def __getitem__(self, idx):
    # item = self.data[0] # test with only the first instance
    item = self.data[idx]
    path = os.path.join(self.img_path, item[0] + '.dcm')
    img = pydicom.dcmread(path).pixel_array.astype(np.float32)
    img = img / 128.0 - 1.0

    img = torch.from_numpy(img).unsqueeze(0)
    label = torch.tensor(int(item[5])).long()

    return img, label

    # return {
    # 'img': img,
    # # 'x': item[1],
    # # 'y': item[2],
    # # 'width': item[3],
    # # 'height': item[4],
    # 'label': item[5],
    # }

# 0004cfab-14fd-4e49-80ba-63a80b6bddd6
