import torch
from torch.utils.data import Dataset
from torchvision import transforms
import csv
import os
import numpy as np
import pydicom
from PIL import Image


class PneuDataset(Dataset):

  def __init__(self, csv_file='data/stage_1_train_labels.csv',
                     img_path='data/train_images',
                     img_resize=512):
    super(PneuDataset, self).__init__()

    with open(csv_file, 'r') as f:
      reader = csv.reader(f, delimiter=',', quotechar='|')
      self.data = [inst for inst in reader]

    self.csv_file = csv_file
    self.img_path = img_path
    self.transform = transforms.Compose([transforms.Resize(img_resize),transforms.ToTensor()])



  def __len__(self):
    return  len(self.data)

  def __getitem__(self, idx):
    # item = self.data[0] # test with only the first instance
    item = self.data[idx]
    path = os.path.join(self.img_path, item[0] + '.dcm')
    img = pydicom.dcmread(path).pixel_array.astype(np.uint8)
    img = Image.fromarray(img)
    img = self.transform(img)
    # import pdb; pdb.set_trace()

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

if __name__ == '__main__':
  trainset = PneuDataset("data/trainset.csv", "data/train_images")
  x, y = trainset[0]
  print(x)
