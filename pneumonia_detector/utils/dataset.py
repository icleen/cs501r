import torch
from torch.utils.data import Dataset
from torchvision import transforms
import csv
import os
import numpy as np
import pydicom
from PIL import Image
from skimage.transform import resize


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
    self.img_resize = img_resize
    self.transform = transforms.Compose([transforms.Resize(img_resize),transforms.ToTensor()])

  def get_image(self, idx):
    item = self.data[idx]
    path = os.path.join(self.img_path, item[0] + '.dcm')
    img = pydicom.dcmread(path).pixel_array.astype(np.uint8)
    img = Image.fromarray(img)
    return img

  def __len__(self):
    return  len(self.data)

  def __getitem__(self, idx):
    # item = self.data[0] # test with only the first instance
    item = self.data[idx]
    img = self.get_image(idx)
    img = self.transform(img)
    img -= 128.0
    img /= 128.0
    size = img.shape
    # import pdb; pdb.set_trace()

    label = int(item[5])
    if label != 0:
      x = (float(item[1]) / size[1]) * self.img_resize
      x = torch.tensor(x).float()
      y = (float(item[2]) / size[0]) * self.img_resize
      y = torch.tensor(y).float()
      w = (float(item[3]) / size[1]) * self.img_resize
      w = torch.tensor(w).float()
      h = (float(item[4]) / size[0]) * self.img_resize
      h = torch.tensor(h).float()
    else:
      x = torch.tensor(0).float()
      y = torch.tensor(0).float()
      w = torch.tensor(0).float()
      h = torch.tensor(0).float()
    label = torch.tensor(int(item[5])).long()
    return img, x, y, w, h, label

    # return {
    # 'img': img,
    # # 'x': item[1],
    # # 'y': item[2],
    # # 'width': item[3],
    # # 'height': item[4],
    # 'label': item[5],
    # }

# 0004cfab-14fd-4e49-80ba-63a80b6bddd6

class PneuYoloDataset(Dataset):

  def __init__(self, csv_file='data/stage_1_train_labels.csv',
                     img_path='data/train_images',
                     img_resize=512):
    super(PneuYoloDataset, self).__init__()

    with open(csv_file, 'r') as f:
      reader = csv.reader(f, delimiter=',', quotechar='|')
      self.data = [inst for inst in reader]

    self.csv_file = csv_file
    self.img_path = img_path
    self.img_resize = img_resize
    self.img_shape = (img_resize, img_resize)
    self.transform = transforms.Compose([transforms.Resize(img_resize),transforms.ToTensor()])
    self.max_objects = 1

  def get_image(self, idx):
    item = self.data[idx]
    path = os.path.join(self.img_path, item[0] + '.dcm')
    img = pydicom.dcmread(path).pixel_array.astype(np.uint8)
    img = Image.fromarray(img)
    return img

  def __len__(self):
    return  len(self.data)

  def __getitem__(self, idx):
    # item = self.data[0] # test with only the first instance
    item = self.data[idx]
    img = self.get_image(idx)
    img = np.array(img)
    img = np.stack((img,)*3, axis=-1)

    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
    padded_h, padded_w, _ = input_img.shape
    # Resize and normalize
    input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float()

    #---------
    #  Label
    #---------

    label = int(item[5])
    labels = None
    if label != 0:
      labels = np.zeros((self.max_objects, 5))
      # Extract coordinates for unpadded + unscaled image
      x1 = float(item[1]) - float(item[3])/2
      y1 = float(item[2]) - float(item[4])/2
      x2 = float(item[1]) + float(item[3])/2
      y2 = float(item[2]) + float(item[4])/2
      # Adjust for added padding
      x1 += pad[1][0]
      y1 += pad[0][0]
      x2 += pad[1][0]
      y2 += pad[0][0]
      # Calculate ratios from coordinates
      labels[:, 1] = ((x1 + x2) / 2) / padded_w
      labels[:, 2] = ((y1 + y2) / 2) / padded_h
      labels[:, 3] = float(item[3]) / padded_w
      labels[:, 4] = float(item[4]) / padded_h

    # Fill matrix
    filled_labels = np.zeros((self.max_objects, 5))
    if labels is not None:
      filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
    filled_labels = torch.from_numpy(filled_labels).float()

    return item[0], input_img, filled_labels


if __name__ == '__main__':
  trainset = PneuDataset("data/trainset.csv", "data/train_images")
  x, y = trainset[0]
  print(x)
