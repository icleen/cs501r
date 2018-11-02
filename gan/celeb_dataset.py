import torchvision
from torchvision import transforms
import torch
from torch.utils.data import Dataset, DataLoader
import os

class CelebaDataset(Dataset):
  def __init__(self, root, size=128, train=True):
    super(CelebaDataset, self).__init__()
    self.dataset_folder = torchvision.datasets.ImageFolder(
      os.path.join(root), transform = transforms.Compose(
        [transforms.Resize((size,size)),transforms.ToTensor()] )
      )
  def __getitem__(self,index):
    img = self.dataset_folder[index]
    return img[0]

  def __len__(self):
    return len(self.dataset_folder)

if __name__ == '__main__':
  dataset = CelebaDataset('/fslhome/iclee141/compute/celebA/img_align_celeba/')
  x = dataset[0]
  print(s.size())
