
import torch
from torch import nn

class YoloLoss(nn.Module):
  """docstring for YoloLoss."""
  def __init__(self, img_size, noobj=0.5, coord=5):
    super(YoloLoss, self).__init__()
    self.img_size = img_size
    self.noobj = noobj
    self.coord = coord
    self.sig = nn.Sigmoid()

    if torch.cuda.is_available():
      self.dtype = torch.cuda.FloatTensor
      self.long = torch.cuda.LongTensor
    else:
      self.dtype = torch.FloatTensor
      self.long = torch.LongTensor


  def forward(self, preds, targets):
    size = preds.size()
    sx = float(self.img_size) / size[3]
    sy = float(self.img_size) / size[2]

    x, y, w, h, label = targets
    tx = ((x % sx) / sx) - 0.5
    tx = tx * label.dtype(self.dtype)
    x = x.dtype(self.long) / sx
    ty = ((y % sy) / sy) - 0.5
    ty = ty * label.dtype(self.dtype)
    y = y.dtype(self.long) / sy
    tw = w / sx
    th = h / sy

    confs = preds[:,0,:,:]
    tlocs = torch.zeros(confs.size()).dtype(self.dtype)
    tlocs[range(confs.size(0)),x,y] = 1
    tlocs[:,0,0] = 0

    # confidence loss
    dif = torch.pow(tlocs - confs, 2).dtype(self.dtype) * self.noobj
    dif[range(confs.size(0)),x,y] /= self.noobj
    loss = torch.sum(dif)

    # location loss
    px = torch.sum(torch.sum((preds[:,1,:,:] * tlocs), dim=-1), dim=-1)
    py = torch.sum(torch.sum((preds[:,2,:,:] * tlocs), dim=-1), dim=-1)
    pw = torch.sum(torch.sum((preds[:,3,:,:] * tlocs), dim=-1), dim=-1)
    ph = torch.sum(torch.sum((preds[:,4,:,:] * tlocs), dim=-1), dim=-1)

    loss += torch.sum(torch.pow((tx - px), 2)) * self.coord
    loss += torch.sum(torch.pow((ty - py), 2)) * self.coord
    loss += torch.sum(torch.pow((torch.sqrt(tw) - torch.sqrt(pw)), 2)) * self.coord
    loss += torch.sum(torch.pow((torch.sqrt(th) - torch.sqrt(ph)), 2)) * self.coord

    return loss


def main():
  from utils.dataset import PneuDataset
  from yolo_model import PneuYoloNet
  imgsize = 512
  model = PneuYoloNet((1, imgsize, imgsize))
  trainset = PneuDataset("data/trainset.csv", "data/train_images", img_resize=imgsize)
  objective = YoloLoss(imgsize)

  for i in range(2):
    img, x, y, w, h, label = trainset[i]
    img = img.unsqueeze(0)
    x, y, w, h, label = x.unsqueeze(0), y.unsqueeze(0), w.unsqueeze(0), h.unsqueeze(0), label.unsqueeze(0)

    preds = model(img)
    preds *= 0

    loss = objective(preds, (x, y, w, h, label))
    print(loss)


if __name__ == '__main__':
  main()
