
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


  def forward(self, preds, targets):
    pc, px, py, pw, ph = preds
    #pc, px, py, pw, ph = pc.cpu(), px.cpu(), py.cpu(), pw.cpu(), ph.cpu()
    sx = float(self.img_size) / pc.size(2)
    sy = float(self.img_size) / pc.size(1)

    x, y, w, h, label = targets
    tx = ((x % sx) / sx) - 0.5
    tx = tx * label.float()
    x = x.long() / sx
    ty = ((y % sy) / sy) - 0.5
    ty = ty * label.float()
    y = y.long() / sy
    tw = w / sx
    th = h / sy

    confs = pc
    tlocs = torch.zeros(confs.size())
    tlocs[range(confs.size(0)),x,y] = 1
    tlocs[:,0,0] = 0
    
    if torch.cuda.is_available():
      tx = tx.cuda()
      ty = ty.cuda()
      tw = tw.cuda()
      th = th.cuda()
      tlocs = tlocs.cuda()

    # confidence loss
    dif = torch.pow(tlocs - confs, 2).float() * self.noobj
    dif[range(confs.size(0)),x,y] /= self.noobj
    loss = torch.sum(dif)
    #print(loss)

    # location loss
    px = torch.sum(torch.sum((px * tlocs), dim=-1), dim=-1)
    py = torch.sum(torch.sum((py * tlocs), dim=-1), dim=-1)
    pw = torch.sum(torch.sum((pw * tlocs), dim=-1), dim=-1)
    ph = torch.sum(torch.sum((ph * tlocs), dim=-1), dim=-1)

    loss += torch.sum(torch.pow((tx - px), 2)) * self.coord
    loss += torch.sum(torch.pow((ty - py), 2)) * self.coord
    #print(1,loss)
    loss += torch.sum(torch.pow((torch.sqrt(tw) - torch.sqrt(pw)), 2)) * self.coord
    loss += torch.sum(torch.pow((torch.sqrt(th) - torch.sqrt(ph)), 2)) * self.coord
    #print(2,loss)
    #input('waiting...')
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
