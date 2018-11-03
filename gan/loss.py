import torch
import torch.nn as nn

class DescriminatorLoss(nn.Module):
  """docstring for DescriminatorLoss."""
  def __init__(self, lam):
    super(DescriminatorLoss, self).__init__()
    self.lam = lam
    if torch.cuda.is_available():
      self.device = torch.device("cuda")
    else:
      self.device = torch.device("cpu")

  def forward(self, predg, predt, predh, xhat):
    # xhat.requires_grad_()
    # loss = 0.0
    # gs = predh.size(0)
    # for g in range(gs):
    #   print(g)
    #   loss += self.lam*(torch.autograd.grad(predh[g], xhat[g], grad_outputs=torch.ones(output.size()).cuda(), create_graph=True).norm(p=2, dim=0)-1).pow(2)
    # loss /= gs
    loss = torch.autograd.grad(predh, xhat, grad_outputs=torch.ones((predh.size(0), 1), device=self.device), create_graph=True)[0]
    # print(loss.size())
    loss = loss.view(-1, loss.size(1)*loss.size(2)*loss.size(3))
    loss = self.lam*(loss.norm(p=2, dim=1)-1).pow(2)
    # print(loss.size())
    # siz = list(xhat.size())
    # sm = 1
    # for s in siz[1:]:
    #   sm *= s
    # xhat = xhat.view(-1, sm)
    # print(xhat.size())
    # print(predh.size())
    # loss = torch.autograd.grad(predh, xhat)
    # print(loss.size())
    # loss = torch.autograd.grad(predh, xhat).norm(p=2, dim=1)
    # loss = loss - 1
    # loss = self.lam*(loss.pow(2))
    loss = predg - predt + loss
    return loss.mean()


class GeneratorLoss(nn.Module):
  """docstring for GeneratorLoss."""
  def __init__(self):
    super(GeneratorLoss, self).__init__()

  def forward(self, pred):
    return -1 * pred.mean()
