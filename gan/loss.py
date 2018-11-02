import torch

class DescriminatorLoss(nn.Module):
  """docstring for DescriminatorLoss."""
  def __init__(self, lam):
    super(DescriminatorLoss, self).__init__()
    self.lam = lam

  def forward(self, predg, predt, predh, xhat):
    loss = torch.autograd.grad(predh, xhat)
    print(loss.size())
    loss = torch.autograd.grad(predh, xhat).norm(p=2, dim=1)
    loss = loss - 1
    loss = self.lam*(loss.pow(2))
    loss = predg - predt + loss
    return loss.mean()


class GeneratorLoss(nn.Module):
  """docstring for GeneratorLoss."""
  def __init__(self):
    super(GeneratorLoss, self).__init__()

  def forward(self, pred):
    return -1 * pred.mean()
