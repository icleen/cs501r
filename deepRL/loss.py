import torch
import torch.nn as nn

class PPOLoss(nn.Module):
  """docstring for PPOLoss."""
  def __init__(self, epsilon):
    super(PPOLoss, self).__init__()
    self.epsilon = epsilon

  def forward(self, ratio, advantage):
    loss = ratio * advantage
    clip = ratio.clamp(1-self.epsilon, 1+self.epsilon) * advantage
    loss = torch.min(loss, clip)
    return loss.mean()
