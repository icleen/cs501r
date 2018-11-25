import torch
import torch.nn as nn

class PPOLoss(nn.Module):
  """docstring for PPOLoss."""
  def __init__(self, epsilon):
    super(PPOLoss, self).__init__()
    self.ppo_low_bnd = 1 - epsilon
    self.ppo_up_bnd = 1 + epsilon

  def forward(self, ratio, advantage):
    loss = ratio * advantage
    clip = torch.clamp(ratio, self.ppo_low_bnd, self.ppo_up_bnd) * advantage
    return -torch.mean(torch.min(loss, clip))
