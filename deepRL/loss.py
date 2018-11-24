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
    return -1*loss.mean()
    # lhs = ratio * advantage
    # rhs = torch.clamp(ratio, ppo_lower_bound, ppo_upper_bound) * advantage
    # policy_loss = -torch.mean(torch.min(lhs, rhs))
