import torch
import numpy as np
import gc

from reversi_env import ReversiEnv

class RolloutFactory(object):
  """docstring for RolloutFactory."""
  def __init__(self, policy_net, env_samples, gamma=0.9):
    super(RolloutFactory, self).__init__()
    self.env = ReversiEnv()
    self.policy_net = policy_net
    self.env_samples = env_samples
    self.gamma = gamma
    self.device = torch.device("cpu")
    if torch.cuda.is_available():
      self.device = torch.device("cuda")


  def get_rollouts(self):
    env = self.env
    rollouts = []
    rounds = env.length()/2
    for p in self.policy_net.parameters():
      p.requires_grad = False
    for _ in range(self.env_samples):
      # don't forget to reset the environment at the beginning of each episode!
      # rollout for a certain number of steps!
      rollout_p1 = []
      rollout_p2 = []
      state = env.reset()
      actions = env.action_space()
      done = False
      for i in range(rounds):
        in_state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        probs, _ = self.policy_net(in_state)

        probs = probs.cpu().detach().numpy()
        np_probs = np.zeros(64)
        for a in actions:
          np_probs[a[0]*8 + a[1]] += probs[a[0]*8 + a[1]]
        np_probs /= np_probs.sum()
        action_one_hot = np.random.multinomial(1, np_probs)
        action_idx = np.argmax(action_one_hot)

        ay = action_idx // 8
        ax = action_idx % 8
        s_prime, reward, done = env.step((ay,ax))

        rollout_p1.append([state, np_probs, action_idx, reward])
        state = s_prime
        if done:
          rollout_p2[-1][-1] = reward * -1
          break

        in_state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)
        probs, _ = self.policy_net(in_state)

        probs = probs.cpu().detach().numpy()
        np_probs = np.zeros(64)
        for a in actions:
          np_probs[a[0]*8 + a[1]] += probs[a[0]*8 + a[1]]
        action_one_hot = np.random.multinomial(1, np_probs)
        action_idx = np.argmax(action_one_hot)

        ay = action_idx // 8
        ax = action_idx % 8
        s_prime, reward, done = env.step((ay,ax))

        rollout_p2.append([state, np_probs, action_idx, reward])
        state = s_prime
        if done:
          rollout_p1[-1][-1] = reward * -1
          break
      # calculate returns
      value = 0.0
      for i in reversed(range(len(rollout_p1))):
        value = rollout_p1[i][-1] + self.gamma * value
        rollout_p1[i].append(value)
      rollouts.append(rollout_p1)
      gc.collect()
      value = 0.0
      for i in reversed(range(len(rollout_p2))):
        value = rollout_p2[i][-1] + self.gamma * value
        rollout_p2[i].append(value)
      rollouts.append(rollout_p2)
      gc.collect()

    for p in self.policy_net.parameters():
      p.requires_grad = True
    return rollouts

if __name__ == '__main__':

  net = ReversiNet()
  factory = RolloutFactory(net, 1)
  rolls = factory.get_rollouts()
