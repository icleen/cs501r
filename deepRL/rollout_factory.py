import torch
import gym
import numpy as np
import gc

class RolloutFactory(object):
  """docstring for RolloutFactory."""
  def __init__(self, env, envname, policy_net, env_samples, episode_length, gamma, cutearly=True):
    super(RolloutFactory, self).__init__()
    self.env = env
    self.policy_net = policy_net
    self.env_samples = env_samples
    self.episode_length = episode_length
    self.gamma = gamma
    self.device = torch.device("cpu")
    self.cutearly = cutearly
    if torch.cuda.is_available():
      self.device = torch.device("cuda")

    self.avg_reward = []

  def get_rollouts(self):
    env = self.env
    rollouts = []
    avg_rw = 0.0
    for p in self.policy_net.parameters():
      p.requires_grad = False
    for _ in range(self.env_samples):
      # don't forget to reset the environment at the beginning of each episode!
      # rollout for a certain number of steps!
      rollout = []
      state = env.reset()
      done = False
      for i in range(self.episode_length):
        in_state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        probs, action = self.policy_net(in_state)
        probs, action = probs[0], action[0]  # Remove the batch dimension
        s_prime, reward, done, _ = env.step(action.item())

        rollout.append([state, probs.cpu().detach().numpy(), action, reward])
        if self.cutearly and done:
          break
        state = s_prime
      # calculate returns
      value = 0.0
      for i in reversed(range(len(rollout))):
        value = rollout[i][-1] + self.gamma * value
        rollout[i].append(value)
      rollouts.append(rollout)
      gc.collect()

    for p in self.policy_net.parameters():
      p.requires_grad = True
    # self.avg_reward.append(avg_rw / self.env_samples)
    # print('avg standing time:', self.avg_reward[-1])
    return rollouts

class RolloutFactory2D(object):
  """docstring for RolloutFactory2D."""
  def __init__(self, env, envname, policy_net, env_samples, episode_length, gamma):
    super(RolloutFactory2D, self).__init__()
    self.env = env
    self.policy_net = policy_net
    self.env_samples = env_samples
    self.episode_length = episode_length
    self.gamma = gamma
    self.device = torch.device("cpu")
    if torch.cuda.is_available():
      self.device = torch.device("cuda")

    self.avg_reward = []

  def get_rollouts(self):
    env = self.env
    rollouts = []
    avg_rw = 0.0
    for p in self.policy_net.parameters():
      p.requires_grad = False
    for _ in range(self.env_samples):
      # don't forget to reset the environment at the beginning of each episode!
      # rollout for a certain number of steps!
      rollout = []
      state = env.reset()
      done = False
      for i in range(self.episode_length):
        state = torch.FloatTensor(np.rollaxis(state,2,0)).unsqueeze(0).to(self.device)
        probs, _ = self.policy_net(state)
        probs = probs.squeeze().cpu()
        probs_np = probs.detach().numpy()
        action_one_hot = np.random.multinomial(1, probs_np)
        action = np.argmax(action_one_hot)
        tp = [state.squeeze().cpu(), probs, torch.LongTensor([action])]
        # next_state, reward, done, info
        state, reward, done, _ = env.step(action)
        avg_rw += reward
        tp.append(torch.FloatTensor([reward]))
        rollout.append(tp)
        gc.collect()
      value = 0.0
      for i in reversed(range(len(rollout))):
        value = rollout[i][-1] + self.gamma * value
        rollout[i].append(value.cpu())
      rollouts.append(rollout)
      gc.collect()

    for p in self.policy_net.parameters():
      p.requires_grad = True
    self.avg_reward.append(avg_rw / self.env_samples)
    # print('avg standing time:', self.avg_reward[-1])
    return rollouts
