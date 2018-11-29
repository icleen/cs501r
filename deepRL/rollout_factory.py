import torch
import gym
import numpy as np
import gc

class RolloutFactory(object):
  """docstring for RolloutFactory."""
  def __init__(self, env, envname, policy_net, env_samples, episode_length, gamma):
    super(RolloutFactory, self).__init__()
    self.env = env
    self.policy_net = policy_net
    self.env_samples = env_samples
    self.episode_length = episode_length
    self.gamma = gamma
    self.device = torch.device("cpu")
    if torch.cuda.is_available():
      self.device = torch.device("cuda")

    self.avg_reward = []
    self.mtcar = (envname == 'MountainCar-v0')

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
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        probs = self.policy_net(state).squeeze()
        probs_np = probs.cpu().detach().numpy()
        action_one_hot = np.random.multinomial(1, probs_np)
        action = np.argmax(action_one_hot)
        tp = [state.squeeze().cpu(), probs.cpu(), torch.LongTensor([action]).cpu()]
        # next_state, reward, done, info
        state, reward, done, _ = env.step(action)
        if self.mtcar and reward < 0:
          reward = -state[1]
        avg_rw += reward
        tp.append(torch.FloatTensor([reward]).cpu())
        rollout.append(tp)
        if not self.mtcar and done:
          break
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
