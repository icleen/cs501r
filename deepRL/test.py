import gym
env = gym.make("MountainCar-v0")
observation = env.reset()
print('state: {}, action space: {}'.format(observation, env.action_space))
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  observation, reward, done, info = env.step(action)
