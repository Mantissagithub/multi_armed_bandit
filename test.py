import gym
import gym_bandits

env = gym.make('BanditTenArmedGaussian-v0')
env.reset()


base_env = env.unwrapped
print("Dir of unwrapped env:", dir(base_env))

for attr in dir(base_env):
  if attr not startswith('_'):
    if atrr == 'means':
      print(getattr(base_env, attr))
    elif attr == 'mu':
      print(getattr(base_env, attr))
    else:
      try:
        base_env = base_env.attr
      except:
        pass
        