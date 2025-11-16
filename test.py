import gym
import gym_bandits

env = gym.make('BanditTenArmedGaussian-v0')
env.reset()

print(str(env.unwrapped['BanditTenArmedGaussian']))