# so this file gonna contain the implementation of the gittins index for multi-armed bandit problem

# resources:
# 1. https://youtu.be/Wreegu4E7DY?si=iinpPtayTuvlz75J
# 2. https://youtu.be/p8AwKiudhZ4?si=6Gn7IM8LIm0dSSug
# 3. https://www.statslab.cam.ac.uk/~rrw1/oc/ocgittins.pdf

import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar

class GittinsIndexAgent:
    def __init__(self, n_actions, discount=0.95, epsilon=0.1, horizon=100):
      self.n_actions = n_actions
      self.discount = discount
      self.epsilon = epsilon
      self.horizon = horizon
      self.posteriors = [(1.0, 1.0) for _ in range(n_actions)]

      self.gittins_index = np.zeros(n_actions)
      self.action_counts = np.zeros(n_actions)
      self.t = 0
      self.history = []

    def compute_gittins_index(self, alpha, beta):
      n_states = self.horizon + 1

      def value_function(gamma):
        V = np.zeros(self.horizon + 1)
        for h in range(self.horizon-1, -1, -1):
          mu = alpha / (alpha + beta)
          exp_V = (alpha / (alpha + beta)) * V[h+1] + (beta / (alpha + beta)) * V[h+1]
          continuation = mu - gamma + self.discount * exp_V
          V[h] = max(0.0, continuation)
        return V[0]

      try:
        result = root_scalar(value_function, bracket=[0.0, 1.0])
        return result.root if result.converged else alpha / (alpha + beta)
      except Exception as e:
        print(f"Error computing Gittins index: {e}")
        return alpha / (alpha + beta)

    def select_action(self):
      self.t += 1
      if self.t <= self.n_actions:
        return self.t - 1

      return np.argmax(self.gittins_index)

    def update_estimates(self, action, reward):
      # store history for histogram visualization
      self.history.append((action, reward))
      alpha, beta = self.posteriors[action]
      if reward == 1:
        alpha += 1
      else:
        beta += 1
      self.posteriors[action] = (alpha, beta)
      self.gittins_index[action] = self.compute_gittins_index(alpha, beta)
      self.action_counts[action] += 1

def train_gittins_index_agent(env_name='BanditTenArmedGaussian-v0', n_episodes=1000):
  env = gym.make(env_name)
  n_actions = env.action_space.n
  agent = GittinsIndexAgent(n_actions)

  rewards = []
  cumulative_avg_rewards = []
  regrets = []
  cumulative_regret = []

  env.reset()
  np.random.seed(42)

  true_means = env.means if hasattr(env, 'means') else None
  if true_means is None:
      # try alternative attribute names
      true_means = getattr(env, 'mu', None)
  if true_means is None:
      # if we can't get true means, we'll estimate optimal reward from observed rewards
      # this is a fallback - we'd have access to true means ideally
      print("Warning: Could not access true mean rewards. Regret will be estimated.")
      true_means = None


  optimal_reward = np.max(true_means) if true_means is not None else None

  for episode in range(n_episodes):
    action = agent.select_action()
    reward = env.step(action)[1]
    agent.update_estimates(action, reward)

    rewards.append(reward)
    cumulative_avg = np.mean(rewards)
    cumulative_avg_rewards.append(cumulative_avg)

    # calculate regret
    if optimal_reward is not None:
      regret = optimal_reward - reward
    else:
      # fallback: use best observed reward as proxy for optimal
      regret = np.max(rewards) - reward if rewards else 0
    regrets.append(regret)
    cumulative_regret.append(np.sum(regrets))

  plt.figure(figsize=(18, 5))

  plt.subplot(1, 3, 1)
  plt.plot(rewards, alpha=0.3, label='Reward per episode')
  plt.plot(cumulative_avg_rewards, label='Cumulative average', linewidth=2)
  if optimal_reward is not None:
    plt.axhline(optimal_reward, 'r--', label='Optimal reward', linewidth=2)
  plt.xlabel('Episode')
  plt.ylabel('Reward')
  plt.title('Gittins Index Agent Performance')
  plt.legend()
  plt.grid(True)

  plt.subplot(1, 3, 2)
  plt.plot(cumulative_regret, label='Cumulative regret', linewidth=2)
  plt.xlabel('Episode')
  plt.ylabel('Cumulative Regret')
  plt.title('Cumulative Regret Over Time')
  plt.legend()
  plt.grid(True)

  plt.subplot(1, 3, 3)
  plt.hist([action for action, _ in agent.history], bins=n_actions, alpha=0.5)
  plt.xlabel('Action')
  plt.ylabel('Frequency')
  plt.title('Action Histogram')
  plt.grid(True)

  plt.tight_layout()
  plt.savefig('gittins_index_agent_results.png')
  # plt.show()

  plt.figure(figsize=(10, 6))
  time_steps = range(len(rewards))
  plt.plot(time_steps, rewards, linewidth=2, label='Reward', alpha=0.6)
  plt.xlabel('Time-steps')
  plt.ylabel('Reward')
  plt.title('Reward vs Time-steps - Gittins Index Agent')
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.savefig('results/gittins_index_total_regret.png')
  plt.close()

  return agent, cumulative_regret[-1] if cumulative_regret else 0

if __name__ == "__main__":
  agent, final_regret = train_gittins_index_agent(n_episodes=10000)
  print(f"Final regret: {final_regret}")