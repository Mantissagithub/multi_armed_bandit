# this is a good resource: https://colab.research.google.com/drive/1AgvnqbumrkPAFKI-Apt1SUtvbws4jVSS#scrollTo=sGDaa_u8fjO3

import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt

class GreedyAgent:
    def __init__(self, n_actions):
        # super().__init__()
        self.n_actions = n_actions
        self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)

    def select_action(self):
        return np.argmax(self.q_values)

    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n

def train_greedy_agent(env_name='BanditTenArmedGaussian-v0', n_episodes=1000):
    env = gym.make(env_name)
    n_actions = env.action_space.n
    agent = GreedyAgent(n_actions)

    rewards = []
    cumulative_avg_rewards = []

    env.reset()
    np.random.seed(42)

    for episode in range(n_episodes):
        action = agent.select_action()
        reward = env.step(action)[1]
        agent.update_estimates(action, reward)

        rewards.append(reward)
        cumulative_avg = np.mean(rewards)
        cumulative_avg_rewards.append(cumulative_avg)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.3, label='Reward per episode')
    plt.plot(cumulative_avg_rewards, label='Cumulative average', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Greedy Agent Performance')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.bar(range(n_actions), agent.q_values)
    plt.xlabel('Action')
    plt.ylabel('Estimated Q-value')
    plt.title('Final Q-value Estimates')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('greedy_agent_results.png')
    # plt.show()

    return agent

if __name__ == "__main__":
    trained_agent = train_greedy_agent(n_episodes=10000)
    print(f"Final q_values: {trained_agent.q_values}")
    print(f"Action counts: {trained_agent.action_counts}")