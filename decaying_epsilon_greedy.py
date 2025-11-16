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

    def select_action(self, epsilon):
        if np.random.rand() < epsilon:
            return np.random.randint(self.n_actions)
        else:
            return np.argmax(self.q_values)

    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        n = self.action_counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n

def train_decaying_epsilon_greedy_agent(env_name='BanditTenArmedGaussian-v0', n_episodes=1000):
    env = gym.make(env_name)
    n_actions = env.action_space.n
    agent = GreedyAgent(n_actions)

    rewards = []
    cumulative_avg_rewards = []
    regrets = []
    cumulative_regret = []

    env.reset()
    np.random.seed(42)

    epsilon = 0.1
    decay_rate = 0.01

    true_means = env.means if hasattr(env, 'means') else None
    if true_means is None:
        true_means = getattr(env, 'mu', None)
    if true_means is None:
        # if we can't get true means, we'll estimate optimal reward from observed rewards
        # this is a fallback - we'd have access to true means ideally
        print("Warning: Could not access true mean rewards. Regret will be estimated.")
        true_means = None

    optimal_reward = np.max(true_means) if true_means is not None else None

    for episode in range(n_episodes):
        if episode > 1:
          epsilon = max(0.01, epsilon * (1 - decay_rate))
        action = agent.select_action(epsilon)
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
        plt.axhline(y=optimal_reward, color='r', linestyle='--', label='Optimal reward', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Decaying Epsilon-Greedy Agent Performance')
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
    plt.bar(range(n_actions), agent.q_values)
    plt.xlabel('Action')
    plt.ylabel('Estimated Q-value')
    plt.title('Final Q-value Estimates')
    if true_means is not None:
        plt.plot(range(n_actions), true_means, 'ro', label='True means', markersize=8)
        plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('decaying_epsilon_greedy_agent_results.png')
    # plt.show()

    return agent, cumulative_regret[-1] if cumulative_regret else 0

if __name__ == "__main__":
    trained_agent, final_regret = train_decaying_epsilon_greedy_agent(n_episodes=10000)
    print(f"Final q_values: {trained_agent.q_values}")
    print(f"Action counts: {trained_agent.action_counts}")
    print(f"Final cumulative regret: {final_regret:.2f}")