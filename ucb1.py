# this is a good resource: https://colab.research.google.com/drive/1AgvnqbumrkPAFKI-Apt1SUtvbws4jVSS#scrollTo=sGDaa_u8fjO3

import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt

class UCB1Agent:
    def __init__(self, n_actions):
        # super().__init__()
        self.n_actions = n_actions
        # self.q_values = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.total_rewards = np.zeros(n_actions)
        self.avg_rewards = np.zeros(n_actions)
        self.t = 0

    def select_action(self):
        self.t += 1
        if self.t <= self.n_actions:
          return self.t - 1

        ucb_values = self.avg_rewards + np.sqrt(2 * np.log(self.t) / self.action_counts)
        return np.argmax(ucb_values)

    def update_estimates(self, action, reward):
        self.action_counts[action] += 1
        self.total_rewards[action] += reward
        self.avg_rewards[action] = self.total_rewards[action] / self.action_counts[action]

def train_ucb1_agent(env_name='BanditTenArmedGaussian-v0', n_episodes=1000):
    env = gym.make(env_name)
    n_actions = env.action_space.n
    agent = UCB1Agent(n_actions)

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
        plt.axhline(y=optimal_reward, color='r', linestyle='--', label='Optimal reward', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('UCB1 Agent Performance')
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
    plt.bar(range(n_actions), agent.avg_rewards, alpha=0.5)
    plt.xlabel('Action')
    plt.ylabel('Estimated Avg Reward')
    plt.title('Final Avg Reward Estimates')
    if true_means is not None:
        plt.plot(range(n_actions), true_means, 'ro', label='True means', markersize=8, alpha=0.5)
        plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('ucb1_agent_results.png')
    # plt.show()

    plt.figure(figsize=(10, 6))
    time_steps = range(len(cumulative_regret))
    plt.plot(time_steps, cumulative_regret, linewidth=2, label='Total Regret')
    plt.xlabel('Time-steps')
    plt.ylabel('Total Regret')
    plt.title('Total Regret vs Time-steps - UCB1 Agent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/ucb1_total_regret.png')
    plt.close()

    return agent, cumulative_regret[-1] if cumulative_regret else 0

if __name__ == "__main__":
    trained_agent, final_regret = train_ucb1_agent(n_episodes=10000)
    print(f"Final avg_rewards: {trained_agent.avg_rewards}")
    print(f"Action counts: {trained_agent.action_counts}")
    print(f"Final cumulative regret: {final_regret:.2f}")