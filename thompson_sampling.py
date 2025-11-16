# this is the code for thompson sampling
# a good read:
# 1. https://www.geeksforgeeks.org/machine-learning/introduction-to-thompson-sampling-reinforcement-learning/
# 2. https://medium.com/@iqra.bismi/thompson-sampling-a-powerful-algorithm-for-multi-armed-bandit-problems-95c15f63a180
# 3. if you are seriously free: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf

# so here we need to store a history of actions and rewards
# and then use that history to select the action
# for gaussian rewards, we use normal posterior distribution instead of beta
# we sample from each arm's posterior (normal distribution with mean = sample mean, variance = 1/n)
# and then pick the arm with the highest sample (thompson sampling)

import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt

class ThompsonSamplingAgent:
    def __init__(self, n_actions):
        # super().__init__()
        self.n_actions = n_actions
        self.history = []
        self.sum_rewards = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.posterior_std = 1.0

    def select_action(self):
        samples = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            n = self.action_counts[a]
            if n == 0:
                samples[a] = 0.0
            else:
                mean = self.sum_rewards[a] / n
                var = self.posterior_std**2 / n
                samples[a] = np.random.normal(mean, np.sqrt(var))
        # Pick arm with highest sample (this balances exploration and exploitation)
        return np.argmax(samples)

    def update_estimates(self, action, reward):
        # Store history for histogram visualization
        self.history.append((action, reward))
        # Update count and sum for posterior calculation
        self.action_counts[action] += 1
        self.sum_rewards[action] += reward
        # No need for alpha/beta with normal posterior - conjugate update is implicit in select_action

def train_thompson_sampling_agent(env_name='BanditTenArmedGaussian-v0', n_episodes=1000):
    env = gym.make(env_name)
    env.reset()
    np.random.seed(42)
    n_actions = env.action_space.n
    agent = ThompsonSamplingAgent(n_actions)

    rewards = []
    cumulative_avg_rewards = []
    regrets = []
    cumulative_regret = []

    true_means = env.env.means if hasattr(env.env, 'means') else None
    if true_means is None:
        # fallback to direct env.means if not nested
        true_means = env.means if hasattr(env, 'means') else None
    if true_means is None:
        true_means = getattr(env, 'mu', None)
    if true_means is None:
        # if we can't get true means, we'll estimate optimal reward from observed rewards
        # this is a fallback - we'd have access to true means ideally
        print("Warning: Could not access true mean rewards. Regret will be estimated.")
        true_means = None

    optimal_reward = np.max(true_means) if true_means is not None else None

    for a in range(n_actions):
        reward = env.step(a)[1]
        agent.update_estimates(a, reward)
        rewards.append(reward)
        cumulative_avg = np.mean(rewards)
        cumulative_avg_rewards.append(cumulative_avg)
        if optimal_reward is not None:
            regret = optimal_reward - reward
        else:
            regret = 0
        regrets.append(regret)
        cumulative_regret.append(np.sum(regrets))

    for episode in range(n_actions, n_episodes):
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
    plt.title('Thompson Sampling Agent Performance')
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
    plt.savefig('thompson_sampling_agent_results.png')
    # plt.show()

    return agent, cumulative_regret[-1] if cumulative_regret else 0

if __name__ == "__main__":
    trained_agent, final_regret = train_thompson_sampling_agent(n_episodes=10000)
    print(f"Final history: {trained_agent.history}")
    print(f"Final cumulative regret: {final_regret:.2f}")