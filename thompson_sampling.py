# this is the code for thompson sampling
# a good read:
# 1. https://www.geeksforgeeks.org/machine-learning/introduction-to-thompson-sampling-reinforcement-learning/
# 2. https://medium.com/@iqra.bismi/thompson-sampling-a-powerful-algorithm-for-multi-armed-bandit-problems-95c15f63a180
# 3. if you are seriously free: https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf

# so here we need to store a history of actions and rewards
# and then use that history to select the action
# we need to use the beta distribution to sample the action
# and then update the history with the reward
# and then use the history to select the action
# and then update the history with the reward

import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt

class ThompsonSamplingAgent:
    def __init__(self, n_actions):
        # super().__init__()
        self.n_actions = n_actions
        self.history = []
        self.alpha = np.ones(n_actions)
        self.beta = np.ones(n_actions)
        # For normalizing rewards to [0, 1] range for beta distribution
        self.min_reward = None
        self.max_reward = None

    def select_action(self):
        return np.argmax(np.random.beta(self.alpha, self.beta))

    def normalize_reward(self, reward):
        """Normalize reward to [0, 1] range for beta distribution."""
        # Initialize min/max on first reward
        if self.min_reward is None:
            self.min_reward = reward
            self.max_reward = reward

        # Update min/max
        self.min_reward = min(self.min_reward, reward)
        self.max_reward = max(self.max_reward, reward)

        # Normalize to [0, 1] with small epsilon to avoid edge cases
        if self.max_reward == self.min_reward:
            return 0.5  # Return middle value if all rewards are the same
        normalized = (reward - self.min_reward) / (self.max_reward - self.min_reward)
        # Clip to [0, 1] to be safe
        return np.clip(normalized, 0.0, 1.0)

    def update_estimates(self, action, reward):
        self.history.append((action, reward))
        normalized_reward = self.normalize_reward(reward)
        self.alpha[action] += normalized_reward
        self.beta[action] += 1 - normalized_reward
        # Ensure alpha and beta stay positive (shouldn't happen with normalization, but safety check)
        self.alpha[action] = max(self.alpha[action], 0.01)
        self.beta[action] = max(self.beta[action], 0.01)

def train_thompson_sampling_agent(env_name='BanditTenArmedGaussian-v0', n_episodes=1000):
    env = gym.make(env_name)
    # env.seed(42)
    env.reset()
    np.random.seed(42)
    n_actions = env.action_space.n
    agent = ThompsonSamplingAgent(n_actions)

    rewards = []
    cumulative_avg_rewards = []
    regrets = []
    cumulative_regret = []

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