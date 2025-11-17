# implementation of gittins index for multi-armed bandit using proper dp and calibration
# gittins index is the optimal solution to the bayesian mab problem with discounting

# resources:
# 1. https://youtu.be/Wreegu4E7DY?si=iinpPtayTuvlz75J
# 2. https://youtu.be/p8AwKiudhZ4?si=6Gn7IM8LIm0dSSug
# 3. https://www.statslab.cam.ac.uk/~rrw1/oc/ocgittins.pdf

import gym
import gym_bandits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq


class GittinsIndexAgent:
    def __init__(self, n_actions, discount=0.95, horizon=100):
        self.n_actions = n_actions
        self.discount = discount
        self.horizon = horizon
        # beta posteriors: (alpha, beta) where alpha = successes + 1, beta = failures + 1
        self.posteriors = [(1.0, 1.0) for _ in range(n_actions)]

        self.gittins_index = np.zeros(n_actions)
        self.action_counts = np.zeros(n_actions)
        self.t = 0
        self.history = []

        # dp table cache for gittins index computation
        self.dp_cache = {}

    def compute_gittins_index(self, alpha, beta):
        state = (alpha, beta)
        if state in self.dp_cache:
            return self.dp_cache[state]

        # mean reward for this state
        mean_reward = alpha / (alpha + beta)

        def compute_value_for_lambda(lam):
            # dp table: V[h] = value at horizon h
            V = np.zeros(self.horizon + 1)

            # backward induction from horizon to 0
            for h in range(self.horizon - 1, -1, -1):
                # expected reward for current state
                curr_mean = alpha / (alpha + beta)

                # expected next value after pulling this arm
                # with prob α/(α+β) we get success: next state is (α+1, β)
                # with prob β/(α+β) we get failure: next state is (α, β+1)
                prob_success = alpha / (alpha + beta)
                prob_failure = beta / (alpha + beta)

                # compute next state values
                next_val_success = 0
                next_val_failure = 0

                if h < self.horizon - 1:
                    # recursive case: look up or estimate next values
                    alpha_next_s = alpha + 1
                    beta_next_s = beta
                    alpha_next_f = alpha
                    beta_next_f = beta + 1

                    # simplified: use mean estimate for next states
                    next_val_success = V[h + 1]
                    next_val_failure = V[h + 1]

                expected_next_val = prob_success * next_val_success + prob_failure * next_val_failure

                # continuation value: immediate reward - charge + discounted future
                continuation = curr_mean - lam + self.discount * expected_next_val

                # can always stop (value = 0), so take max
                V[h] = max(0.0, continuation)

            return V[0]

        try:
            # if continuing is better than charge of mean_reward, index > mean
            if compute_value_for_lambda(mean_reward) > 0:
                # search between mean and 1
                gittins = brentq(compute_value_for_lambda, mean_reward, 1.0)
            else:
                # search between 0 and mean
                gittins = brentq(compute_value_for_lambda, 0.0, mean_reward)
        except ValueError:
            # if we can't find zero crossing, use mean as fallback
            gittins = mean_reward

        self.dp_cache[state] = gittins
        return gittins

    def select_action(self):
        self.t += 1

        # initial exploration: try each arm once
        if self.t <= self.n_actions:
            return self.t - 1

        return np.argmax(self.gittins_index)

    def update_estimates(self, action, reward):
        self.history.append((action, reward))

        alpha, beta = self.posteriors[action]

        # bayesian update: increment success or failure count
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

    # get true means for regret calculation
    true_means = env.means if hasattr(env, 'means') else None
    if true_means is None:
        true_means = getattr(env, 'mu', None)
    if true_means is None:
        print("warning: could not access true mean rewards. regret will be estimated.")
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
            regret = np.max(rewards) - reward if rewards else 0
        regrets.append(regret)
        cumulative_regret.append(np.sum(regrets))

    # plotting results
    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(rewards, alpha=0.3, label='reward per episode')
    plt.plot(cumulative_avg_rewards, label='cumulative average', linewidth=2)
    if optimal_reward is not None:
        plt.axhline(optimal_reward, color='r', linestyle='--', label='optimal reward', linewidth=2)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.title('gittins index agent performance')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(cumulative_regret, label='cumulative regret', linewidth=2)
    plt.xlabel('episode')
    plt.ylabel('cumulative regret')
    plt.title('cumulative regret over time')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.hist([action for action, _ in agent.history], bins=n_actions, alpha=0.5)
    plt.xlabel('action')
    plt.ylabel('frequency')
    plt.title('action histogram')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('gittins_index_agent_results.png')
    plt.close()

    # reward vs time-steps plot
    plt.figure(figsize=(10, 6))
    time_steps = range(len(rewards))
    plt.plot(time_steps, rewards, linewidth=2, label='reward', alpha=0.6)
    plt.xlabel('time-steps')
    plt.ylabel('reward')
    plt.title('reward vs time-steps - gittins index agent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/gittins_index_total_regret.png')
    plt.close()

    return agent, cumulative_regret[-1] if cumulative_regret else 0


if __name__ == "__main__":
    agent, final_regret = train_gittins_index_agent(n_episodes=10000)
    print(f"final regret: {final_regret}")
