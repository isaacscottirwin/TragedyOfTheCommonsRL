import matplotlib.pyplot as plt
import numpy as np

def plot_policy_heatmap(Q, env, num_buckets=10):
    """
    Visualizes the learned policy as a heatmap.
    
    x-axis: resource bucket (0..num_buckets-1)
    y-axis: num_agents_before (0..K)
    value: best action at that state
    """

    # policy grid (rows=num_before, cols=buckets)
    policy = np.zeros((env.K + 1, num_buckets))

    for b in range(num_buckets):
        for num_before in range(env.K + 1):
            state = (b, num_before)
            if state in Q:
                policy[num_before, b] = np.argmax(Q[state])
            else:
                policy[num_before, b] = -1  # unknown / never visited

    plt.figure(figsize=(10, 5))
    plt.imshow(policy, aspect='auto', cmap='viridis', origin='lower')
    plt.colorbar(label="Action index")
    plt.xlabel("Resource Bucket (0 = low, 9 = high)")
    plt.ylabel("Num Non-Agents Before Agent")
    plt.title("Learned Policy Heatmap")
    plt.xticks(range(num_buckets))
    plt.yticks(range(env.K + 1))
    plt.show()

def plot_survival(q_rewards, sarsa_rewards, window=50, beta=0.99):
    def smooth(x, w):
        if len(x) < w: return x
        return np.convolve(x, np.ones(w)/w, mode='valid')

    def ewma(x, beta):
        ew = []
        s = 0
        for r in x:
            s = beta * s + (1 - beta) * r
            ew.append(s)
        return ew

    q_smooth = smooth(q_rewards, window)
    sarsa_smooth = smooth(sarsa_rewards, window)

    q_ewma = ewma(q_rewards, beta)
    sarsa_ewma = ewma(sarsa_rewards, beta)

    plt.figure(figsize=(12, 5))
    plt.plot(q_ewma, label="Q-learning (EWMA)", linewidth=2)
    plt.plot(sarsa_ewma, label="SARSA (EWMA)", linewidth=2)
    plt.title("Smoothed Survival Reward Over Training")
    plt.xlabel("Episode")
    plt.ylabel("EWMA Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()

def plot_resource(q_resource, sarsa_resource):
    plt.figure(figsize=(12, 5))
    plt.plot(q_resource, label="Q-learning Avg Resource", alpha=0.9)
    plt.plot(sarsa_resource, label="SARSA Avg Resource", alpha=0.9)

    plt.title("Average Resource Level Per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Avg Resource")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.show()