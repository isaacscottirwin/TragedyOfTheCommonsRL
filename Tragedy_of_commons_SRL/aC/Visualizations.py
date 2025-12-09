import matplotlib.pyplot as plt
import numpy as np


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
    plt.figure(figsize=(12, 4))
    plt.plot(q_resource, label="Q-learning End Resource", alpha=0.9, color="tab:blue")
    plt.title("Ending Resource Level Per Episode (Q-learning)")
    plt.xlabel("Episode")
    plt.ylabel("End Resource")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.plot(sarsa_resource, label="SARSA End Resource", alpha=0.9, color="tab:orange")
    plt.title("Ending Resource Level Per Episode (SARSA)")
    plt.xlabel("Episode")
    plt.ylabel("End Resource")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend()
    plt.show()
