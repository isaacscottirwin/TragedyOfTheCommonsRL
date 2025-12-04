import numpy as np

from TocEnv import TocEnv
from Learning import Qlearning, Sarsa
from Visualizations import plot_survival, plot_resource

# Set a fixed seed using a dedicated RNG and pass it everywhere
SEED = 42
GLOBAL_RNG = np.random.default_rng(SEED)
env = TocEnv(rng=GLOBAL_RNG)

# train Q-learning
q_train = Qlearning(env, rng=GLOBAL_RNG)
Q_q, q_rewards, q_resource = q_train.train()

# train SARSA
sarsa_train = Sarsa(env, rng=GLOBAL_RNG)
Q_s, sarsa_rewards, sarsa_resource = sarsa_train.train()

# plot survival curves
plot_survival(q_rewards, sarsa_rewards)

plot_resource(q_resource, sarsa_resource)
