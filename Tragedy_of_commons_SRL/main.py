from TocEnv import TocEnv
from Learning import Qlearning, Sarsa
from Visualizations import plot_survival, plot_policy_heatmap, plot_resource
env = TocEnv()

# train Q-learning
q_train = Qlearning(env)
Q_q, q_rewards, q_resource = q_train.train()

# train SARSA
sarsa_train = Sarsa(env)
Q_s, sarsa_rewards, sarsa_resource = sarsa_train.train()

# plot survival curves
plot_survival(q_rewards, sarsa_rewards)

plot_resource(q_resource, sarsa_resource)
