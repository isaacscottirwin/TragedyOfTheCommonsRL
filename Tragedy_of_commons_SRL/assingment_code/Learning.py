import numpy as np
from collections import defaultdict

def discretize_state(obs, env, num_buckets = 10):
    """
    Convert continuous observation into a discrete state tuple.
    Ex: if number of buckets is 10 and current resource level is 87.5 
    and max resource is 100 then ratio is .875, then it will 
    be put into 8 bucket

    allows for the use of sarsa and q-learning as the oberservation
    space is now discrete

    the action space is already discrete

    obs = [resource, num_agents_before]
    """
    resource = obs[0]
    num_before = int(obs[1])

    resource_frac = resource / env.resource_max
    resource_bucket = int(resource_frac * num_buckets)

    if resource_bucket >= num_buckets:
        resource_bucket = num_buckets - 1
    
    return (resource_bucket, num_before)
    


class Sarsa:
    def __init__(self, env, episodes=10000, max_steps=600, eta=0.04, gamma=0.99,
                 epsilon=0.02, epsilon_decay=0.995, epsilon_min=0.005,
                 num_buckets=15, rng=None):
        """
        feel free to play with these parameters, these are the parameters that gave me good results
        """

        self.env = env
        self.episodes = episodes
        self.max_steps = max_steps
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_buckets = num_buckets
        self.rng = rng if rng is not None else np.random.default_rng()
        
        self.q_values = defaultdict(lambda: np.zeros(env.num_actions))
        self.rewards_per_episode = []
        self.resource_per_episode = []

    def choose_action(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.env.num_actions)
        return np.argmax(self.q_values[state])

    def train(self):
        ## todo implement sarsa algorithm

        return self.q_values, self.rewards_per_episode, self.resource_per_episode

class Qlearning:
    def __init__(self, env, episodes=10000, max_steps=300, eta=0.04, gamma=0.99,
                 epsilon=0.05, epsilon_decay=0.99, epsilon_min=0.01,
                 num_buckets=15, rng=None):
        """
        feel free to play with these parameters, these are the parameters that gave me good results
        """
        self.env = env
        self.episodes = episodes
        self.max_steps = max_steps 
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_buckets = num_buckets
        self.rng = rng if rng is not None else np.random.default_rng()

        self.q_values = defaultdict(lambda: np.zeros(env.num_actions))
        self.rewards_per_episode = []
        self.resource_per_episode = []

    def choose_action(self, state):
        if self.rng.random() < self.epsilon:
            return self.rng.integers(self.env.num_actions)
        return np.argmax(self.q_values[state])

    def train(self):
        ## todo implement Q-learning algorithm

        return self.q_values, self.rewards_per_episode, self.resource_per_episode
