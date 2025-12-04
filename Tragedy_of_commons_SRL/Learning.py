import numpy as np
from collections import defaultdict

def discretize_state(obs, env, num_buckets = 10):
    """
    Convert continuous observation into a discrete state tuple.
    Ex: if number of buckets is 10 and current resource level is 87.5 
    and max resource is 100 then ratio is .875, then it will 
    be put into 8 bucket

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
    def __init__(self, env, episodes=7000, max_steps = 300, eta=0.5, gamma=0.99,
                 epsilon=0.1, epsilon_decay=0.9995, epsilon_min=0.01,
                 num_buckets=10):
        self.env = env
        self.episodes = episodes
        self.max_steps = max_steps
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_buckets = num_buckets
        
        self.q_values = defaultdict(lambda: np.zeros(env.num_actions))
        self.rewards_per_episode = []
        self.resource_per_episode = []

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.num_actions)
        return np.argmax(self.q_values[state])

    def train(self):
        for episode in range(self.episodes):
            obs = self.env.reset()
            state = discretize_state(obs, self.env, self.num_buckets)
            action = self.choose_action(state)

            done = False
            total_reward = 0
            resource_sum = 0
            steps = 0

            while not done or steps <= self.max_steps:
                next_obs, reward, done = self.env.step(action)
                total_reward += reward
                resource_sum += next_obs[0]
                steps += 1

                next_state = discretize_state(next_obs, self.env, self.num_buckets)
                next_action = self.choose_action(next_state)

                # SARSA update
                delta = reward + self.gamma * self.q_values[next_state][next_action] \
                        - self.q_values[state][action]
                self.q_values[state][action] += self.eta * delta

                state = next_state
                action = next_action

            # anneal epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            self.rewards_per_episode.append(total_reward)
            self.resource_per_episode.append(resource_sum / steps)

        return self.q_values, self.rewards_per_episode, self.resource_per_episode

class Qlearning:
    def __init__(self, env, episodes=7000, max_steps = 300, eta=0.5, gamma=0.99,
                 epsilon=0.1, epsilon_decay=0.9995, epsilon_min=0.01,
                 num_buckets=10):
        self.env = env
        self.episodes = episodes
        self.max_steps = max_steps 
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.num_buckets = num_buckets

        self.q_values = defaultdict(lambda: np.zeros(env.num_actions))
        self.rewards_per_episode = []
        self.resource_per_episode = []

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.num_actions)
        return np.argmax(self.q_values[state])

    def train(self):
        for episode in range(self.episodes):
            obs = self.env.reset()
            state = discretize_state(obs, self.env, self.num_buckets)

            done = False
            total_reward = 0
            resource_sum = 0
            steps = 0

            while not done or steps <= self.max_steps:
                action = self.choose_action(state)
                next_obs, reward, done = self.env.step(action)
                total_reward += reward
                resource_sum += next_obs[0]
                steps += 1

                next_state = discretize_state(next_obs, self.env, self.num_buckets)

                # Q-learning update (off-policy)
                delta = reward + self.gamma * np.max(self.q_values[next_state]) \
                        - self.q_values[state][action]
                self.q_values[state][action] += self.eta * delta

                state = next_state

            # epsilon annealing
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            self.rewards_per_episode.append(total_reward)
            self.resource_per_episode.append(resource_sum / steps)

        return self.q_values, self.rewards_per_episode, self.resource_per_episode






