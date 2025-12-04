import torch
import torch.optim as optim
import torch.nn.functional as F

from Actor import Actor
from Critic import Critic

class IacAgent:
    """
    Independent Actor-Critic agent.
    Each agent has:
        - Its own actor network (policy)
        - Its own critic network (value function)
        - Its own memory buffer for on-policy updates
    """
    def __init__(self, obs_dim, lr=1e-3, gamma=0.99):
        """
        Parameters:
            obs_dim (int): dimension of observation vector
            lr (float): learning rate for both actor and critic
            gamma (float): discount factor
        """
        self.gamma = gamma
        self.actor = Actor(obs_dim)
        self.critic = Critic(obs_dim)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)

        # storage for transitiions
        self.memory = []
        # fixed exploration noise for a simple stochastic policy
        self.action_std = 0.5

    def act(self, obs, explore=True):
        """
        Choose an action given an observation.

        obs: numpy array, shape (obs_dim,)
        returns: (clipped_action, raw_action_before_clipping)
        """
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean = self.actor(obs_tensor)

        if explore:
            dist = torch.distributions.Normal(mean, self.action_std)
            raw_action = dist.sample()
        else:
            raw_action = mean

        # The environment expects an action in [0, 1]
        clipped_action = torch.clamp(raw_action, 0.0, 1.0)

        return float(clipped_action.item()), float(raw_action.item())
    
    def store(self, obs, raw_action, reward, next_obs, done):
        """
        Store one transition (s, a_raw, r, s', done).
        We keep the raw (unclipped) action so we can compute log-probs consistently during learning.
        """
        self.memory.append((obs, raw_action, reward, next_obs, done))

    def learn(self):
        """
        Perform one round of actor-critic updates using all stored transitions.
        Called once per episode.
        """
        if not self.memory:
            return
        
        # Convert memory into tensors
        obs, raw_action, reward, next_obs, done = zip(*self.memory)
        obs = torch.tensor(obs, dtype=torch.float32)
        raw_action = torch.tensor(raw_action, dtype=torch.float32).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Compute discounted returns (advantages will use critic baseline)
        returns = []
        G = 0.0
        for r, d in zip(reversed(reward), reversed(done)):
            G = r + self.gamma * G * (1 - d)
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32)

        # Critic update: fit V(s) to returns
        value = self.critic(obs).squeeze()
        critic_loss = F.mse_loss(value, returns)

        # Actor update: policy gradient with baseline
        dist = torch.distributions.Normal(self.actor(obs), self.action_std)
        log_prob = dist.log_prob(raw_action).squeeze()
        entropy = dist.entropy().squeeze()
        advantage = (returns - value).detach()
        # Normalize advantage to keep updates stable across episodes
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        actor_loss = -(log_prob * advantage + 0.001 * entropy).mean()

        # Optimize
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Clear memory after update
        self.memory.clear()
