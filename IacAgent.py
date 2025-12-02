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

    def act(self, obs):
        """
        Choose an action given an observation.

        obs: numpy array, shape (obs_dim,)
        returns: float in [0, 1]
        """
        obs_tensor = torch.tensor(obs,dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_tensor).item()
        
        return float(action)
    
    def store(self, obs, action, reward, next_obs, done):
        """
        Store one transition (s, a, r, s', done)
        """
        self.memory.append((obs, action, reward, next_obs, done))

    def learn(self):
        """
        Perform one round of actor-critic updates using all stored transitions.
        Called once per episode.
        """
        if not self.memory:
            return
        
        # Convert memory into tensors
        obs, action, reward, next_obs, done = zip(*self.memory)
        obs = torch.tensor(obs, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32).unsqueeze(1)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        # Compute value targets
        with torch.no_grad():
            target = reward + self.gamma * self.critic(next_obs).squeeze() * (1 - done)

        # Critic update
        value = self.critic(obs).squeeze()
        critic_loss = F.mse_loss(value, target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update
        advantage = (target - value).detach()
        pred_action = self.actor(obs)

        # Squared difference policy loss
        actor_loss = torch.mean(advantage * (pred_action - action)**2)

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Clear memory after update
        self.memory.clear()

