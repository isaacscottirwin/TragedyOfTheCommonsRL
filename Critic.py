import torch
import torch.nn as nn

class Critic(nn.Module):
    """
    Critic network for IAC.
    Takes an observation and outputs a single scalar value V(s).
    """
    def __init__(self, obs_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # value estimate
        )
    
    def forward(self, obs):
        """
        obs: Tensor of shape (batch, obs_dim)
        returns: Tensor of shape (batch, 1)
        """
        return self.net(obs)