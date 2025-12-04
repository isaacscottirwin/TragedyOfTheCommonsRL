import torch
import torch.nn as nn

class Actor(nn.Module):
    """
    Actor network for Independent Actor-Critic (IAC).
    Takes an observation and outputs an action in [0, 1].

    The environment later scales this action to [0, max_extract].
    """
    
    def __init__(self, obs_dim):
        super().__init__()
        # Simple fully connected policy network
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),    # one continuous action
            nn.Sigmoid()         # restrict to [0, 1]
        )

    def forward(self, obs):
        """
        obs: Tensor of shape (batch, obs_dim)
        returns: Tensor of shape (batch, 1)
        """
        return self.net(obs)

    

