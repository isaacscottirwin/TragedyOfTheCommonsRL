import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, obs_dim) -> None:
        super().__init__()
        hidden = 64

        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Policy head: outputs mean of a 1D Gaussian
        self.mu_head = nn.Linear(hidden, 1)

        # Value head
        self.v_head = nn.Linear(hidden, 1)
        # Learnable log standard deviation (state-independent)
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        """
        obs: (batch, obs_dim)
        returns: mu (batch, 1), value (batch, 1), log_std (1)
        """
        x = self.body(obs)
        mu = self.mu_head(x)
        value = self.v_head(x)
        return mu, value, self.log_std

        
