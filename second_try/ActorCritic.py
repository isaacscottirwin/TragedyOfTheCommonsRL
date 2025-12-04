import torch
import torch.nn as nn
from torch.distributions import Normal

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
        # Learnable log standard deviation (scalar)
        # Broadcast automatically across batch dimension
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, obs):
        """
        Forward pass.

        Inputs:
            obs: Tensor of shape (batch, obs_dim)

        Outputs:
            mu:      (batch, 1) — Gaussian mean
            value:   (batch, 1) — State value estimate
            log_std: (1,)       — shared log standard deviation
        """
        x = self.body(obs)
        mu = self.mu_head(x)
        value = self.v_head(x)

        return mu, value, self.log_std

        
