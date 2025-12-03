import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from ActorCritic import ActorCritic

class PpoAgent:
    """
    Independent PPO Agent (IPPO-style).
    Each agent has its own PPO instance.
    
    Key points:
    - Stochastic Gaussian policy in R, squashed to [0,1] via sigmoid for the env.
    - We compute log-probs in the *un-squashed* space (standard trick).
    - We store transitions per episode, then run PPO updates on that batch.
    """

    def __init__(self,obs_dim,lr=5e-4,gamma=0.99,lam=0.95,clip_eps=0.2,train_iters=10):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.train_iters = train_iters
        self.ac = ActorCritic(obs_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

        # Buffer for one trajectory (one episode)
        self.reset_buffer()

    def reset_buffer(self):
        self.obs_buf = []
        self.act_raw_buf = []   # un-squashed action from Normal
        self.logp_buf = []
        self.rew_buf = []
        self.done_buf = []
        self.val_buf = []
    
    def act(self, obs):
        """
        Given a single observation (np.array), return:
        - action in [0,1] to feed to env
        - raw_action in R (for PPO logprob)
        - logprob
        - value estimate
        """
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mu, v, log_std = self.ac(obs_t)
            std = torch.exp(log_std)
            dist = Normal(mu, std)
            raw_action = dist.sample()          # in R
            logprob = dist.log_prob(raw_action) # logprob in R-space

        # Squash to [0,1] for env
        action = torch.sigmoid(raw_action).item()

        return action, raw_action.item(), logprob.item(), v.item()
    
    def store(self, obs, raw_action, logprob, reward, value, done):
        self.obs_buf.append(obs)
        self.act_raw_buf.append(raw_action)
        self.logp_buf.append(logprob)
        self.rew_buf.append(reward)
        self.done_buf.append(done)
        self.val_buf.append(value)
    
    def finish_episode(self):
        """
        Compute returns and advantages from the episode,
        then perform PPO updates and clear the buffer.
        """
        # Convert to tensors
        obs = torch.tensor(self.obs_buf, dtype=torch.float32)
        act_raw = torch.tensor(self.act_raw_buf, dtype=torch.float32).unsqueeze(-1)
        logp_old = torch.tensor(self.logp_buf, dtype=torch.float32).unsqueeze(-1)
        rew = np.array(self.rew_buf, dtype=np.float32)
        done = np.array(self.done_buf, dtype=np.float32)
        val = np.array(self.val_buf, dtype=np.float32)

        # compute returns
        returns = []
        adv = []
        gae = 0.0
        G = 0.0
        #  treat done=True as terminal
        next_val = 0.0
        for t in reversed(range(len(rew))):
            if done[t]:
                next_val = 0.0
                G = 0.0
                gae = 0.0

            delta = rew[t] + self.gamma * next_val - val[t]
            gae = delta + self.gamma * self.lam * gae
            G = rew[t] + self.gamma * G
            next_val = val[t]

            adv.insert(0, gae)
            returns.insert(0, G)

        adv = torch.tensor(adv, dtype=torch.float32).unsqueeze(-1)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)
        val_t = torch.tensor(val, dtype=torch.float32).unsqueeze(-1)

        # normalize
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        #ppo update
        for _ in range(self.train_iters):
            mu, v_pred, log_std = self.ac(obs)
            std = torch.exp(log_std)
            dist = Normal(mu, std)
            logp = dist.log_prob(act_raw)

            # PPO ratio
            ratio = torch.exp(logp - logp_old)

            # Clipped surrogate objective
            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
            actor_loss = -torch.mean(torch.min(unclipped, clipped))

            # Value loss
            critic_loss = F.mse_loss(v_pred, returns)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss  # 0.5 weight on value loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear buffers
        self.reset_buffer()



