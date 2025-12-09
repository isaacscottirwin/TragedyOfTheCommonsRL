import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from ActorCritic import ActorCritic

class PpoAgent:
    def __init__(self, obs_dim, lr=5e-4, gamma=0.99, lam=0.95, clip_eps=0.2, train_iters=10):
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.train_iters = train_iters
        self.ac = ActorCritic(obs_dim)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)

        self.reset_buffer()

    def reset_buffer(self):
        self.obs_buf = []
        self.act_raw_buf = []
        self.logp_buf = []
        self.reward_buf = []
        self.done_buf = []
        self.value_buf = []

    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mu, v, log_std = self.ac(obs_t)
            std = torch.exp(log_std).expand_as(mu)
            dist = Normal(mu, std)
            raw_action = dist.sample()
            logprob = dist.log_prob(raw_action)

        action = torch.sigmoid(raw_action).item()

        return (
            action,
            raw_action.squeeze().item(),
            logprob.squeeze().detach(),   
            v.squeeze().detach()          
        )

    def store(self, obs, raw_action, logprob, reward, value, done):
        self.obs_buf.append(np.array(obs, dtype=np.float32))
        self.act_raw_buf.append(raw_action)
        self.logp_buf.append(logprob.detach())
        self.reward_buf.append(reward)
        self.done_buf.append(done)
        self.value_buf.append(value.detach())

    def finish_episode(self):
        obs = torch.tensor(self.obs_buf, dtype=torch.float32)
        act_raw = torch.tensor(self.act_raw_buf, dtype=torch.float32).unsqueeze(-1)

        logp_old = torch.stack(self.logp_buf).unsqueeze(-1)
        val = torch.stack(self.value_buf).squeeze().numpy()

        reward = np.array(self.reward_buf, dtype=np.float32)
        done = np.array(self.done_buf, dtype=np.float32)

        # returns: Monte-Carlo returns G_t = r_t + gamma r_{t+1} + gamma^2 r_{t+2} + ...
        returns = []

        # adv: Advantage estimates A_t, computed using GAE (Generalized Advantage Estimation)
        #      Tells PPO how much better/worse an action was compared to baseline V(s)
        adv = []

        # gae: Running GAE accumulator (lambda-discounted TD-residual smoothing)
        #      gae_t = delta_t + gamma*lambda * gae_{t+1}
        #      This reduces variance vs pure returns, but keeps bias low.
        gae = 0.0

        # G: Running Monte-Carlo return accumulator
        #    G_t = r_t + gamma * G_{t+1}
        #    Used as the regression target for the critic V(s)
        G = 0.0

        # next_val: V(s_{t+1}) used in TD residual
        #    Reset to 0 when hitting done =True (terminal state)
        next_val = 0.0

        for t in reversed(range(len(reward))):
            if done[t]:
                next_val = 0.0
                G = 0.0
                gae = 0.0

            delta = reward[t] + self.gamma * next_val - val[t]
            gae = delta + self.gamma * self.lam * gae
            G = reward[t] + self.gamma * G
            next_val = val[t]

            adv.insert(0, gae)
            returns.insert(0, G)

        adv = torch.tensor(adv, dtype=torch.float32).unsqueeze(-1)
        returns = torch.tensor(returns, dtype=torch.float32).unsqueeze(-1)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # PPO update 
        entropy_coef = 0.01

        for _ in range(self.train_iters):
            mu, v_pred, log_std = self.ac(obs)
            std = torch.exp(log_std).expand_as(mu)
            dist = Normal(mu, std)

            logp = dist.log_prob(act_raw)
            ratio = torch.exp(logp - logp_old)

            unclipped = ratio * adv
            clipped = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv

            actor_loss = -torch.mean(torch.min(unclipped, clipped))
            critic_loss = F.mse_loss(v_pred, returns)

            entropy = dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss - entropy_coef * entropy

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), 0.5)
            self.optimizer.step()

        self.reset_buffer()