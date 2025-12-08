import matplotlib.pyplot as plt
import numpy as np

from CommonsEnv import CommonsEnv
from PpoAgent import PpoAgent

#HYPERPARAMETER CONFIG
DEFAULT_CONFIG = {
    "num_agents": 5,

    # Environment params
    "resource_max": 300,
    # More forgiving dynamics to avoid constant collapse
    "resource_regen_rate": 0.12,
    "max_extract": 4.0,
    # Allow any take to count toward reward
    "min_extract": .3,
    # Moderate horizon
    "max_steps": 200,
    # Penalties to discourage collapse, ramping in reasonably
    "collapse_penalty": 40.0,
    "penalty_ramp_episodes": 100,
    "penalty_scale": 0.8,
    # Modest end-of-episode bonus
    "scale_bonus": 40,

    # PPO params
    "learning_rate": 5e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_eps": 0.2,
    "train_iters": 5,

    # Training settings
    "num_episodes": 1500
}

CONFIG_MODE_BOOM_BUST = {
    "num_agents": 5,
    "resource_max": 300,
    "resource_regen_rate": 0.20,   # VERY high regen
    "max_extract": 6.0,            # agents can overharvest instantly
    "min_extract": 0.5,
    "max_steps": 200,
    "collapse_penalty": 30.0,
    "penalty_ramp_episodes": 50,
    "penalty_scale": 0.6,
    "scale_bonus": 30,

    "learning_rate": 5e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_eps": 0.2,
    "train_iters": 5,
    "num_episodes": 300
}

CONFIG_MODE_HARSH = {
    "num_agents": 5,
    "resource_max": 200,
    "resource_regen_rate": 0.04,   # very slow growth
    "max_extract": 3.0,
    "min_extract": 2.0,
    "max_steps": 700,
    "collapse_penalty": 100.0,     # falling below 1.0 is extremely costly
    "penalty_ramp_episodes": 600,  # agents learn slowly at first
    "penalty_scale": 1.0,
    "scale_bonus": 60,

    "learning_rate": 3e-4,
    "gamma": 0.9995,                # longer-term credit assignment
    "lambda": 0.97,
    "clip_eps": 0.15,
    "train_iters": 3,
    "num_episodes": 300
}

CONFIG_MODE_SHAPED = {
    "num_agents": 5,
    "resource_max": 300,
    "resource_regen_rate": 0.10,
    "max_extract": 4.0,
    "min_extract": 0.0,
    "max_steps": 250,
    "collapse_penalty": 30.0,
    "penalty_ramp_episodes": 30,
    "penalty_scale": 1.0,
    "scale_bonus": 80,     # strong horizon incentive

    "learning_rate": 5e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_eps": 0.2,
    "train_iters": 5,
    "num_episodes": 2500
}

CONFIG_MODE_CHAOS = {
    "num_agents": 5,
    "resource_max": 300,
    "resource_regen_rate": 0.12,
    "max_extract": 4.0,
    "min_extract": 0.3,
    "max_steps": 200,
    "collapse_penalty": 40.0,
    "penalty_ramp_episodes": 100,
    "penalty_scale": 0.8,
    "scale_bonus": 40,

    "learning_rate": 1e-3,     # too high
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_eps": 0.3,           # too permissive
    "train_iters": 40,         # over-updating
    "num_episodes": 300
}

CONFIG_MODE_SLOW = {
    "num_agents": 5,
    "resource_max": 300,
    "resource_regen_rate": 0.12,
    "max_extract": 4.0,
    "min_extract": 0.3,
    "max_steps": 1000,
    "collapse_penalty": 40.0,
    "penalty_ramp_episodes": 100,
    "penalty_scale": 0.8,
    "scale_bonus": 40,

    "learning_rate": 1e-4,   # slow learning
    "gamma": 0.98,
    "lambda": 0.95,
    "clip_eps": 0.1,        # tiny updates
    "train_iters": 3,
    "num_episodes": 300
}

CONFIG_MODE_GREED = {
    "num_agents": 5,
    "resource_max": 300,
    "resource_regen_rate": 0.10,
    "max_extract": 5.0,
    "min_extract": 0.0,
    "max_steps": 200,
    "collapse_penalty": 0.0,    # no penalty
    "penalty_ramp_episodes": 0,
    "penalty_scale": 0.0,
    "scale_bonus": 0.0,         # no horizon reward

    "learning_rate": 5e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_eps": 0.2,
    "train_iters": 5,
    "num_episodes": 200
}

CONFIG_MODE_ULTRA_STABLE = {
    "num_agents": 5,

    # Environment parameters
    "resource_max": 300,
    "resource_regen_rate": 0.10,   # strong enough growth for recovery, not too high
    "max_extract": 3.0,           # lower aggressiveness → fewer collapses
    "min_extract": 0.3,
    "max_steps": 250,

    "collapse_penalty": 80.0,     # strong discouragement
    "penalty_ramp_episodes": 200, # very slow ramp → early exploration, late stability
    "penalty_scale": 1.0,

    "scale_bonus": 100,           # VERY strong horizon reward → pushes sustainability

    # PPO parameters
    "learning_rate": 3e-4,        # slightly lower → smoother updates
    "gamma": 0.995,               # longer-term focus, but not extreme
    "lambda": 0.97,
    "clip_eps": 0.15,             # tighter constraint → avoids policy blowups
    "train_iters": 4,             # lower than 5 → prevents oscillatory over-updating

    # Training settings
    "num_episodes": 2000
}



# TRAINER CLASS
class CommonsTrainer:
    def __init__(self, config=CONFIG_MODE_GREED):
        self.config = config
        self.num_agents = config["num_agents"]

        # build environment + agents
        self.env = self._build_env()
        self.agents = self._build_agents()

        # storage for metrics
        self.resource_over_episodes = []
        self.agent_rewards = {i: [] for i in range(self.num_agents)}
        self.agent_actions = {i: [] for i in range(self.num_agents)}
        self.delta_R_per_episode = []
        self.total_reward_over_episodes = []

    # Environment + Agent Builders
    def _build_env(self):
        return CommonsEnv(
            num_agents=self.config["num_agents"],
            resource_max=self.config["resource_max"],
            resource_regen_rate=self.config["resource_regen_rate"],
            max_extract=self.config["max_extract"],
            min_extract=self.config["min_extract"],
            max_steps=self.config["max_steps"],
            collapse_penalty=self.config["collapse_penalty"],
            penalty_ramp_episodes=self.config["penalty_ramp_episodes"],
            penalty_scale=self.config["penalty_scale"],
            scale_bonus=self.config["scale_bonus"],
        )

    def _build_agents(self):
        agents = {}
        obs_dim = 3  # [R_norm, ΔR, prev_action]

        for i in range(self.num_agents):
            agents[i] = PpoAgent(
                obs_dim=obs_dim,
                lr=self.config["learning_rate"],
                gamma=self.config["gamma"],
                lam=self.config["lambda"],
                clip_eps=self.config["clip_eps"],
                train_iters=self.config["train_iters"]
            )
        return agents

    # Main Training Loop
    def train(self):
        num_episodes = self.config["num_episodes"]

        for ep in range(num_episodes):
            obs = self.env.reset(ep)
            done = False

            episode_reward = {i: 0 for i in range(self.num_agents)}
            episode_actions = {i: [] for i in range(self.num_agents)}
            start_resource = self.env.resource

            while not done:
                actions = {}
                step_info = {}

                # Each agent takes one action
                for i in self.agents:
                    act, raw, logp, val = self.agents[i].act(obs[i])
                    actions[i] = act
                    step_info[i] = (raw, logp, val)
                    episode_actions[i].append(act)

                # Step environment
                next_obs, rewards, done, _ = self.env.step(actions)

                # Store transitions
                for i in self.agents:
                    raw, logp, val = step_info[i]
                    self.agents[i].store(
                        obs=obs[i],
                        raw_action=raw,
                        logprob=logp,
                        reward=rewards[i],
                        value=val,
                        done=done
                    )
                    episode_reward[i] += rewards[i]

                obs = next_obs

            # End of episode: PPO update 
            for i in self.agents:
                self.agents[i].finish_episode()
                self.agent_rewards[i].append(episode_reward[i])
                self.agent_actions[i].append(episode_actions[i])

            final_resource = self.env.resource
            self.resource_over_episodes.append(final_resource)
            self.delta_R_per_episode.append(final_resource - start_resource)
            self.total_reward_over_episodes.append(sum(episode_reward.values()))

            print(f"Episode {ep} | Total Reward: {sum(episode_reward.values()):.2f} | Resource: {final_resource:.2f}")

    # PLOTTING HELPERS

    def plot_resources(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.resource_over_episodes, color="green", label="Resource")
        plt.xlabel("Episode")
        plt.ylabel("Resource Level")
        plt.title("Resource Over Episodes")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_rewards(self):
        plt.figure(figsize=(10, 5))
        for i in range(self.num_agents):
            plt.plot(self.agent_rewards[i], label=f"Agent {i}")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("Per-Agent Episode Rewards")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_delta_R(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.delta_R_per_episode, color="purple")
        plt.xlabel("Episode")
        plt.ylabel("ΔR")
        plt.title("Resource Change Per Episode")
        plt.grid(True)
        plt.show()

    def plot_average_actions(self):
        plt.figure(figsize=(10, 5))
        for i in range(self.num_agents):
            avg_a = [np.mean(ep) for ep in self.agent_actions[i]]
            plt.plot(avg_a, label=f"Agent {i}")
        plt.xlabel("Episode")
        plt.ylabel("Avg Action")
        plt.title("Average Extraction Per Agent")
        plt.grid(True)
        plt.legend()
        plt.show()

    def plot_action_correlation(self):
        data = np.array([[np.mean(ep) for ep in self.agent_actions[i]] for i in range(self.num_agents)])

        plt.figure(figsize=(6, 5))
        plt.imshow(np.corrcoef(data), cmap="viridis")
        plt.colorbar(label="Correlation")
        plt.title("Agent Action Correlation")
        plt.xticks(range(self.num_agents), [f"A{i}" for i in range(self.num_agents)])
        plt.yticks(range(self.num_agents), [f"A{i}" for i in range(self.num_agents)])
        plt.show()

    def plot_total_reward(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.total_reward_over_episodes, color="blue", label="Total Reward")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward (sum over agents)")
        plt.title("Total Reward Across Episodes")
        plt.grid(True)
        plt.legend()
        plt.show()    


# MAIN ENTRY POINT

if __name__ == "__main__":
    # Switch config here to control number of agents/behavior profile
    trainer = CommonsTrainer(DEFAULT_CONFIG)
    trainer.train()

    trainer.plot_resources()
    trainer.plot_rewards()
    trainer.plot_delta_R()
    trainer.plot_average_actions()
    trainer.plot_action_correlation()
    trainer.plot_total_reward()
