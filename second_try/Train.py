import matplotlib.pyplot as plt
import numpy as np
from CommonsEnv import CommonsEnv
from IacAgent import IacAgent
from PpoAgent import PpoAgent

hyperparameters = {
    "resource_max": 200,
    "resource_regen_rate": 0.05, 
    "max_extract": 1.5, 
    "max_steps": 300,
    "collapse_penalty": 90.0,
    "penalty_ramp_episodes": 50, 
    "penalty_scale": 1.0,
    "scale_bonus": 60,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "lambda": 0.95,
    "clip_eps": 0.2,
    "train_iters": 10
    
}

def trainIac():
    num_agents = 5
    env = CommonsEnv(
        num_agents=num_agents,
        resource_max=hyperparameters.get("resource_max"),
        resource_regen_rate=hyperparameters.get("resource_regen_rate"),
        max_extract=hyperparameters.get("max_extract"),
        max_steps=hyperparameters.get("max_steps"),
        collapse_penalty=hyperparameters.get("collapse_penalty"),
        penalty_ramp_episodes=hyperparameters.get("penalty_ramp_episodes"),
        penalty_scale=hyperparameters.get("penalty_scale"),
        scale_bonus=hyperparameters.get("scale_bonus")

        
    )

    # Create an IAC agent for each agent in the environment
    agents = {i: IacAgent(obs_dim=1, lr = hyperparameters.get("learning_rate"), 
            gamma=hyperparameters.get("gamma")) for i in range(num_agents)}

    num_episodes = 200
    resource_over_episodes = []
    agent_reward_per_episode = {i: [] for i in range(num_agents)}


    for ep in range(num_episodes):
        obs = env.reset(ep)
        done = False
        total_reward = 0

        # temp reward accumulator for this episode
        episode_reward = {i: 0 for i in range(num_agents)}

        while not done:
            # Each agent chooses its action
            actions = {}
            raw_actions = {}
            for i in agents:
                act_out = agents[i].act(obs[i])  # exploration is default in the agent
                if isinstance(act_out, tuple):
                    act, raw_act = act_out
                else:
                    # backward compatibility: older act returned only a scalar
                    act, raw_act = float(act_out), float(act_out)
                actions[i] = act
                raw_actions[i] = raw_act

            # Environment step
            next_obs, rewards, done, _ = env.step(actions)

            # Store transitions
            for i in agents:
                agents[i].store(
                    obs[i], raw_actions[i], rewards[i], next_obs[i], done
                )
                # accumulate reward for this episode
                episode_reward[i] += rewards[i]

                total_reward += rewards[i]

            obs = next_obs

        # After each episode, update all agents
        for i in agents:
            agents[i].learn()
            agent_reward_per_episode[i].append(episode_reward[i])
        

        resource_over_episodes.append(env.resource)
        print(f"Episode {ep} | Total reward: {total_reward:.2f} | Resource: {env.resource:.2f}")
    return resource_over_episodes, agent_reward_per_episode


def trainPpo():
    num_agents = 5
    env = CommonsEnv(
        num_agents=num_agents,
        resource_max=hyperparameters.get("resource_max"),
        resource_regen_rate=hyperparameters.get("resource_regen_rate"),
        max_extract=hyperparameters.get("max_extract"),
        max_steps=hyperparameters.get("max_steps"),
        collapse_penalty=hyperparameters.get("collapse_penalty"),
        penalty_ramp_episodes=hyperparameters.get("penalty_ramp_episodes"),
        penalty_scale=hyperparameters.get("penalty_scale"),
        scale_bonus=hyperparameters.get("scale_bonus")
    )
    agents = {i: PpoAgent(
                    obs_dim=1,
                    lr=hyperparameters.get("learning_rate"),
                    gamma=hyperparameters.get("gamma"),
                    lam=hyperparameters.get("lambda"),
                    clip_eps=hyperparameters.get("clip_eps"),
                    train_iters=hyperparameters.get("train_iters")
                    )
              for i in range(num_agents)}

    num_episodes = 200
    resource_over_episodes = []
    agent_reward_per_episode = {i: [] for i in range(num_agents)}
    agent_actions_per_episode = {i: [] for i in agents}
    delta_R_per_episode = []

    for ep in range(num_episodes):
        obs = env.reset(ep)
        done = False
        total_reward = 0
        episode_reward = {i: 0 for i in range(num_agents)}
        episode_actions = {i: [] for i in range(num_agents)}
        start_resource = env.resource

        while not done:
            actions = {}
            step_info = {}

            for i in agents:
                act, raw_act, logp, val = agents[i].act(obs[i])
                actions[i] = act
                step_info[i] = (raw_act, logp, val)
                episode_actions[i].append(act)

            next_obs, rewards, done, _ = env.step(actions)

            for i in agents:
                raw_act, logp, val = step_info[i]
                agents[i].store(
                    obs=obs[i],
                    raw_action=raw_act,
                    logprob=logp,
                    reward=rewards[i],
                    value=val,
                    done=done
                )
                episode_reward[i] += rewards[i]
                total_reward += rewards[i]

            obs = next_obs

        for i in agents:
            agents[i].finish_episode()
            agent_reward_per_episode[i].append(episode_reward[i])
            agent_actions_per_episode[i].append(episode_actions[i])

        resource_over_episodes.append(env.resource)
        delta_R = env.resource - start_resource
        delta_R_per_episode.append(delta_R)
        print(f"Episode {ep} | Total reward: {total_reward:.2f} | Resource: {env.resource:.2f}")

    return resource_over_episodes, agent_reward_per_episode, delta_R_per_episode, agent_actions_per_episode

    


def plot_resource_pool(resource_over_episodes):
    plt.figure(figsize=(8,4))
    plt.plot(resource_over_episodes, label="Resource Level", color='green')
    plt.xlabel("Episode")
    plt.ylabel("Resource Level")
    plt.title("Resource Level at End of Each Episode")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_reward_per_agent(agent_reward_per_episode):
    plt.figure(figsize=(10,5))

    num_agents = len(agent_reward_per_episode)

    for agent_id in range(num_agents):
        plt.plot(
            agent_reward_per_episode[agent_id],
            label=f"Agent {agent_id}",
            linewidth=2
        )

    plt.xlabel("Episode")
    plt.ylabel("Total Reward Per Episode")
    plt.title("Reward per Agent across Episodes")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_delta_resource(delta_R_per_episode):
    plt.figure(figsize=(8,4))
    plt.plot(delta_R_per_episode, color="purple", linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("ΔR")
    plt.title("Resource Derivative per Episode")
    plt.grid(True)
    plt.show()

def plot_reward_histograms(agent_reward_per_episode):
    num_agents = len(agent_reward_per_episode)
    plt.figure(figsize=(12,8))

    for agent_id in range(num_agents):
        plt.hist(agent_reward_per_episode[agent_id], bins=40, alpha=0.5, label=f"Agent {agent_id}")

    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution per Agent")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_agent_correlation(agent_actions_per_episode):
    # Convert to numpy matrix (agents × episodes)
    data = []
    for ag in agent_actions_per_episode:
        avg_actions = [sum(ep)/len(ep) for ep in agent_actions_per_episode[ag]]
        data.append(avg_actions)

    data = np.array(data)   # shape (N_agents, N_episodes)

    plt.figure(figsize=(6,5))
    plt.imshow(np.corrcoef(data), cmap="viridis")
    plt.colorbar(label="Correlation")
    plt.title("Correlation of Extraction Behavior Between Agents")
    plt.xticks(range(data.shape[0]), [f"A{i}" for i in range(data.shape[0])])
    plt.yticks(range(data.shape[0]), [f"A{i}" for i in range(data.shape[0])])
    plt.show()    

def plot_average_extraction(agent_actions_per_episode):
    plt.figure(figsize=(10,5))
    num_agents = len(agent_actions_per_episode)

    for agent_id in range(num_agents):
        avg_actions = [sum(ep)/len(ep) for ep in agent_actions_per_episode[agent_id]]
        plt.plot(avg_actions, label=f"Agent {agent_id}", linewidth=2)

    plt.xlabel("Episode")
    plt.ylabel("Average Action (0–1)")
    plt.title("Average Extraction Fraction per Agent per Episode")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    resource_over_episodes, agent_reward_per_episode, delta_R_per_episode, agent_actions_per_episode = trainPpo()
    plot_resource_pool(resource_over_episodes)
    plot_reward_per_agent(agent_reward_per_episode)
    plot_delta_resource(delta_R_per_episode)
    plot_average_extraction(agent_actions_per_episode)
    plot_agent_correlation(agent_actions_per_episode)
    plot_reward_histograms(agent_reward_per_episode)
