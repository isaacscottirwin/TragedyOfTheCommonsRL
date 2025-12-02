import matplotlib.pyplot as plt
from CommonsEnv import CommonsEnv
from IacAgent import IacAgent

hyperparameters = {"learning_rate": 5e-3, "resource_regen_rate": 0.08,
                 "resource_max": 150, "max_extract": 1, "gamma": 0.99,
                 "max_steps": 10}

def train():
    num_agents = 5
    env = CommonsEnv(num_agents=num_agents, resource_max=hyperparameters.get("resource_max"), 
        resource_regen_rate=hyperparameters.get("resource_regen_rate"), max_extract=hyperparameters.get("max_extract"), max_steps=hyperparameters.get("max_steps"))

    # Create an IAC agent for each agent in the environment
    agents = {i: IacAgent(obs_dim=1, lr = hyperparameters.get("learning_rate"), 
            gamma=hyperparameters.get("gamma")) for i in range(num_agents)}

    num_episodes = 300
    resource_over_episodes = []
    agent_reward_per_episode = {i: [] for i in range(num_agents)}


    for ep in range(num_episodes):
        obs = env.reset()
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

if __name__ == "__main__":
    resource_over_episodes, agent_reward_per_episode = train()
    plot_resource_pool(resource_over_episodes)
    plot_reward_per_agent(agent_reward_per_episode)
