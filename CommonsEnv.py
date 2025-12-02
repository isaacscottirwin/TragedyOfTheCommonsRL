import numpy as np

class CommonsEnv:
    """
    An implementation of a Tragedy-of-the-Commons environment.

    - There is a single shared renewable resource.
    - Each agent chooses how much to extract each step.
    - Resource regenerates using logistic growth.
    - If agents extract too much → collapse.
    - Each agent receives reward = amount it extracts.
    - No explicit cooperation reward or penalty.
    """
    def __init__(self, num_agents=5, resource_max=100, resource_regen_rate=0.05, max_extract=5, max_steps=200):
        """
        Parameters:
            num_agents (int): number of agents in the environment.
            R_max (float): maximum capacity of the shared resource.
            regen_rate (float): logistic growth rate of the resource.
            max_extract (float): maximum a single agent can extract in a step.
            max_steps (int): hard cap on episode length to prevent endless loops.
        """
        self.num_agents = num_agents
        self.resource_max = resource_max
        self.resource_regen_rate = resource_regen_rate
        self.max_extract = max_extract
        self.max_steps = max_steps

        self.reset()

    def reset(self):
        """Reset the environment to its initial state."""
        self.resource = self.resource_max  # Start full
        self.step_count = 0
        # Return initial observations for each agent
        obs = {i: self._get_obs(i) for i in range(self.num_agents)}
        return obs

    def _get_obs(self, agent_id):
        """
        Observation for each agent.
        Currently: only the global resource fraction.
        """
        return np.array([self.resource / self.resource_max], dtype=np.float32)
    
    def step(self, actions):
        """
        Apply actions and update the environment.

        Parameters:
            actions (dict):
                Keys = agent IDs
                Values = action in [0,1] (fraction of max_extract)

        Returns:
            next_obs (dict): next observation per agent
            rewards (dict): reward per agent (equal to extraction)
            done (bool): episode termination flag
            info (dict): extra info (unused)
        """
        # Convert normalized actions to actual extraction amounts
        actions = {i: float(np.clip(actions[i], 0, 1)) * self.max_extract
            for i in actions}

        # calculate rewards
        rewards = {i: actions[i] for i in actions}

        #total resources taken
        total_taken = sum(actions.values())

        # update resource using logistic growth
        # logistic growth formula: resource = resource + regen_rate * resource *(1 - resource / resource_max) - total_taken
        # where resource is the current resource, regen_rate is the growth rate, and total_taken is the total resources taken
        self.resource = self.resource + self.resource_regen_rate * self.resource *(1 - self.resource / self.resource_max) - total_taken

        # clip resource to be between 0 and resource_max resource cannot be less than 0
        self.resource = np.clip(self.resource, 0, self.resource_max)

        self.step_count += 1

        # Episode ends if resource collapses or we hit the step limit (prevents infinite loops)
        done = bool(self.resource <= 0 or self.step_count >= self.max_steps)

        # Produce new observations
        next_obs = {i: self._get_obs(i) for i in actions}

        return next_obs, rewards, done, {}


