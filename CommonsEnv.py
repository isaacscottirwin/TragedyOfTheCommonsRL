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
    def __init__(self, num_agents=5, resource_max=100, resource_regen_rate=0.05, max_extract=5, max_steps=200,
                 collapse_penalty=50.0, scarcity_threshold=0.3, scarcity_penalty=5.0, penalty_ramp_episodes=50):
        """
        Parameters:
            num_agents (int): number of agents in the environment.
            R_max (float): maximum capacity of the shared resource.
            regen_rate (float): logistic growth rate of the resource.
            max_extract (float): maximum a single agent can extract in a step.
            max_steps (int): hard cap on episode length to prevent endless loops.
            collapse_penalty (float): negative reward applied to all agents on collapse.
            scarcity_threshold (float): fraction of capacity below which extra penalty is applied each step.
            scarcity_penalty (float): scale of penalty when resource is below the threshold.
            penalty_ramp_episodes (int): number of episodes over which penalties ramp from 0 to full strength.
        """
        self.num_agents = num_agents
        self.resource_max = resource_max
        self.resource_regen_rate = resource_regen_rate
        self.max_extract = max_extract
        self.max_steps = max_steps
        self.collapse_penalty = collapse_penalty
        self.scarcity_threshold = scarcity_threshold
        self.scarcity_penalty = scarcity_penalty
        self.penalty_ramp_episodes = penalty_ramp_episodes
        self.penalty_scale = 1.0

        self.reset()

    def reset(self, episode_idx=0):
        """Reset the environment to its initial state."""
        self.resource = self.resource_max  # Start full
        self.step_count = 0
        # Linearly ramp penalty strength over initial episodes to allow early over-exploitation
        if self.penalty_ramp_episodes > 0:
            self.penalty_scale = min(1.0, episode_idx / float(self.penalty_ramp_episodes))
        else:
            self.penalty_scale = 1.0
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

        # Per-step penalty when the resource is scarce (shapes the reward away from greed)
        resource_frac = self.resource / self.resource_max
        if resource_frac < self.scarcity_threshold:
            scarcity_cost = self.penalty_scale * self.scarcity_penalty * (self.scarcity_threshold - resource_frac)
            rewards = {i: r - scarcity_cost for i, r in rewards.items()}

        # major penalty if resource goes to 0
        collapsed = self.resource <= 0
        if collapsed:
            rewards = {i: -self.penalty_scale * self.collapse_penalty for i in rewards}

        self.step_count += 1

        # Episode ends if resource collapses or we hit the step limit (prevents infinite loops)
        done = bool(collapsed or self.step_count >= self.max_steps)

        # Produce new observations
        next_obs = {i: self._get_obs(i) for i in actions}

        return next_obs, rewards, done, {}
