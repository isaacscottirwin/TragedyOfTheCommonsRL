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
                 collapse_penalty=100.0, penalty_ramp_episodes=50, penalty_scale = 1.0, scale_bonus = 30):
        """
        # Parameters:
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
        self.penalty_ramp_episodes = penalty_ramp_episodes
        self.penalty_scale = penalty_scale
        self.scale_bonus = scale_bonus
        self.prev_resource = None

        self.reset()

    def reset(self, episode_idx=0):
        """Reset the environment to its initial state."""
        self.resource = self.resource_max  # Start full
        self.step_count = 0
        self.prev_resource = self.resource
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

        # Convert actions 0-1 -> extraction amount
        actions = {i: float(np.clip(actions[i], 0, 1)) * self.max_extract
                for i in actions}

        # Base reward = extraction
        rewards = {i: actions[i] for i in actions}
        
        total_taken = sum(actions.values())

        # Logistic growth
        self.resource = (
            self.resource
            + self.resource_regen_rate * self.resource * (1 - self.resource / self.resource_max)
            - total_taken
        )
        self.resource = np.clip(self.resource, 0, self.resource_max)

        # compute resource derivative
        delta_R = self.resource - self.prev_resource
        self.prev_resource = self.resource   

        if delta_R < 0:
            for i in rewards:
                rewards[i] += 3.0 * delta_R

        # did evn collapse
        collapsed = self.resource <=1.0

        if collapsed:
            # Immediate collapse penalty; no extraction reward allowed
            rewards = {i: -self.penalty_scale * self.collapse_penalty for i in actions}
            return {i: self._get_obs(i) for i in actions}, rewards, True, {}
        
        # increase step
        self.step_count += 1
        reached_hoizon = self.step_count >= self.max_steps    
        
        # if episode has reached horizon
        if reached_hoizon:
            resource_bonus = float(self.resource/self.resource_max)

            rewards = { i: rewards[i] + self.scale_bonus * self.penalty_scale * resource_bonus
                        for i in rewards}
            done = True
            next_obs = {i : self._get_obs(i) for i in actions}
            return next_obs, rewards, done, {}

        # otherwise continue
        done = False
        next_obs = {i: self._get_obs(i) for i in actions}
        return next_obs, rewards, done, {}
