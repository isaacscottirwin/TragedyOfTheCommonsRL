import numpy as np

class CommonsEnv:
    """
    Tragedy of the Commons environment with:
    - Logistic growth
    - Extraction-based per-agent reward
    - Derivative-based shaping (changeR)
    - Observation = [R_norm, changeR, previous_action_i]
    - Asymmetric collapse penalty (worst offender penalized most)
    """

    def __init__(self, num_agents=5, resource_max=100, resource_regen_rate=0.05,
                 max_extract=10.0, min_extract=1.2, max_steps=200,
                 collapse_penalty=100.0, penalty_ramp_episodes=50,
                 penalty_scale=1.0, scale_bonus=30):

        self.num_agents = num_agents
        self.resource_max = resource_max
        self.resource_regen_rate = resource_regen_rate
        self.max_extract = max_extract
        self.min_extract = min_extract
        self.max_steps = max_steps

        self.collapse_penalty = collapse_penalty
        self.penalty_ramp_episodes = penalty_ramp_episodes
        self.penalty_scale = penalty_scale
        self.scale_bonus = scale_bonus

        # For derivative penalty
        self.prev_resource = None

        # Per-agent previous actions (stored in [0,1])
        self.prev_actions = np.zeros(self.num_agents, dtype=np.float32)

        self.reset()

    def reset(self, episode_idx=0):
        """Reset resource, penalties, and previous actions."""
        self.resource = self.resource_max
        self.prev_resource = self.resource
        self.step_count = 0

        # Reset previous actions
        self.prev_actions = np.zeros(self.num_agents, dtype=np.float32)

        # Ramp penalty early in training (lets them be greedy at first)
        if self.penalty_ramp_episodes > 0:
            self.penalty_scale = min(
                1.0, episode_idx / float(self.penalty_ramp_episodes)
            )
        else:
            self.penalty_scale = 1.0

        return {i: self._get_obs(i) for i in range(self.num_agents)}

    def _get_obs(self, agent_id):
        """
        Observation for each agent:
            [ normalized resource, changeR, previous_action_i ]
        """
        normalized_R = self.resource / self.resource_max
        delta_R = self.resource - self.prev_resource
        prev_a = self.prev_actions[agent_id]

        return np.array([normalized_R, delta_R, prev_a], dtype=np.float32)

    def _reward(self, actions, old_resource):
        """
        Compute per-agent rewards after the resource has been updated.
        """
        rewards = {}

        # Base extraction reward with "starvation" threshold:
        #    if you take less than min_extract, you get 0.
        for i in actions:
            if actions[i] > self.min_extract:
                rewards[i] = actions[i]
            else:
                rewards[i] = -self.max_extract

        # Derivative-based shaping on the *global* resource change
        delta_R = self.resource - old_resource

        if delta_R < 0:
            # Penalize resource decline (more negative if delta_R is more negative)
            for i in rewards:
                rewards[i] += 3.0 * delta_R
        else:
            # Small bonus for resource growth
            for i in rewards:
                rewards[i] += 1.0 * delta_R

        # Collapse check and asymmetric penalty
        collapsed = self.resource <= 1.0
        if collapsed:
            # Identify the biggest extractor this step
            offender = max(actions, key=lambda k: actions[k])

            # Offender gets 2x penalty; others get 1x penalty
            rewards = {
                i: (
                    -2.0 * self.penalty_scale * self.collapse_penalty
                    if i == offender
                    else -1.0 * self.penalty_scale * self.collapse_penalty
                )
                for i in actions
            }
            return rewards, True

        return rewards, False

    def _logistic_growth(self, total_taken):
        """ logistic growth with harvesting."""
        # Remove harvested amount
        R = self.resource - total_taken

        # Prevent negative values before growth is applied
        R = max(R, 0.0)

        # Logistic growth formula
        growth = self.resource_regen_rate * R * (1 - R / self.resource_max)
        R = R + growth

        # make sure resource is within bounds
        self.resource = np.clip(R, 0, self.resource_max)

    def step(self, actions):
        """
        One environment step.

        actions: dict {agent_id: scalar in [0,1]} (normalized extraction fraction)
        """
        # Convert normalized [0,1] -> actual extraction amount
        actions = {
            i: float(np.clip(actions[i], 0, 1)) * self.max_extract
            for i in actions
        }

        total_taken = sum(actions.values())
        old_resource = self.resource

        # Update resource with logistic growth
        self.prev_resource = self.resource
        self._logistic_growth(total_taken)

        # Update per-agent previous actions (store normalized [0,1])
        for i in actions:
            self.prev_actions[i] = actions[i] / self.max_extract

        # Compute rewards and collapse flag
        rewards, done = self._reward(actions, old_resource)

        # Horizon termination + end-of-episode resource bonus
        if not done and self.step_count >= self.max_steps:
            done = True
            resource_bonus = (self.resource / self.resource_max) * self.scale_bonus
            rewards = {
                i: rewards[i] + self.penalty_scale * resource_bonus
                for i in rewards
            }

        # Next observations
        next_obs = {i: self._get_obs(i) for i in actions}

        self.step_count += 1
        return next_obs, rewards, done, {}