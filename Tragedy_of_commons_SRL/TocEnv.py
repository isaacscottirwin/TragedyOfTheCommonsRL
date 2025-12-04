import numpy as np


class TocEnv:
    """
    Pure Python ToC environment (discrete actions):
      - One RL agent
      - K non-agents
      - Random turn order each step
      - Agent sees only: resource and non-agent takes BEFORE it acts
      - Regeneration occurs after each step
      - Discrete action = index selecting fraction of max_agent_take
    """

    def __init__(
        self,
        K=3,
        resource_max=100,
        regen_rate=1.2,
        max_agent_take=5,
        max_nonagent_take=3,
        survival_min_take=1,
        num_actions=5,
    ):

        self.K = K
        self.resource_max = resource_max
        self.regen_rate = regen_rate
        self.max_agent_take = max_agent_take
        self.max_nonagent_take = max_nonagent_take
        self.survival_min_take = survival_min_take

        if num_actions < 2:
            raise ValueError("num_actions must be >= 2")

        self.num_actions = num_actions
        self.action_fracs = np.linspace(0.0, 1.0, num_actions)

        self.reset()

    def reset(self):
        self.resource = float(self.resource_max)
        self.prev_nonagent_takes = []
        return self._get_obs()

    def _action(self, action_index):
        """Convert discrete action index to extraction amount."""
        if not (0 <= action_index < self.num_actions):
            raise ValueError(f"Action {action_index} is out of range.")

        return self.action_fracs[action_index] * self.max_agent_take

    def _reward(self, agent_action = None):
        """
        Computes the reward and returns (reward, done).
        - If agent_action < survival_min_take → starvation
        - If resource <= 0 → collapse
        - Otherwise survival reward
        """
        if agent_action is not None:
            if agent_action < self.survival_min_take:
                return -1, True # starved/died

        if self.resource <= 0:
            return -1, True # env collapsed

        return +1, False  # survived step
    
    def logistic_regen(self):
        return self.regen_rate * self.resource * (1 - self.resource/self.resource_max)

    def step(self, action_index):
        """
        action_index: integer 0 .. num_actions-1
        """

        # random order
        order = list(range(self.K + 1))
        np.random.shuffle(order)

        # visible non-agent takes (before agent acts)
        self.prev_nonagent_takes = []

        # process non-agents before agent
        for p in order:
            if p == self.K:
                break

            take = float(np.random.uniform(0, self.max_nonagent_take))
            self.prev_nonagent_takes.append(take)

            self.resource -= take
            if self.resource <= 0:
                reward, done = self._reward()
                return self._end_step(reward, done)

        # agent acts
        action = self._action(action_index)

        self.resource -= action
        if self.resource <= 0:
            reward , done = self._reward(agent_action=action)
            return self._end_step(reward, done)

        # process non-agents after agent
        after_agent = False
        for p in order:
            if p == self.K:
                after_agent = True
                continue
            if after_agent:
                take = float(np.random.uniform(0, self.max_nonagent_take))
                self.resource -= take
                if self.resource <= 0:
                    reward, done = self._reward()
                    return self._end_step(reward, done)

        # regenerate resource (simple)
        growth = self.logistic_regen()
        self.resource += growth
        if self.resource > self.resource_max:
            self.resource = self.resource_max

        # final reward
        reward, done = self._reward(agent_action=action)
        return self._end_step(reward, done)

    def _end_step(self, reward, done):
        return self._get_obs(), reward, done

    def _get_obs(self):
        return [self.resource] + self.prev_nonagent_takes.copy()

    def render(self):
        print(f"Resource={self.resource:.2f}, visible takes={self.prev_nonagent_takes}")
