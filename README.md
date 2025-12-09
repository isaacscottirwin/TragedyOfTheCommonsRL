# Tragedy of the Commons — Multi-Agent Reinforcement Learning (IPPO + PPO)

This project explores whether independent reinforcement learning (RL) agents can learn sustainable behavior in a Tragedy of the Commons (ToC) environment using only reward and environmental feedback. Agents share a renewable resource, extract from it, and indirectly influence each other through the environment — no communication or coordination.

------------------------------------------------------------
Project Purpose
------------------------------------------------------------

Real-world ToC systems (fisheries, forests, groundwater, emissions) show how individually rational behavior can lead to collective collapse.

This project asks:

    Can independent PPO agents learn to avoid collapse and maintain a sustainable equilibrium?

The project includes:
- A renewable-resource environment with logistic growth
- Independent PPO (IPPO) agents
- Reward shaping to encourage sustainability
- Visualization tools to analyze collapse and equilibrium behavior

------------------------------------------------------------
Installation
------------------------------------------------------------

1. Clone the repository:

    git clone https://github.com/isaacscottirwin/TragedyOfTheCommonsRL.git
    

2. Create a virtual environment (optional):

    python3 -m venv .venv
    source .venv/bin/activate        (macOS / Linux)
    .venv\Scripts\activate           (Windows)

3. Install dependencies:

    pip install -r requirements.txt

------------------------------------------------------------
Running the Project
------------------------------------------------------------

To train the agents, run:

    python Train.py

This will:
- Instantiate the CommonsEnv
- Create one PPO agent per learner
- Train for a configurable number of episodes
- Print per-episode total reward and final resource level
- Generate plots such as:
    • Resource level over time
    • Per-agent rewards
    • ΔResource per episode
    • Average extraction per agent
    • Correlation between agents’ actions

------------------------------------------------------------
Configuration Parameters
------------------------------------------------------------

Defined in DEFAULT_CONFIG inside Train.py.

Environment parameters:
- num_agents: Number of learning agents
- resource_max: Carrying capacity of the shared resource
- resource_regen_rate: Regeneration rate for logistic growth
- max_extract: Maximum extraction per step
- min_extract: Minimum extraction before a penalty
- max_steps: Episode horizon
- collapse_penalty: Penalty when the resource collapses
- penalty_ramp_episodes: Gradually increases penalty strength
- penalty_scale: Scales all penalties/bonuses
- scale_bonus: Horizon bonus when resource is high

PPO parameters:
- learning_rate: Step size for Adam optimizer
- gamma: Discount factor
- lambda: GAE parameter
- clip_eps: PPO clipping range
- train_iters: Number of PPO update passes per episode
- num_episodes: Total training episodes

------------------------------------------------------------
Environment Summary
------------------------------------------------------------

State (per agent):
- Normalized resource R/R_max
- Change in resource ΔR = R_t - R_{t-1}
- Agent’s previous action

Action:
A continuous scalar in [0,1], scaled to [0, max_extract].

Reward:
- Extraction reward. The amount an agent extracts/consumes is the reward they get. Ex. agent 1 takes 3 resources -> agent 1 gets a reward of 3.
- Derivative shaping reward based on whether the resource is growing
- Asymmetric collapse penalty: largest extractor is punished most
- Horizon bonus for maintaining sustainability

This produces a fully decentralized, non-stationary multi-agent environment.

------------------------------------------------------------
Future Work
------------------------------------------------------------

- Centralized critic (e.g., MAPPO or MADDPG) to improve stability
- Team-based rewards for cooperative sustainability
- Alternative shaping signals or observation channels

------------------------------------------------------------
Repository
------------------------------------------------------------

https://github.com/isaacscottirwin/TragedyOfTheCommonsRL.git
