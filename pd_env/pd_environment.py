import functools, random
from copy import copy
import numpy as np
from typing import List, Dict
import gymnasium.spaces as spaces
from pettingzoo import ParallelEnv
import pd_env.CONST as CONST
import pd_env.utils as utils

class PrisonerDilemmaEnvironment(ParallelEnv):
    metadata = {
        "name": "PrisonerDilemma_environment_v0",
    }

    def __init__(self):
        self.agents = [agent for agent in range(CONST.NUM_AGENTS)]
        self.agents_name_mapping = {
            agent: name for agent, name in zip(self.agents, CONST.AGENTS_NAMES)
        }

        self.actions = [action for action in range(CONST.NUM_ACTIONS)]
        self.action_name_mapping = {
            action: name for action, name in zip(self.actions, CONST.ACTIONS_NAMES)
        }

        self.timestep = None

    def reset(self, seed: int=CONST.SEED, options=None):
        self.timestep = 0
        self.seed = seed

        pairs = utils.pd_pairs(self.agents)
        actions = [None for _ in self.agents]
        payoffs = [0 for _ in self.agents]
        
        observations = [{
            agent: (
                agent, # self identity 
                self.agents, # identity of each agent
                pairs, # pairs of agents
                actions, # actions of each agent
                payoffs, # payoffs of each agent
            ) for agent in self.agents
        }]

        # Get dummy infos. Necessary for proper parallel_to_aec conversion
        infos = {agent: self.info(agent) for agent in self.agents}

        return observations, infos
    
    def step(self, actions: List[int]):
        # TODO: Implement the reward later
        rewards = {a: 0 for a in self.agents}

        # Check termination conditions
        terminations = {a: False for a in self.agents}
        
        # Check truncation conditions (overwrites termination conditions)
        if self.timestep > CONST.MAX_ROUNDS:
            truncations = {agent: True for agent in self.agents}
        else:
            truncations = {agent: False for agent in self.agents}

        self.timestep += 1

        # Get dummy infos (not used in this example)
        infos = {agent: self.info(agent) for agent in self.agents}

        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Renders the environment."""
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None) # Cache the action space
    def action_space(self, agent) -> spaces.Space:
        # lru_cache allows action spaces to be memoized, reducing clock cycles required to get each agent's space.
        return spaces.Discrete(CONST.NUM_ACTIONS)

    def info(self, agent: int) -> Dict:
        info = {} # TODO: return the info
        return info