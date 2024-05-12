import numpy as np
from gymnasium import spaces
import pd_env.CONST as CONST
from typing import List, Tuple
from copy import copy

def pd_pairs(agents: List[int]) -> List[Tuple[int]]:
    """
    Randomly pairs agents into two groups for the prisoner's dilemma game.

    Args:
        agents (List[int]): A list of agent IDs.

    Returns:
        List[Tuple[int]]: A list containing two tuples, where the first tuple represents the first pair of agents
                          and the second tuple represents the second pair of agents.
    """
    assert len(agents) == CONST.NUM_AGENTS, f"Number of agents must be {CONST.NUM_AGENTS}."
    first_pair = np.random.choice(agents, 2, replace=False)
    second_pair = set(agents) - set(first_pair)
    return [tuple(first_pair), tuple(second_pair)]

from typing import List, Tuple

def my_partner(agent: int, pairs: Tuple[List[int]]) -> int:
    """
    Find the partner of the given agent in a list of pairs.

    Args:
        agent (int): The agent for which to find the partner.
        pairs (Tuple[List[int]]): A list of pairs, where each pair is represented as a list of two agents.

    Returns:
        int: The partner of the given agent.

    Raises:
        None

    Examples:
        >>> my_partner(1, [(1, 2), (3, 4)])
        2
        >>> my_partner(3, [(1, 2), (3, 4)])
        4
    """
    for pair in pairs:
        if agent in pair:
            partner = set(pair) - {agent}
            if len(partner) == 1:
                return int(list(partner)[0])
            return list(partner)
