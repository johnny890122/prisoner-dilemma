import numpy as np
from gymnasium import spaces
import pd_env.CONST as CONST
from typing import List, Tuple

from typing import List, Tuple
import numpy as np

def pd_pairs(agents: List[int]) -> Tuple[List[int]]:
    """
    Randomly pairs agents into two groups for the prisoner's dilemma game.

    Args:
        agents (List[int]): A list of agent IDs.

    Returns:
        Tuple[List[int]]: A tuple containing two lists, where the first list represents the first pair of agents
                          and the second list represents the second pair of agents.
    """
    assert len(agents) == CONST.NUM_AGENTS, f"Number of agents must be {CONST.NUM_AGENTS}."
    first_pair = np.random.choice(agents, 2, replace=False).tolist()
    second_pair = list(set(agents) - set(first_pair))
    return first_pair, second_pair
