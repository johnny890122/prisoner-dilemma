from pd_env.pd_environment import PrisonerDilemmaEnvironment
import pd_env.utils as utils
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    print("Running test for Prisoner Dilemma Environment")
    env = PrisonerDilemmaEnvironment()
    observations, infos = env.reset()
    actions = [0,0,0,0]
    env.step(actions)
    agent0_obs = utils.pretty_print_obs(env.observe_all(0))
    print(agent0_obs)
