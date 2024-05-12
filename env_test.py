from pd_env.pd_environment import PrisonerDilemmaEnvironment
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    print("Running test for Prisoner Dilemma Environment")
    env = PrisonerDilemmaEnvironment()
    observations, infos = env.reset()
    actions = [0,0,0,0]
    env.step(actions)
    print(env.obeseve_all(0))