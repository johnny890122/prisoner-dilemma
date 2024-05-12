from pd_env.pd_environment import PrisonerDilemmaEnvironment
from pettingzoo.test import parallel_api_test

if __name__ == "__main__":
    print("Running test for CustomEnvironment")
    env = PrisonerDilemmaEnvironment()
    print(env)
    # parallel_api_test(env, num_cycles=1_000_000)