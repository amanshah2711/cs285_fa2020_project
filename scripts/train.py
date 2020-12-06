import logging
import os

from rlproj.config import config, update_config
from rlproj import utils
from rlproj.envs.multi_agent import CurryVecEnv,FlattenSingletonVecEnv
from rlproj.policies.base import RandomPolicy
from rlproj.policies.loader import *

import numpy as np
import tensorflow as tf
import stable_baselines

def main():

    #logger = setup_logger() #TODO: MAKE WORK
    env = utils.build_multi_env("multicomp/KickAndDefend-v0", 2, 0, False)


    random_policy = load_policy('random', None, env, None, 0)
    env = CurryVecEnv(env, random_policy)
    env = FlattenSingletonVecEnv(env)
    for _ in range(1000):
        env.render()
        action = np.array([np.array(env.action_space.sample()) for _ in range(2)])
        env.step(action)  # take a random action
    env.close()


if __name__ == '__main__':
    main()