import logging
import os
import argparse

#from rlproj.config import config, update_config
from rlproj import utils
from rlproj.envs.multi_agent import CurryVecEnv,FlattenSingletonVecEnv, FakeSingleSpacesVec
from rlproj.policies.base import RandomPolicy
from rlproj.policies.loader import *
from rlproj.utils.embedding import embed_agent
from rlproj.utils.callbacks import load_params

import numpy as np
import tensorflow as tf
import stable_baselines


def main():

    #logger = setup_logger() #TODO: MAKE WORK
    env = utils.build_multi_env("multicomp/KickAndDefend-v0", 2, 0, False)

    random_policy = load_policy('zoo', 1, env,"multicomp/KickAndDefend-v0" , 0)
    env = CurryVecEnv(env, random_policy)
    item = env.get_policy().policy_obj
    load_params(item, '')
    env = FlattenSingletonVecEnv(env)
    for _ in range(1000):
        env.render()
        action = np.array([np.array(env.action_space.sample()) for _ in range(2)])
        env.step(action)  # take a random action
    env.close()

if __name__ == '__main__':
    main()