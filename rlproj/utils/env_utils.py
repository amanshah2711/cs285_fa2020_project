import functools
import gym
import gym_compete
from stable_baselines.common.vec_env import DummyVecEnv

from .embedding import embed_agent

def make_env(env_name, seed, i, horizon=None):
    env = gym.make(env_name)
    env.seed(seed + i)
    env.reset() # TODO: Check we don't need to do more think about unwrapping
    return env

def build_multi_env(env_name, num_env, seed, debug):


    env_fn = lambda i: make_env(env_name, seed, i) #TODO: Figure out better way to resolve circular dependency aside from local import
    from ..envs.multi_agent import make_dummy_vec_multi_env #TODO: Check if we can have dummy multi vec
    multi_env = make_dummy_vec_multi_env([functools.partial(env_fn, i) for i in range(num_env)]) # creates a num_env environments crammed into vec_multi environment with actions tranposed

    return multi_env

def build_env(env_name, num_env, seed, debug):

    multi_env = build_multi_env(env_name, num_env, seed, debug)
    #TODO: MIGHT BE FORCED TO EMBED HERE
    from ..envs.multi_agent import FlattenSingletonVecEnv
    multi_env = embed_agent(multi_env)
    single_agent_env = FlattenSingletonVecEnv(multi_env)
    return single_agent_env

def getattr_unwrapped(env, attr):
    """Get attribute attr from env, or one of the nested environments.
    Args:
        - env(gym.Wrapper or gym.Env): a (possibly wrapped) environment.
        - attr: name of the attribute
    Returns:
        env.attr, if present, otherwise env.unwrapped.attr and so on recursively.
    """
    try:
        return getattr(env, attr)
    except AttributeError:
        if env.env == env:
            raise
        else:
            return getattr_unwrapped(env.env, attr)
