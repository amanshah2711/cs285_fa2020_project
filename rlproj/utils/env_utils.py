import functools
import gym

def make_env(env_name, seed, i, horizon=None):
    env = gym.make(env_name)
    env.seed(seed + i)
    env.reset() # TODO: Check we don't need to do more think about unwrapping
    return env

def build_multi_env(env_name, num_env, seed, debug):

    env_fn = lambda i: make_env(env_name, seed, i)
    from ..envs.multi_agent import make_dummy_vec_multi_env, make_subproc_vec_multi_env
    if not debug and num_env > 1:
        make_vec_env = make_subproc_vec_multi_env
    else:
        make_vec_env = make_dummy_vec_multi_env
    multi_venv = make_vec_env([functools.partial(env_fn, i) for i in range(num_env)])

    return multi_venv


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
