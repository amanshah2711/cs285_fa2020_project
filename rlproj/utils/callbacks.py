import os
import os.path as osp
import pickle
import pkgutil
import pandas as pd

from typing import Callable, Iterable
from stable_baselines.common import callbacks

SaveCallback = Callable[[str], None]

def _save(model, root_dir: str, save_callbacks: Iterable[SaveCallback]) -> None:
    os.makedirs(root_dir, exist_ok=True)
    model_path = osp.join(root_dir, "model.pkl")
    model.save(model_path)
    for f in save_callbacks:
        f(root_dir)


class CheckpointCallback(callbacks.BaseCallback):
    """Custom checkpointing, saving model in directory and recursively calling `save_callbacks`."""

    def __init__(self, out_dir: str, save_callbacks: Iterable[SaveCallback], *args, **kwargs):
        """
        Builds a CheckpointCallback.
        `save_callbacks` used to save auxiliary information, e.g. `VecNormalize` instances.
        :param out_dir: directory to save checkpoints to.
        :param save_callbacks: callbacks to recursively invoke.
        """
        super(CheckpointCallback, self).__init__(*args, **kwargs)
        self.out_dir = out_dir
        self.save_callbacks = save_callbacks

    def _on_step(self) -> bool:
        checkpoint_dir = osp.join(self.out_dir, "checkpoint", f"{self.num_timesteps:012}")
        _save(self.model, checkpoint_dir, self.save_callbacks)
        return True


class LoggerOnlyLogCallback(callbacks.BaseCallback):
    """Calls `obj.log_callback(self.logger)`."""

    def __init__(self, obj, *args, **kwargs):
        super(LoggerOnlyLogCallback, self).__init__(*args, **kwargs)
        assert hasattr(obj, "log_callback")
        self.obj = obj

    def _on_step(self) -> bool:
        self.obj.log_callback(self.logger)
        return True

class SwapCallback(callbacks.BaseCallback):
    """Designed to be used with CurryVecEnv to swap the opponent policy periodically through some opponent sampling strategy"""

    def __init__(self, model_dir: str, opponent_sampler, save_callbacks: Iterable[SaveCallback]) -> None:
        self.model_dir = model_dir
        self.opponent_sampler = opponent_sampler

    def _on_rollout_end(self) -> None:
        adversary = self.opponent_sampler()
        self.model.get_env()._policy


def load_params(policy_obj, loc):
    with open(os.path.join(loc, 'model.pkl'), 'rb') as f:
        import pdb
        pdb.set_trace()
        pd.read_pickle(f)
        print(params)
    policy_obj.restore(params)


class RetrainCallback(callbacks.BaseCallback):

    def __init__(self):
        pass

    def _on_step(self) -> bool:
        pass
