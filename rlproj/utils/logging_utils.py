import time
import datetime

import numpy as np
from stable_baselines.bench import Monitor

from .env_utils import getattr_unwrapped

def make_timestamp():
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    return datetime.datetime.now().strftime(ISO_TIMESTAMP)

class MultiMonitor(Monitor):
    def __init__(
        self,
        env,
        filename,
        our_idx=None,
        allow_early_resets=False,
        reset_keywords=(),
        info_keywords=(),
    ):
        num_agents = getattr_unwrapped(env, "num_agents")
        extra_rks = tuple("r{:d}".format(i) for i in range(num_agents))
        super().__init__(
            env,
            filename,
            allow_early_resets=allow_early_resets,
            reset_keywords=reset_keywords,
            info_keywords=extra_rks + info_keywords,
        )
        self.our_idx = our_idx
        self.info_keywords = info_keywords

    def step(self, action):
        """
        Step the environment with the given action
        :param action: ([int] or [float]) the action
        :return: ([int] or [float], [float], [bool], dict) observation, reward, done, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.needs_reset = True
            eplen = len(self.rewards)
            ep_rew = np.asarray(self.rewards).sum(axis=0).round(6)
            our_rew = float("nan") if self.our_idx is None else ep_rew[self.our_idx]
            ep_info = {"r": our_rew, "l": eplen, "t": round(time.time() - self.t_start, 6)}
            for i, rew in enumerate(ep_rew):
                ep_info["r{:d}".format(i)] = rew
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_rewards.append(ep_rew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            if self.logger:
                self.logger.writerow(ep_info)
                self.file_handler.flush()
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


"""Logging for RL algorithms.
Configures Baseline's logger and TensorBoard appropriately."""

import os
from os import path as osp

from stable_baselines import logger
from tensorboard.plugins.custom_scalar import layout_pb2
import tensorboard.summary as summary_lib
from tensorflow.core.util import event_pb2

from aprl.common import utils


def gen_multiline_charts(cfg):
    charts = []
    for title, tags in cfg:
        charts.append(
            layout_pb2.Chart(title=title, multiline=layout_pb2.MultilineChartContent(tag=tags))
        )
    return charts


def tb_layout():
    episode_rewards = layout_pb2.Category(
        title="Episode Reward",
        chart=gen_multiline_charts(
            [
                ("Shaped Reward", [r"shaping/eprewmean_true"]),
                ("Episode Length", [r"eplenmean"]),
                ("Sparse Reward", [r"shaping/epsparsemean"]),
                ("Dense Reward", [r"shaping/epdensemean"]),
                ("Dense Reward Annealing", [r"shaping/rew_anneal_c"]),
                ("Unshaped Reward", [r"ep_rewmean"]),
                ("Victim Action Noise", [r"shaping/victim_noise"]),
            ]
        ),
    )

    game_outcome = layout_pb2.Category(
        title="Game Outcomes",
        chart=gen_multiline_charts(
            [
                ("Agent 0 Win Proportion", [r"game_win0"]),
                ("Agent 1 Win Proportion", [r"game_win1"]),
                ("Tie Proportion", [r"game_tie"]),
                ("# of games", [r"game_total"]),
            ]
        ),
    )

    training = layout_pb2.Category(
        title="Training",
        chart=gen_multiline_charts(
            [
                ("Policy Loss", [r"policy_loss"]),
                ("Value Loss", [r"value_loss"]),
                ("Policy Entropy", [r"policy_entropy"]),
                ("Explained Variance", [r"explained_variance"]),
                ("Approx KL", [r"approxkl"]),
                ("Clip Fraction", [r"clipfrac"]),
            ]
        ),
    )

    # Intentionally unused:
    # + serial_timesteps (just total_timesteps / num_envs)
    # + time_elapsed (TensorBoard already logs wall-clock time)
    # + nupdates (this is already logged as step)
    time = layout_pb2.Category(
        title="Time",
        chart=gen_multiline_charts([("Total Timesteps", [r"total_timesteps"]), ("FPS", [r"fps"])]),
    )

    categories = [episode_rewards, game_outcome, training, time]
    return summary_lib.custom_scalar_pb(layout_pb2.Layout(category=categories))


def setup_logger(out_dir="results", exp_name="test", output_formats=None):
    timestamp = utils.make_timestamp()
    exp_name = exp_name.replace("/", "_")  # environment names can contain /'s
    filename = "{}-{}".format(timestamp, exp_name)[0:255]  # Linux has filename limit of 255
    out_dir = osp.join(out_dir, filename)
    os.makedirs(out_dir, exist_ok=True)

    logger.configure(folder=osp.join(out_dir, "rl"), format_strs=["tensorboard", "stdout"])
    logger_instance = logger.Logger.CURRENT

    if output_formats is not None:
        logger_instance.output_formats += output_formats

    for fmt in logger_instance.output_formats:
        if isinstance(fmt, logger.TensorBoardOutputFormat):
            writer = fmt.writer
            layout = tb_layout()
            event = event_pb2.Event(summary=layout)
            writer.WriteEvent(event)
            writer.Flush()

    return out_dir, logger_instance