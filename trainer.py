"""
Base code for RL/IL training.
Collects rollouts and updates policy networks.
"""

import os
import gzip
import pickle
import copy
from time import time
from collections import defaultdict, OrderedDict

import torch
import wandb
import h5py
import gym
import numpy as np
import moviepy.editor as mpy
from tqdm import tqdm, trange

from .algorithms import RL_ALGOS, IL_ALGOS, get_agent_by_name
from .algorithms.rollouts import RolloutRunner
from .utils.info_dict import Info
from .utils.logger import logger
from .utils.pytorch import get_ckpt_path, count_parameters
from .utils.mpi import mpi_sum, mpi_average, mpi_gather_average
from .environments import make_env


class Trainer(object):
    """
    Trainer class for SAC, PPO, DDPG, BC, and GAIL in PyTorch.
    """

    def __init__(self, config):
        """
        Initializes class with the configuration.
        """
        self._config = config
        self._is_chef = config.is_chef
        self._is_rl = config.algo in RL_ALGOS
        self._average_info = config.average_info

        # create environment
        self._env = make_env(config.env, config)
        ob_space = env_ob_space = self._env.observation_space
        ac_space = self._env.action_space
        logger.info("Observation space: " + str(ob_space))
        logger.info("Action space: " + str(ac_space))

        config_eval = copy.copy(config)
        if hasattr(config_eval, "port"):
            config_eval.port += 1
        self._env_eval = make_env(config.env, config_eval) if self._is_chef else None

        # create a new observation space after data augmentation (random crop)
        if config.encoder_type == "cnn":
            assert (
                not config.ob_norm
            ), "Turn off the observation norm (--ob_norm False) for pixel inputs"
            ob_space = gym.spaces.Dict(spaces=dict(ob_space.spaces))
            for k in ob_space.spaces.keys():
                if len(ob_space.spaces[k].shape) == 3:
                    shape = [
                        ob_space.spaces[k].shape[0],
                        config.encoder_image_size,
                        config.encoder_image_size,
                    ]
                    ob_space.spaces[k] = gym.spaces.Box(
                        low=0, high=255, shape=shape, dtype=np.uint8
                    )

        # build agent and networks for algorithm
        self._agent = get_agent_by_name(config.algo)(
            config, ob_space, ac_space, env_ob_space
        )

        # build rollout runner
        self._runner = RolloutRunner(config, self._env, self._env_eval, self._agent)

        # setup log
        if self._is_chef and config.is_train:
            exclude = ["device"]
            if not config.wandb:
                os.environ["WANDB_MODE"] = "dryrun"

            wandb.init(
                resume=config.run_name,
                project=config.wandb_project,
                config={k: v for k, v in config.__dict__.items() if k not in exclude},
                dir=config.log_dir,
                entity=config.wandb_entity,
                notes=config.notes,
            )

    def _save_ckpt(self, ckpt_num, update_iter):
        """
        Save checkpoint to log directory.

        Args:
            ckpt_num: number appended to checkpoint name. The number of
                environment step is used in this code.
            update_iter: number of policy update. It will be used for resuming training.
        """
        ckpt_path = os.path.join(self._config.log_dir, "ckpt_%08d.pt" % ckpt_num)
        state_dict = {"step": ckpt_num, "update_iter": update_iter}
        state_dict["agent"] = self._agent.state_dict()
        torch.save(state_dict, ckpt_path)
        logger.warn("Save checkpoint: %s", ckpt_path)

        if self._agent.is_off_policy():
            replay_path = os.path.join(
                self._config.log_dir, "replay_%08d.pkl" % ckpt_num
            )
            with gzip.open(replay_path, "wb") as f:
                replay_buffers = {"replay": self._agent.replay_buffer()}
                pickle.dump(replay_buffers, f)

    def _load_ckpt(self, ckpt_path, ckpt_num):
        """
        Loads checkpoint with path @ckpt_path or index number @ckpt_num. If @ckpt_num is None,
        it loads and returns the checkpoint with the largest index number.
        """
        if ckpt_path is None:
            ckpt_path, ckpt_num = get_ckpt_path(self._config.log_dir, ckpt_num)
        else:
            ckpt_num = int(ckpt_path.rsplit("_", 1)[-1].split(".")[0])

        if ckpt_path is not None:
            logger.warn("Load checkpoint %s", ckpt_path)
            ckpt = torch.load(ckpt_path, map_location=self._config.device)
            self._agent.load_state_dict(ckpt["agent"])

            if self._config.is_train and self._agent.is_off_policy():
                replay_path = os.path.join(
                    self._config.log_dir, "replay_%08d.pkl" % ckpt_num
                )
                logger.warn("Load replay_buffer %s", replay_path)
                if os.path.exists(replay_path):
                    with gzip.open(replay_path, "rb") as f:
                        replay_buffers = pickle.load(f)
                        self._agent.load_replay_buffer(replay_buffers["replay"])
                else:
                    logger.warn("Replay buffer not exists at %s", replay_path)

            if (
                self._config.init_ckpt_path is not None
                and "bc" in self._config.init_ckpt_path
            ):
                return 0, 0
            else:
                return ckpt["step"], ckpt["update_iter"]
        logger.warn("Randomly initialize models")
        return 0, 0

    def _log_train(self, step, train_info, ep_info):
        """
        Logs training and episode information to wandb.
        Args:
            step: the number of environment steps.
            train_info: training information to log, such as loss, gradient.
            ep_info: episode information to log, such as reward, episode time.
        """
        for k, v in train_info.items():
            if np.isscalar(v) or (hasattr(v, "shape") and np.prod(v.shape) == 1):
                wandb.log({"train_rl/%s" % k: v}, step=step)
            else:
                wandb.log({"train_rl/%s" % k: [wandb.Image(v)]}, step=step)

        for k, v in ep_info.items():
            wandb.log({"train_ep/%s" % k: np.mean(v)}, step=step)
            wandb.log({"train_ep_max/%s" % k: np.max(v)}, step=step)

    def _log_test(self, step, ep_info):
        """
        Logs episode information during testing to wandb.
        Args:
            step: the number of environment steps.
            ep_info: episode information to log, such as reward, episode time.
        """
        if self._config.is_train:
            for k, v in ep_info.items():
                if isinstance(v, wandb.Video):
                    wandb.log({"test_ep/%s" % k: v}, step=step)
                elif isinstance(v, list) and isinstance(v[0], wandb.Video):
                    for i, video in enumerate(v):
                        wandb.log({"test_ep/%s_%d" % (k, i): video}, step=step)
                else:
                    wandb.log({"test_ep/%s" % k: np.mean(v)}, step=step)

    def train(self):
        """ Trains an agent. """
        config = self._config

        # load checkpoint
        step, update_iter = self._load_ckpt(config.init_ckpt_path, config.ckpt_num)

        # sync the networks across the cpus
        self._agent.sync_networks()

        logger.info("Start training at step=%d", step)
        if self._is_chef:
            pbar = tqdm(
                initial=update_iter, total=config.max_global_step, desc=config.run_name
            )
            ep_info = Info()
            train_info = Info()

        # decide how many episodes or how long rollout to collect
        if self._config.algo == "bc":
            runner = None
        elif self._config.algo == "gail":
            runner = self._runner.run(
                every_steps=self._config.rollout_length, step=step
            )
        elif self._config.algo == "ppo":
            runner = self._runner.run(
                every_steps=self._config.rollout_length, step=step
            )
        elif self._config.algo in ["sac", "ddpg", "td3"]:
            runner = self._runner.run(every_steps=1, step=step)
            # runner = self._runner.run(every_episodes=1)
        elif self._config.algo == "dac":
            runner = self._runner.run(every_steps=1, step=step)

        st_time = time()
        st_step = step

        while runner and step < config.warm_up_steps:
            rollout, info = next(runner)
            self._agent.store_episode(rollout)
            step_per_batch = mpi_sum(len(rollout["ac"]))
            step += step_per_batch
            if runner and step < config.max_ob_norm_step:
                self._update_normalizer(rollout)
            if self._is_chef:
                pbar.update(step_per_batch)

        if self._config.algo == "bc" and self._config.ob_norm:
            self._agent.update_normalizer()

        while step < config.max_global_step:
            # collect rollouts
            if runner:
                rollout, info = next(runner)
                if self._average_info:
                    info = mpi_gather_average(info)
                self._agent.store_episode(rollout)
                step_per_batch = mpi_sum(len(rollout["ac"]))
            else:
                step_per_batch = mpi_sum(1)
                info = {}

            # train an agent
            _train_info = self._agent.train()

            if runner and step < config.max_ob_norm_step:
                self._update_normalizer(rollout)

            step += step_per_batch
            update_iter += 1

            # log training and episode information or evaluate
            if self._is_chef:
                pbar.update(step_per_batch)
                ep_info.add(info)
                train_info.add(_train_info)

                if update_iter % config.log_interval == 0:
                    train_info.add(
                        {
                            "sec": (time() - st_time) / config.log_interval,
                            "steps_per_sec": (step - st_step) / (time() - st_time),
                            "update_iter": update_iter,
                        }
                    )
                    st_time = time()
                    st_step = step
                    self._log_train(step, train_info.get_dict(), ep_info.get_dict())
                    ep_info = Info()
                    train_info = Info()

                if update_iter % config.evaluate_interval == 1:
                    logger.info("Evaluate at %d", update_iter)
                    rollout, info = self._evaluate(
                        step=step, record_video=config.record_video
                    )
                    self._log_test(step, info)

                if update_iter % config.ckpt_interval == 0:
                    self._save_ckpt(step, update_iter)

        self._save_ckpt(step, update_iter)
        logger.info("Reached %s steps. worker %d stopped.", step, config.rank)

    def _update_normalizer(self, rollout):
        """ Updates normalizer with @rollout. """
        if self._config.ob_norm:
            self._agent.update_normalizer(rollout["ob"])

    def _evaluate(self, step=None, record_video=False):
        """
        Runs one rollout if in eval mode (@idx is not None).
        Runs num_record_samples rollouts if in train mode (@idx is None).

        Args:
            step: the number of environment steps.
            record_video: whether to record video or not.
        """
        logger.info("Run %d evaluations at step=%d", self._config.num_eval, step)
        rollouts = []
        info_history = Info()
        for i in range(self._config.num_eval):
            logger.warn("Evalute run %d", i + 1)
            rollout, info, frames = self._runner.run_episode(
                is_train=False, record_video=record_video
            )
            rollouts.append(rollout)
            logger.info(
                "rollout: %s", {k: v for k, v in info.items() if not "qpos" in k}
            )

            if record_video:
                ep_rew = info["rew"]
                ep_success = (
                    "s"
                    if "episode_success" in info and info["episode_success"]
                    else "f"
                )
                fname = "{}_step_{:011d}_{}_r_{}_{}.mp4".format(
                    self._config.env, step, i, ep_rew, ep_success,
                )
                video_path = self._save_video(fname, frames)
                if self._config.is_train:
                    info["video"] = wandb.Video(video_path, fps=15, format="mp4")

            info_history.add(info)

        return rollouts, info_history

    def evaluate(self):
        """ Evaluates an agent stored in chekpoint with @self._config.ckpt_num. """
        step, update_iter = self._load_ckpt(
            self._config.init_ckpt_path, self._config.ckpt_num
        )

        logger.info(
            "Run %d evaluations at step=%d, update_iter=%d",
            self._config.num_eval,
            step,
            update_iter,
        )
        rollouts, info = self._evaluate(
            step=step, record_video=self._config.record_video
        )

        info_stat = info.get_stat()
        os.makedirs("result", exist_ok=True)
        with h5py.File("result/{}.hdf5".format(self._config.run_name), "w") as hf:
            for k, v in info.items():
                hf.create_dataset(k, data=info[k])
        with open("result/{}.txt".format(self._config.run_name), "w") as f:
            for k, v in info_stat.items():
                f.write("{}\t{:.03f} $\\pm$ {:.03f}\n".format(k, v[0], v[1]))


        if self._config.record_demo:
            new_rollouts = []
            for rollout in rollouts:
                new_rollout = {
                    "obs": rollout["ob"],
                    "actions": rollout["ac"],
                    "rewards": rollout["rew"],
                    "dones": rollout["done"],
                }
                new_rollouts.append(new_rollout)

            fname = "{}_step_{:011d}_{}_trajs.pkl".format(
                self._config.run_name, step, self._config.num_eval,
            )
            path = os.path.join(self._config.demo_dir, fname)
            logger.warn("[*] Generating demo: {}".format(path))
            with open(path, "wb") as f:
                pickle.dump(new_rollouts, f)

    def _save_video(self, fname, frames, fps=15.0):
        """ Saves @frames into a video with file name @fname. """
        path = os.path.join(self._config.record_dir, fname)
        logger.warn("[*] Generating video: {}".format(path))

        def f(t):
            frame_length = len(frames)
            new_fps = 1.0 / (1.0 / fps + 1.0 / frame_length)
            idx = min(int(t * new_fps), frame_length - 1)
            return frames[idx]

        video = mpy.VideoClip(f, duration=len(frames) / fps + 2)

        video.write_videofile(path, fps, verbose=False)
        logger.warn("[*] Video saved: {}".format(path))
        return path
