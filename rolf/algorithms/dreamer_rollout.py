"""
Runs rollouts (RolloutRunner class) and collects transitions using Rollout class.
"""

import numpy as np
import gymnasium as gym

from .rollout import Rollout, RolloutRunner
from ..utils import Logger, Info, Every, rmap


class DreamerRolloutRunner(RolloutRunner):
    """Rollout a policy."""

    def __init__(self, cfg, env, env_eval, agent):
        """
        Args:
            cfg: configurations for the environment.
            env: training environment.
            env_eval: testing environment.
            agent: policy.
        """
        self._cfg = cfg
        self._env = env
        self._env_eval = env_eval
        self._agent = agent
        self._exclude_rollout_log = ["episode_success_state"]

    def run(self, every_steps=None, every_episodes=None, log_prefix="", step=0):
        """
        Collects trajectories for training and yield every `every_steps`/`every_episodes`.

        In rollout:
          ob:         [o(1), o(2), o(3), ..., o(T)]
          ac:         [0   , a(1), a(2), ..., a(T-1)]
          rew:        [0   , r(1), r(2), ..., r(T-1)]
          done:       [0   , d(1), d(2), ..., d(T-1)]
          terminated: [0   , t(1), t(2), ..., t(T-1)]

        Args:
            every_steps: if not None, returns rollouts `every_steps`
            every_episodes: if not None, returns rollouts `every_epiosdes`
            log_prefix: log as `log_prefix` rollout: %s
        """
        if every_steps is None:
            raise ValueError("For parallel rollouts, only every_steps is available.")
        every_steps = Every(every_steps, step)

        cfg = self._cfg
        env = self._env
        agent = self._agent
        num_envs = env.num_envs

        # Initialize rollout buffer.
        rollout = [
            Rollout(["ob", "ac", "rew", "done", "terminated"], cfg.rolf.precision)
            for _ in range(num_envs)
        ]
        rollout_info = [Info() for _ in range(num_envs)]
        ep = Rollout(["ob", "ac", "rew", "done", "terminated"], cfg.rolf.precision)
        ep_info = Info()
        rollout_len = 0
        dummy_ac = np.zeros(gym.spaces.flatdim(env.single_action_space))

        def add_step(ob, ac, rew, done, terminated, info):
            if "final_observation" in info:
                ob_final = info.pop("final_observation")
                info_final = info.pop("final_info")

            transitions = dict(ac=ac, rew=rew, done=done, terminated=terminated)

            for i in range(num_envs):
                rollout[i].add(rmap(lambda x: x[i], transitions))
                rollout_info[i].add(dict(rew=rew[i]))
                if done[i]:
                    rollout[i].add(dict(ob=rmap(lambda x: x[i], ob_final)))
                    rollout_info[i].add(rmap(lambda x: x[i], info_final))

                    ep.extend(rollout[i].get())
                    log = rollout_info[i].get_dict(reduction="sum", only_scalar=True)
                    ep_info.add(log)
                    Logger.info(f"[{log_prefix}] rollout: {log}")

                    # Starting a new episode
                    rollout[i].add(dict(ac=dummy_ac, rew=0.0))
                    rollout[i].add(dict(done=False, terminated=False))
                rollout[i].add(dict(ob=rmap(lambda x: x[i], ob)))
                rollout_info[i].add(rmap(lambda x: x[i], info))
                rollout_info[i].add(dict(len=1))

        # Add dummy previous action for the first transition.
        ob_next, info = env.reset()
        state_next = None
        ac = np.zeros(gym.spaces.flatdim(env.action_space)).reshape(num_envs, -1)
        rew = np.zeros(num_envs)
        done = np.full((num_envs,), False)
        terminated = np.full((num_envs,), False)
        add_step(ob_next, ac, rew, done, terminated, info)

        while True:
            ob, state = ob_next, state_next

            # Sample an action from the policy.
            if step < cfg.rolf.warm_up_step:
                ac, state_next = env.action_space.sample(), None
            else:
                ac, state_next = agent.act(ob, state, is_train=True)
                ac = gym.spaces.unflatten(env.action_space, ac.flatten())

            # Take a step.
            ob_next, reward, terminated, truncated, info = env.step(ac)
            done = np.logical_or(terminated, truncated)
            flat_ac = gym.spaces.flatten(env.action_space, ac).reshape(num_envs, -1)
            add_step(ob_next, flat_ac, reward, done, terminated, info)
            state_next = agent.mask_state(state_next, done)

            step += num_envs
            rollout_len += num_envs

            if every_steps(step):
                yield ep.get(), rollout_len, ep_info.get_dict(only_scalar=True)
                rollout_len = 0

    def run_episode(self, record_video=False):
        """
        Runs one episode and returns the rollout for evaluation.

        Args:
            record_video: record video of rollout if True.
        """
        cfg = self._cfg
        env = self._env_eval
        agent = self._agent

        assert env.num_envs == 1

        # Initialize rollout buffer.
        ep = Rollout(["ob", "ac", "rew", "done", "terminated"], cfg.rolf.precision)
        ep_info = Info()

        ob_next, info = env.reset()
        state_next = None
        done = False
        ep_len = 0
        ep_rew = 0.0

        record_frames = []
        if record_video:
            record_frames.append(self._render_frame(ep_len, ep_rew))

        # Rollout one episode
        while not done:
            ob, state = ob_next, state_next

            # Sample an action from the policy.
            ac, state_next = agent.act(ob, state, is_train=False)
            ac = gym.spaces.unflatten(env.action_space, ac.flatten())

            # Take a step.
            ob_next, reward, terminated, truncated, info = env.step(ac)
            done = terminated[0] or truncated[0]
            info.update(dict(ac=ac, len=1, rew=reward[0]))
            ep_len += 1
            ep_rew += reward[0]

            flat_ac = gym.spaces.flatten(env.action_space, ac).reshape(1, -1)
            ep.add(dict(ob=ob, ac=flat_ac, rew=reward))
            ep.add(dict(done=np.logical_or(terminated, truncated)))
            ep.add(dict(terminated=terminated))
            ep_info.add(info)
            if record_video:
                frame_info = info.copy()
                record_frames.append(self._render_frame(ep_len, ep_rew, frame_info))

        # Add last observation.
        ep.add({"ob": info["final_observation"][0]})
        ep_info.add(info["final_info"][0])
        ep_info = ep_info.get_dict(reduction="sum", only_scalar=True)

        Logger.info(f"eval rollout: {ep_info}")
        return ep.get(), ep_info, record_frames
