"""
Helper functions to make a vector environment.

Code modified based on
https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
and
https://github.com/openai/gym/blob/master/gym/vector/async_vector_env.py
"""

import multiprocessing as mp
import sys
from copy import deepcopy

import numpy as np
import gym
from gym.vector.utils import (
    CloudpickleWrapper,
    clear_mpi_env_vars,
    concatenate,
    create_empty_array,
    create_shared_memory,
    iterate,
    read_from_shared_memory,
    write_to_shared_memory,
)

from .vec_env import VecEnv


def worker(remote, parent_remote, env_fn_wrappers):
    def step_env(env, action):
        ob, reward, done, info = env.step(action)
        if done:
            ob = env.reset()
        return ob, reward, done, info

    parent_remote.close()
    envs = [env_fn_wrapper() for env_fn_wrapper in env_fn_wrappers.x]
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send([step_env(env, action) for env, action in zip(envs, data)])
            elif cmd == "reset":
                remote.send([env.reset() for env in envs])
            elif cmd == "render":
                remote.send([env.render(mode="rgb_array") for env in envs])
            elif cmd == "close":
                remote.close()
                break
            elif cmd == "get_spaces_spec":
                remote.send(
                    CloudpickleWrapper(
                        (envs[0].observation_space, envs[0].action_space, envs[0].spec)
                    )
                )
            else:
                raise NotImplementedError
    except KeyboardInterrupt:
        print("SubprocVecEnv worker: got KeyboardInterrupt")
    finally:
        for env in envs:
            env.close()


class SubprocEnv(gym.Env):
    def __init__(self, env_fn, context=None):
        ctx = mp.get_context(context)
        self.env_fn = env_fn

        dummy_env = env_fn()
        self.observation_space = dummy_env.observation_space
        self.action_space = dummy_env.action_space
        dummy_env.close()
        del dummy_env

        _obs_buffer = create_shared_memory(self.observation_space, n=1, ctx=ctx)
        self.observation = read_from_shared_memory(
            self.observation_space, _obs_buffer, n=1
        )

        self.error_queue = ctx.Queue()
        with clear_mpi_env_vars():
            self.parent_pipe, child_pipe = ctx.Pipe()
            self.process = ctx.Process(
                target=_worker_shared_memory,
                name=f"Worker<{type(self).__name__}>-0",
                args=(
                    0,
                    CloudpickleWrapper(env_fn),
                    child_pipe,
                    self.parent_pipe,
                    _obs_buffer,
                    self.error_queue,
                ),
            )

            self.process.daemon = True
            self.process.start()
            child_pipe.close()

    def reset(self, seed=None, options=None):
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = options

        self.parent_pipe.send(("reset", kwargs))
        result, success = self.parent_pipe.recv()
        result, info = result

        return deepcopy(self.observation)[0], info

    def step(self, action):
        self.parent_pipe.send(("step", action))
        result, success = self.parent_pipe.recv()
        ob, rew, terminated, truncated, info = result
        return deepcopy(self.observation)[0], rew, terminated, truncated, info

    def render(self, *args, **kwargs):
        self.parent_pipe.send(("_call", ("render", args, kwargs)))
        result, success = self.parent_pipe.recv()
        return result

    def call(self, name, *args, **kwargs):
        self.parent_pipe.send(("_call", (name, args, kwargs)))
        result, success = self.parent_pipe.recv()
        return result

    def get_attr(self, name):
        return self.call(name)

    def set_attr(self, name, value):
        self.parent_pipe.send(("_setattr", (name, value)))
        _, success = self.parent_pipe.recv()

    def close(self):
        if self.process is None:
            return

        if self.parent_pipe is not None and not self.parent_pipe.closed:
            self.parent_pipe.send(("close", None))
            self.parent_pipe.recv()
        self.parent_pipe.close()
        self.process.join()


class SubprocVecEnv(VecEnv):
    """
    VecEnv that runs multiple environments in parallel in subproceses and communicates with them via pipes.
    Recommended to use when num_envs > 1 and step() can be a bottleneck.
    """

    def __init__(self, env_fns, spaces=None, context="spawn", in_series=1):
        """
        Arguments:

        env_fns: iterable of callables -  functions that create environments to run in subprocesses. Need to be cloud-pickleable
        in_series: number of environments to run in series in a single process
        (e.g. when len(env_fns) == 12 and in_series == 3, it will run 4 processes, each running 3 envs in series)
        """
        self.waiting = False
        self.closed = False
        self.in_series = in_series
        nenvs = len(env_fns)
        assert (
            nenvs % in_series == 0
        ), "Number of envs must be divisible by number of envs to run in series"
        self.nremotes = nenvs // in_series
        env_fns = np.array_split(env_fns, self.nremotes)
        ctx = mp.get_context(context)
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(self.nremotes)]
        )
        self.ps = [
            ctx.Process(
                target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn))
            )
            for (work_remote, remote, env_fn) in zip(
                self.work_remotes, self.remotes, env_fns
            )
        ]
        for p in self.ps:
            p.daemon = (
                True  # if the main process crashes, we should not cause things to hang
            )
            with clear_mpi_env_vars():
                p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(("get_spaces_spec", None))
        observation_space, action_space, self.spec = self.remotes[0].recv().x
        self.viewer = None
        VecEnv.__init__(self, nenvs, observation_space, action_space)

    def step_async(self, actions):
        self._assert_not_closed()
        actions = np.array_split(actions, self.nremotes)
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        self.waiting = True

    def step_wait(self):
        self._assert_not_closed()
        results = [remote.recv() for remote in self.remotes]
        results = _flatten_list(results)
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return _flatten_obs(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        self._assert_not_closed()
        for remote in self.remotes:
            remote.send(("reset", None))
        obs = [remote.recv() for remote in self.remotes]
        obs = _flatten_list(obs)
        return _flatten_obs(obs)

    def close_extras(self):
        self.closed = True
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()

    def get_images(self):
        self._assert_not_closed()
        for pipe in self.remotes:
            pipe.send(("render", None))
        imgs = [pipe.recv() for pipe in self.remotes]
        imgs = _flatten_list(imgs)
        return imgs

    def _assert_not_closed(self):
        assert (
            not self.closed
        ), "Trying to operate on a SubprocVecEnv after calling close()"

    def __del__(self):
        if not self.closed:
            self.close()


def _flatten_obs(obs):
    assert isinstance(obs, (list, tuple))
    assert len(obs) > 0

    if isinstance(obs[0], dict):
        keys = obs[0].keys()
        return {k: np.stack([o[k] for o in obs]) for k in keys}
    else:
        return np.stack(obs)


def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                if len(data) == 0:
                    ret = env.reset()
                else:
                    ret = env.reset(**data)
                if isinstance(ret, tuple) and len(ret) == 2:
                    observation, info = ret
                else:
                    obesrvation = ret
                    info = {}
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(((None, info), True))
            elif command == "step":
                ret = env.step(data)
                observation, reward, terminated, truncated, info = ret
                if terminated or truncated:
                    old_observation, old_info = observation, info
                    observation, info = env.reset()
                    info["final_observation"] = old_observation
                    info["final_info"] = old_info
                write_to_shared_memory(
                    observation_space, index, observation, shared_memory
                )
                pipe.send(((None, reward, terminated, truncated, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_spaces":
                pipe.send(
                    ((data[0] == observation_space, data[1] == env.action_space), True)
                )
            else:
                raise RuntimeError(
                    f"Received unknown command `{command}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, `_call`, "
                    "`_setattr`, `_check_spaces`}."
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
