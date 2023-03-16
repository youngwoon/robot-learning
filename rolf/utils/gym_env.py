"""
Converting dm_control to gym is from https://github.com/denisyarats/dmc2gym
* Observation space has to be gym.spaces.Box or gym.spaces.Dict of multiple gym.spaces.Box
"""

import os
import copy
from collections import deque, OrderedDict

import gym
import numpy as np

from . import Logger
from .subproc_vec_env import SubprocEnv, SubprocVecEnv


def make_env(id, cfg=None, seed=0, wrapper=True):
    """Creates a new environment instance with `id` and `cfg`."""
    # Create a maze environment
    if id == "maze":
        from envs.maze import ACRandMaze0S40Env

        env = ACRandMaze0S40Env(cfg)

    # Create a kitchen environment
    elif id == "kitchen":
        from envs.kitchen import NoGoalKitchenEnv, KitchenEnv

        env_class = NoGoalKitchenEnv
        # Skill prior checkpoint has state dim as 60
        if name in ["spirl", "spirl_dreamer", "spirl_tdmpc"]:
            env_class = KitchenEnv
        env = env_class(cfg)

    # Create a calvin environment
    elif id == "calvin":
        from envs.calvin import CalvinEnv

        env = CalvinEnv(**cfg)

    # Create any environment registered in Gym.
    else:
        # Get a default config if not provided
        cfg = cfg or {}
        env = get_gym_env(id, cfg, seed)

    if not wrapper:
        return env

    env_cfg = cfg.get("env_cfg", {})
    # Action repeat.
    env = ActionRepeatWrapper(env, env_cfg.get("action_repeat", 1))
    # Dictionary observation and action.
    env = DictWrapper(env)
    # Normalize state to [-1, 1].
    if env_cfg.get("normalize_state", False):
        env = StateNormWrapper(env)
    # Normalize action to [-1, 1].
    env = ActionNormWrapper(env)
    # Pixel observation and random seed.
    env = PixelWrapper(
        env,
        pixel_ob=env_cfg.get("pixel_ob", False),
        pixel_only=env_cfg.get("pixel_only", False),
        pixel_keys=env_cfg.get("pixel_keys", {"image": [64, 64, 3]}),
        render_kwargs=env_cfg.get("render_kwargs", None),
        seed=seed,
    )
    # Stack frames.
    # FIXME: something weird with state stacking
    pixel_ob = env_cfg.get("pixel_ob", False)
    frame_stack = env_cfg.get("frame_stack", 1)
    if pixel_ob and frame_stack > 1:
        env = FrameStackWrapper(env, frame_stack=frame_stack)

    return env


def get_gym_env(env_id, cfg, seed):
    """Creates gym environment."""
    env_cfg = cfg["env_cfg"]
    env_kwargs = cfg.copy()
    del env_kwargs["env_cfg"]
    del env_kwargs["id"]

    if env_id.startswith("dm."):
        os.environ["MUJOCO_GL"] = "egl"

        # Environment name of dm_control: dm.DOMAIN_NAME.TASK_NAME
        _, domain_name, task_name = env_id.split(".")
        # Use closer camera for quadruped
        camera_id = 2 if domain_name == "quadruped" else 0
        env = DMCGymEnv(
            domain_name=domain_name,
            task_name=task_name,
            seed=seed,
            height=env_cfg.screen_size[0],
            width=env_cfg.screen_size[1],
            camera_id=camera_id,
        )
    elif "IKEA" in env_id or "furniture" in env_id:
        env = gym.make(env_id, **env_kwargs)
    elif env_cfg.subprocess:
        env = SubprocEnv(lambda: gym.make(env_id, **env_kwargs))
    else:
        env = gym.make(env_id, **env_kwargs)

    return env


class DMCGymEnv(gym.Env):
    def __init__(
        self,
        domain_name,
        task_name,
        seed=0,
        height=84,
        width=84,
        camera_id=0,
    ):
        from dm_control import suite

        self._env = suite.load(domain_name, task_name, task_kwargs=dict(random=seed))
        self.render_mode = "rgb_array"
        self.height = height
        self.width = width
        self.camera_id = camera_id

    @property
    def action_space(self):
        spec = self._env.action_spec()
        return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

    @property
    def observation_space(self):
        spec = self._env.observation_spec()
        spaces = OrderedDict()
        from dm_env import specs

        def type_map(dtype):
            if dtype == np.uint8:
                return np.uint8
            elif dtype == np.float32 or dtype == np.float64:
                return np.float32
            else:
                raise NotImplementedError(dtype)

        for k, v in spec.items():
            if type(v) == specs.Array:
                if len(v.shape) == 0:
                    spaces[k] = gym.spaces.Box(-np.inf, np.inf, (1,), type_map(v.dtype))
                else:
                    spaces[k] = gym.spaces.Box(
                        -np.inf, np.inf, v.shape, type_map(v.dtype)
                    )
            elif type(v) == specs.BoundedArray:
                spaces[k] = gym.spaces.Box(
                    -v.minimum, v.maximum, v.shape, type_map(v.dtype)
                )
        return gym.spaces.Dict(spaces)

    def _wrap_observation(self, ob):
        for k in ob:
            if len(ob[k].shape) == 0:
                ob[k] = np.array([ob[k]])
        return ob

    def reset(self, seed=None):
        timestep = self._env.reset()
        return self._wrap_observation(timestep.observation), {}

    def step(self, action):
        timestep = self._env.step(action)
        ob = self._wrap_observation(timestep.observation)
        reward = timestep.reward or 0.0
        terminated = timestep.last()
        truncated = timestep.discount == 0 and not terminated
        return ob, reward, terminated, truncated, {}

    def render(self, *args, **kwargs):
        return self._env.physics.render(
            height=self.height, width=self.width, camera_id=self.camera_id
        )


def make_vec_env(env_id, num_env, cfg=None, seed=0):
    """
    Creates a wrapped SubprocVecEnv using OpenAI gym interface.
    Unity app will use the port number from @cfg.port to (@cfg.port + @num_env - 1).

    Code modified based on
    https://github.com/openai/baselines/blob/master/baselines/common/cmd_util.py

    Args:
        env_id: environment id registered in in `env/__init__.py`.
        num_env: number of environments to launch.
        cfg: general configuration for the environment.
    """
    env_kwargs = {}

    if cfg is not None:
        for key, value in cfg.__dict__.items():
            env_kwargs[key] = value

    def make_thunk(rank):
        new_env_kwargs = env_kwargs.copy()
        if "port" in new_env_kwargs:
            new_env_kwargs["port"] = env_kwargs["port"] + rank
        return lambda: get_gym_env(env_id, new_env_kwargs, seed + rank)

    return SubprocVecEnv([make_thunk(i) for i in range(num_env)])


def cat_spaces(spaces):
    if isinstance(spaces[0], gym.spaces.Box):
        out_space = gym.spaces.Box(
            low=np.concatenate([s.low for s in spaces]),
            high=np.concatenate([s.high for s in spaces]),
        )
    elif isinstance(spaces[0], gym.spaces.Discrete):
        out_space = gym.spaces.Discrete(sum([s.n for s in spaces]))
    return out_space


def stacked_space(space, k):
    if isinstance(space, gym.spaces.Box):
        space_stack = gym.spaces.Box(
            low=np.concatenate([space.low] * k, axis=-1),
            high=np.concatenate([space.high] * k, axis=-1),
            dtype=space.dtype,
        )
    elif isinstance(space, gym.spaces.Discrete):
        space_stack = gym.spaces.Discrete(space.n * k)
    return space_stack


def value_to_space(value):
    if isinstance(value, dict):
        space = gym.spaces.Dict([(k, value_to_space(v)) for k, v in value.items()])
    elif isinstance(value, np.ndarray):
        space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=value.shape)
    else:
        raise NotImplementedError

    return space


def space_to_shape(space):
    if isinstance(space, gym.spaces.Dict):
        return {k: space_to_shape(v) for k, v in space.spaces.items()}
    elif isinstance(space, gym.spaces.Box):
        return space.shape
    elif isinstance(space, gym.spaces.Discrete):
        return [space.n]


class PixelWrapper(gym.Wrapper):
    def __init__(
        self,
        env,
        pixel_ob=False,
        pixel_only=False,
        pixel_keys=None,
        render_kwargs=None,
        seed=None,
    ):
        super().__init__(env)
        self._pixel_ob = pixel_ob
        self._pixel_only = pixel_only
        self._pixel_keys = pixel_keys
        self._render_kwargs = render_kwargs
        self._seed = seed

        wrapped_ob_space = env.observation_space
        if pixel_ob and pixel_only:
            self.observation_space = gym.spaces.Dict()
        else:
            self.observation_space = copy.deepcopy(wrapped_ob_space)

        if pixel_ob:
            for key, shape in pixel_keys.items():
                pixel_ob_space = gym.spaces.Box(
                    low=0, high=255, shape=shape, dtype=np.uint8
                )
                self.observation_space.spaces[key] = pixel_ob_space

    def reset(self, seed=None):
        ob, info = self.env.reset(seed=seed if seed is not None else self._seed)
        self._seed = None  # Initialize seed only once.

        return self._get_obs(ob, reset=True), info

    def step(self, ac):
        ob, reward, terminated, truncated, info = self.env.step(ac)
        return self._get_obs(ob), reward, terminated, truncated, info

    def _get_obs(self, ob, reset=False):
        if not self._pixel_ob:
            return ob

        ob = {} if self._pixel_only else ob.copy()

        if self._render_kwargs is None:
            pixels = self.render()
            if reset:
                pixels = self.render()
            if not isinstance(pixels, list):
                pixels = [pixels]
        else:
            pixels = []
            for render_kwargs in self._render_kwargs:
                pixel = self.render(**render_kwargs)
                if reset:
                    pixel = self.render(**render_kwargs)
                if not isinstance(pixel, list):
                    pixel = [pixel]
                pixels += pixel

        pixel_ob = {key: pixel for key, pixel in zip(self._pixel_keys.keys(), pixels)}
        ob.update(pixel_ob)
        return ob

    def render(self, mode="rgb_array", **kwargs):
        return self.env.render(mode, **kwargs)


class ActionRepeatWrapper(gym.Wrapper):
    def __init__(self, env, action_repeat=1):
        super().__init__(env)
        self._action_repeat = action_repeat

    def step(self, ac):
        reward = 0
        for _ in range(self._action_repeat):
            ob, _reward, terminated, truncated, info = self.env.step(ac)
            done = terminated or truncated
            reward += _reward
            if done:
                break
        return ob, reward, terminated, truncated, info


class DictWrapper(gym.Wrapper):
    """Make observation space and action space gym.spaces.Dict."""

    def __init__(self, env):
        super().__init__(env)

        self._is_ob_dict = isinstance(env.observation_space, gym.spaces.Dict)
        if not self._is_ob_dict:
            self.key = "image" if len(env.observation_space.shape) == 3 else "ob"
            self.observation_space = gym.spaces.Dict({self.key: env.observation_space})
        else:
            self.observation_space = env.observation_space

        self._is_ac_dict = isinstance(env.action_space, gym.spaces.Dict)
        if not self._is_ac_dict:
            self.action_space = gym.spaces.Dict({"ac": env.action_space})
        else:
            self.action_space = env.action_space

    def reset(self, seed=None):
        ob, info = self.env.reset(seed=seed)
        return self._get_obs(ob), info

    def step(self, ac):
        if not self._is_ac_dict:
            ac = ac["ac"]
        ob, reward, terminated, truncated, info = self.env.step(ac)
        return self._get_obs(ob), reward, terminated, truncated, info

    def _get_obs(self, ob):
        if not self._is_ob_dict:
            ob = {self.key: ob}
        return ob


class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, frame_stack=3):
        super().__init__(env)

        # Both observation and action spaces must be gym.spaces.Dict.
        assert isinstance(env.observation_space, gym.spaces.Dict), env.observation_space
        assert isinstance(env.action_space, gym.spaces.Dict), env.action_space

        self._frame_stack = frame_stack
        self._frames = deque([], maxlen=frame_stack)

        ob_space = []
        for k, space in env.observation_space.spaces.items():
            space_stack = stacked_space(space, frame_stack)
            ob_space.append((k, space_stack))
        self.observation_space = gym.spaces.Dict(ob_space)

    def reset(self, seed=None):
        ob, info = self.env.reset(seed=seed)
        for _ in range(self._frame_stack):
            self._frames.append(ob)
        return self._get_obs(), info

    def step(self, ac):
        ob, reward, terminated, truncated, info = self.env.step(ac)
        self._frames.append(ob)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        frames = list(self._frames)
        obs = []
        for k in self.env.observation_space.spaces.keys():
            obs.append((k, np.concatenate([f[k] for f in frames], axis=-1)))
        return OrderedDict(obs)


class ActionNormWrapper(gym.Wrapper):
    """Normalize action space to [-1, 1]."""

    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.action_space, gym.spaces.Dict), env.action_space

        ac_space = []
        self._low = {}
        self._high = {}
        for k, space in env.action_space.spaces.items():
            if isinstance(space, gym.spaces.Box):
                self._low[k] = low = space.low
                self._high[k] = high = space.high
                space = gym.spaces.Box(
                    -np.ones_like(low), np.ones_like(high), dtype=np.float32
                )
            ac_space.append((k, space))
        self.action_space = gym.spaces.Dict(ac_space)

    def step(self, action):
        action = action.copy()
        for k in self._low:
            action[k] = (action[k] + 1) / 2 * (
                self._high[k] - self._low[k]
            ) + self._low[k]
            action[k] = np.clip(action[k], self._low[k], self._high[k])
        return self.env.step(action)


class StateNormWrapper(gym.ObservationWrapper):
    """Normalize state information to [-1, 1], but keeps images in [0, 255]."""

    def __init__(self, env):
        super().__init__(env)

        assert isinstance(env.observation_space, gym.spaces.Dict), env.observation_space

        ob_space = []
        self._low = {}
        self._high = {}
        for k, space in env.observation_space.spaces.items():
            assert isinstance(space, gym.spaces.Box)
            if space.dtype != np.uint8:
                self._low[k] = low = space.low
                self._high[k] = high = space.high
                space = gym.spaces.Box(
                    -np.ones_like(low), np.ones_like(high), dtype=space.dtype
                )
            ob_space.append((k, space))
        self.observation_space = gym.spaces.Dict(ob_space)

    def observation(self, ob):
        """Normalizes states within [-1, 1]."""
        ob = ob.copy()
        for k in self._low:
            ob[k] = ((ob[k] - self._low[k]) / (self._high[k] - self._low[k]) - 0.5) * 2

        return ob
