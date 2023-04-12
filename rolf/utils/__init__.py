from time import time
from collections import OrderedDict, defaultdict

import numpy as np

from .info_dict import Info, LOG_TYPES
from .logger import Logger
from .normalizer import Normalizer
from .gym_env import make_env


def rmap(func, x):
    """Recursively applies `func` to list or dictionary `x`."""
    if isinstance(x, dict):
        return OrderedDict([(k, rmap(func, v)) for k, v in x.items()])
    if isinstance(x, list):
        return [rmap(func, v) for v in x]
    return func(x)


# Adapted from https://github.com/danijar/dreamer/blob/master/tools.py
class Every(object):
    def __init__(self, every, step=0):
        self._every = every
        self._last = step
        self._step = step

    def __call__(self, step=None):
        if step is not None:
            self._step = step
        else:
            self._step += 1
            step = self._step
        if self._every is None:
            return False
        if self._last is None:
            self._last = step
            return True
        if step >= self._last + self._every:
            self._last += self._every
            return True
        return False


# Adapted from https://github.com/danijar/dreamer/blob/master/tools.py
class Once(object):
    def __init__(self, do=True):
        self._do = do

    def __call__(self):
        do = self._do
        self._do = False
        return do

    def reset(self):
        self._do = True


class StopWatch(object):
    def __init__(self, step=0):
        self._last = step
        self.reset()

    def __call__(self, step):
        t = time()
        fps = (step - self._last) / (t - self._time)
        self._time = t
        self._last = step

        return fps

    def reset(self):
        self._elapsed_times = defaultdict(list)
        self._time = time()
        self._times = {}

    def start(self, key=None):
        self._times[key] = time()

    def stop(self, key=None):
        t = time()
        self._elapsed_times[key].append(t - self._times[key])
        self._times[key] = t
        return self._elapsed_times[key][-1]

    def average(self, key=None):
        return sum(self._elapsed_times[key]) / len(self._elapsed_times[key])


class LinearDecay(object):
    def __init__(self, start, end, interval):
        self._start = start
        self._end = end
        self._interval = interval

    def __call__(self, step):
        mix = np.clip(step / self._interval, 0.0, 1.0)
        return (1.0 - mix) * self._start + mix * self._end
