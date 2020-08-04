from collections import defaultdict

import numpy as np


class Info(object):
    def __init__(self, info=None):
        self._info = defaultdict(list)
        if info:
            self.add(info)

    def add(self, info):
        if info is None:
            return
        if isinstance(info, Info):
            for k, v in info._info.items():
                self._info[k].extend(v)
        elif isinstance(info, dict):
            for k, v in info.items():
                if isinstance(v, list):
                    self._info[k].extend(v)
                else:
                    self._info[k].append(v)
        else:
            raise ValueError("info should be dict or Info (%s)" % info)

    def clear(self):
        self._info = defaultdict(list)

    def get_dict(self, reduction="mean", only_scalar=False):
        ret = {}
        for k, v in self._info.items():
            if np.isscalar(v):
                ret[k] = v
            elif isinstance(v[0], (int, float, bool, np.float32, np.int64, np.ndarray)):
                if "_mean" in k or reduction == "mean":
                    ret[k] = np.mean(v)
                elif reduction == "sum":
                    ret[k] = np.sum(v)
            elif not only_scalar:
                ret[k] = v
        self.clear()
        return ret

    def get_stat(self):
        ret = {}
        for k, v in self._info.items():
            if np.isscalar(v):
                ret[k] = (v, 0)
            elif isinstance(v[0], (int, float, bool, np.float32, np.int64, np.ndarray)):
                ret[k] = (np.mean(v), np.std(v))
        return ret

    def __getitem__(self, key):
        return self._info[key]

    def __setitem__(self, key, value):
        self._info[key].append(value)

    def items(self):
        return self._info.items()
