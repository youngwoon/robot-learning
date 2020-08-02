import os
import pickle
import glob

import torch
from torch.utils.data import Dataset
import numpy as np
import gym.spaces

from ..utils.logger import logger
from ..utils.gym_env import get_non_absorbing_state, get_absorbing_state, zero_value


class ExpertDataset(Dataset):
    """ Dataset class for Imitation Learning. """

    def __init__(
        self,
        path,
        subsample_interval=1,
        ac_space=None,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        use_low_level=False,
    ):
        self.train = train  # training set or test set

        self._data = []
        self._ac_space = ac_space

        assert (
            path is not None
        ), "--demo_path should be set (e.g. demos/Sawyer_toy_table)"
        demo_files = self._get_demo_files(path)
        num_demos = 0

        # now load the picked numpy arrays
        for file_path in demo_files:
            with open(file_path, "rb") as f:
                demos = pickle.load(f)
                if not isinstance(demos, list):
                    demos = [demos]

                for demo in demos:
                    if len(demo["obs"]) != len(demo["actions"]) + 1:
                        logger.error(
                            "Mismatch in # of observations (%d) and actions (%d) (%s)",
                            len(demo["obs"]),
                            len(demo["actions"]),
                            file_path,
                        )
                        continue

                    offset = np.random.randint(0, subsample_interval)
                    num_demos += 1

                    if use_low_level:
                        length = len(demo["low_level_obs"])
                        for i in range(offset, length, subsample_interval):
                            transition = {
                                "ob": demo["low_level_obs"][i],
                            }
                            if isinstance(demo["low_level_actions"][i], dict):
                                transition["ac"] = demo["low_level_actions"][i]
                            else:
                                transition["ac"] = gym.spaces.unflatten(
                                    ac_space, demo["low_level_actions"][i]
                                )
                            if i + 1 < length:
                                transition["ob_next"] = demo["low_level_obs"][i + 1]
                            else:
                                transition["ob_next"] = demo["obs"][-1]

                            transition["done"] = 1 if i + 1 == length else 0

                            self._data.append(transition)

                        continue

                    length = len(demo["actions"])
                    for i in range(offset, length, subsample_interval):
                        transition = {
                            "ob": demo["obs"][i],
                            "ob_next": demo["obs"][i + 1],
                        }
                        if isinstance(demo["actions"][i], dict):
                            transition["ac"] = demo["actions"][i]
                        else:
                            transition["ac"] = gym.spaces.unflatten(
                                ac_space, demo["actions"][i]
                            )
                        if "rewards" in demo:
                            transition["rew"] = demo["rewards"][i]
                        if "dones" in demo:
                            transition["done"] = int(demo["dones"][i])
                        else:
                            transition["done"] = 1 if i + 1 == length else 0

                        self._data.append(transition)

        logger.warn(
            "Load %d demonstrations with %d states from %d files",
            num_demos,
            len(self._data),
            len(demo_files),
        )

    def add_absorbing_states(self, ob_space, ac_space):
        new_data = []
        absorbing_state = get_absorbing_state(ob_space)
        absorbing_action = zero_value(ac_space, dtype=np.float32)
        for i in range(len(self._data)):
            transition = self._data[i].copy()
            transition["ob"] = get_non_absorbing_state(self._data[i]["ob"])
            # learn reward for the last transition regardless of timeout (different from paper)
            if self._data[i]["done"]:
                transition["ob_next"] = absorbing_state
                transition["done_mask"] = 0  # -1 absorbing, 0 done, 1 not done
            else:
                transition["ob_next"] = get_non_absorbing_state(
                    self._data[i]["ob_next"]
                )
                transition["done_mask"] = 1  # -1 absorbing, 0 done, 1 not done
            new_data.append(transition)

            if self._data[i]["done"]:
                transition = {
                    "ob": absorbing_state,
                    "ob_next": absorbing_state,
                    "ac": absorbing_action,
                    "rew": np.float64(0.0),
                    "done": 0,
                    "done_mask": -1,  # -1 absorbing, 0 done, 1 not done
                }
                new_data.append(transition)

        self._data = new_data

    def _get_demo_files(self, demo_file_path):
        demos = []
        if not demo_file_path.endswith(".pkl"):
            demo_file_path = demo_file_path + "*.pkl"
        for f in glob.glob(demo_file_path):
            if os.path.isfile(f):
                demos.append(f)
        return demos

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (ob, ac) where target is index of the target class.
        """
        return self._data[index]

    def __len__(self):
        return len(self._data)
