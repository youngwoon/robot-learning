import os
import pickle
import glob

import torch
from torch.utils.data import Dataset
import numpy as np

from ..utils.logger import logger
from ..utils.gym_env import get_non_absorbing_state, get_absorbing_state, zero_value


class ExpertDataset(Dataset):
    """ Dataset class for Imitation Learning. """

    def __init__(
        self,
        path,
        subsample_interval=1,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.train = train  # training set or test set

        self._data = []

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
                    assert len(demo["obs"]) == len(demo["actions"]) + 1, (
                        "# observations (%d) should be # actions (%d) + 1"
                        % (len(demo["obs"]), len(demo["actions"]))
                    )

                    offset = np.random.randint(0, subsample_interval)
                    length = len(demo["actions"])
                    num_demos += 1

                    for i in range(offset, length, subsample_interval):
                        transition = {
                            "ob": demo["obs"][i],
                            "ac": demo["actions"][i],
                            "ob_next": demo["obs"][i + 1],
                        }
                        if "rewards" in demo:
                            transition["rew"] = demo["rewards"][i]
                        if "dones" in demo:
                            transition["done"] = demo["dones"][i]
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
        absorbing_action = zero_value(ac_space)
        for i in range(len(self._data)):
            transition = self._data[i].copy()
            transition["ob"] = get_non_absorbing_state(self._data[i]["ob"])
            # learn reward for the last transition regardless of timeout (different from paper)
            if self._data[i]["done"]:
                transition["ob_next"] = absorbing_state
            else:
                transition["ob_next"] = get_non_absorbing_state(
                    self._data[i]["ob_next"]
                )
            transition["done_mask"] = 0
            new_data.append(transition)

            if self._data[i]["done"]:
                transition = {
                    "ob": absorbing_state,
                    "ob_next": absorbing_state,
                    "ac": absorbing_action,
                    "rew": 0,
                    "done": 0,
                    "done_mask": -1,
                }
                new_data.append(transition)

        self._data = new_data

    def _get_demo_files(self, demo_file_path):
        demos = []
        for f in glob.glob(demo_file_path + "*.pkl"):
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
