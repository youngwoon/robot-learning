import os
import pickle
import glob

import torch
from torch.utils.data import Dataset
import numpy as np

from ..utils.logger import logger


class ExpertDataset(Dataset):
    """ Dataset class for Imitation Learning. """

    def __init__(
        self, path, subsample_interval=1, train=True, transform=None, target_transform=None, download=False
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
                    assert len(demo["obs"]) == len(demo["actions"]) + 1, \
                        "# observations (%d) should be # actions (%d) + 1" % \
                        (len(demo["obs"]), len(demo["actions"]))

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
