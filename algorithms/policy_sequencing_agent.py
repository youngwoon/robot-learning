from collections import OrderedDict
import os

import numpy as np
import torch

from .base_agent import BaseAgent
from .gail_agent import GAILAgent
from ..utils.info_dict import Info
from ..utils.logger import logger
from ..utils.mpi import mpi_average
from ..utils.pytorch import get_ckpt_path


class PolicySequencingAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, env_ob_space):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        ckpts = config.ps_ckpts.split("/")
        self._agents = [
            GAILAgent(config, ob_space, ac_space, env_ob_space)
            for _ in ckpts
        ]
        for i, ckpt in enumerate(ckpts):
            ckpt_dir = os.path.join(config.ps_dir, ckpt)
            ckpt_path, ckpt_num = get_ckpt_path(ckpt_dir, ckpt_num=None)
            assert ckpt_path is not None

            logger.warn("Load checkpoint %s", ckpt_path)
            self._agents[i].load_state_dict(
                torch.load(ckpt_path, map_location=self._config.device)["agent"]
            )

        self._log_creation()

    def __getitem__(self, key):
        return self._agents[key]

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a policy sequencing agent")

    def store_episode(self, rollouts):
        pass

    def state_dict(self):
        return {
        }

    def load_state_dict(self, ckpt):
        if "rl_agent" in ckpt:
            self._rl_agent.load_state_dict(ckpt["rl_agent"])
        else:
            self._rl_agent.load_state_dict(ckpt)
            self._network_cuda(self._config.device)
            return

    def sync_networks(self):
        self._rl_agent.sync_networks()

    def update_normalizer(self, obs=None):
        pass

    def train(self):
        return {}
