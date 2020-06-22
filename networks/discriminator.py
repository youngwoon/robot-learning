from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

from .utils import MLP
from ..utils.pytorch import to_tensor


class Discriminator(nn.Module):
    def __init__(self, config, ob_space, ac_space=None):
        super().__init__()
        self._config = config
        self._no_action = ac_space == None

        input_dim = gym.spaces.flatdim(ob_space)
        if not self._no_action:
            input_dim += gym.spaces.flatdim(ac_space)

        self.fc = MLP(
            config,
            input_dim,
            1,
            config.discriminator_mlp_dim,
            getattr(F, config.discriminator_activation),
        )

    def forward(self, ob, ac=None):
        # flatten observation
        ob = list(ob.values())
        if len(ob[0].shape) == 1:
            ob = [x.unsqueeze(0) for x in ob]
        ob = torch.cat(ob, dim=-1)

        if ac is not None:
            # flatten action
            if isinstance(ac, OrderedDict):
                ac = list(ac.values())
                if len(ac[0].shape) == 1:
                    ac = [x.unsqueeze(0) for x in ac]
                ac = torch.cat(ac, dim=-1)
            ob = torch.cat([ob, ac], dim=-1)

        out = self.fc(ob)
        return out
