from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym.spaces

from .distributions import (
    FixedCategorical,
    FixedNormal,
    Identity,
    MixedDistribution,
)
from .utils import MLP, flatten_ac
from .encoder import Encoder
from ..utils.pytorch import to_tensor, center_crop
from ..utils.logger import logger


class Actor(nn.Module):
    def __init__(self, config, ob_space, ac_space, tanh_policy):
        super().__init__()
        self._config = config
        self._ac_space = ac_space
        self._activation_fn = getattr(F, config.policy_activation)
        self._tanh = tanh_policy
        self._gaussian = config.gaussian_policy

        self.encoder = Encoder(config, ob_space)

        self.fc = MLP(
            config, self.encoder.output_dim, config.policy_mlp_dim[-1], config.policy_mlp_dim[:-1]
        )
        self.fc_means = nn.ModuleDict()
        self.fc_log_stds = nn.ModuleDict()

        for k, v in ac_space.spaces.items():
            self.fc_means.update(
                {k: MLP(config, config.policy_mlp_dim[-1], gym.spaces.flatdim(v))}
            )
            if isinstance(v, gym.spaces.Box) and self._gaussian:
                self.fc_log_stds.update(
                    {k: MLP(config, config.policy_mlp_dim[-1], gym.spaces.flatdim(v))}
                )

    @property
    def info(self):
        return {}

    def forward(self, ob: dict, detach_conv=False):
        out = self.encoder(ob, detach_conv=detach_conv)
        out = self._activation_fn(self.fc(out))

        means, stds = OrderedDict(), OrderedDict()
        for k, v in self._ac_space.spaces.items():
            mean = self.fc_means[k](out)
            if k in self.fc_log_stds:
                log_std = self.fc_log_stds[k](out)
                log_std = torch.clamp(log_std, -10, 2)
                std = torch.exp(log_std.double())
            else:
                std = None

            means[k] = mean
            stds[k] = std

        return means, stds

    def act(self, ob, deterministic=False, activations=None, return_log_prob=False, detach_conv=False):
        """ Samples action for rollout. """
        means, stds = self.forward(ob, detach_conv=detach_conv)

        dists = OrderedDict()
        for k, v in self._ac_space.spaces.items():
            if isinstance(v, gym.spaces.Box):
                if self._gaussian:
                    dists[k] = FixedNormal(means[k], stds[k])
                else:
                    dists[k] = Identity(means[k])
            else:
                dists[k] = FixedCategorical(logits=means[k])

        actions = OrderedDict()
        mixed_dist = MixedDistribution(dists)
        if activations is None:
            if deterministic:
                activations = mixed_dist.mode()
            else:
                activations = mixed_dist.rsample()

        if return_log_prob:
            log_probs = mixed_dist.log_probs(activations)

        for k, v in self._ac_space.spaces.items():
            z = activations[k]
            if self._tanh and isinstance(v, gym.spaces.Box):
                action = torch.tanh(z)
                if return_log_prob:
                    # follow the Appendix C. Enforcing Action Bounds
                    log_det_jacobian = 2 * (np.log(2.0) - z - F.softplus(-2.0 * z)).sum(
                        dim=-1, keepdim=True
                    )
                    log_probs[k] = log_probs[k] - log_det_jacobian
            else:
                action = z

            actions[k] = action

        if return_log_prob:
            log_probs = torch.cat(list(log_probs.values()), -1).sum(-1, keepdim=True)
            entropy = mixed_dist.entropy()
        else:
            log_probs = None
            entropy = None

        return actions, activations, log_probs, entropy


class Critic(nn.Module):
    def __init__(self, config, ob_space, ac_space=None):
        super().__init__()
        self._config = config

        self.encoder = Encoder(config, ob_space)

        input_dim = self.encoder.output_dim
        if ac_space is not None:
            input_dim += gym.spaces.flatdim(ac_space)
        self.fc = MLP(config, input_dim, 1, config.critic_mlp_dim)

    def forward(self, ob, ac=None, detach_conv=False):
        out = self.encoder(ob, detach_conv=detach_conv)

        if ac is not None:
            out = torch.cat([out, flatten_ac(ac)], dim=-1)
        assert len(out.shape) == 2

        out = self.fc(out)
        return out
