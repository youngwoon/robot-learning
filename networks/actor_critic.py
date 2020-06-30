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
from ..utils.pytorch import to_tensor
from ..utils.logger import logger


class Actor(nn.Module):
    def __init__(self, config, ob_space, ac_space, tanh_policy, encoder=None):
        super().__init__()
        self._config = config
        self._ac_space = ac_space
        self._activation_fn = getattr(F, config.policy_activation)
        self._tanh = tanh_policy
        self._gaussian = config.gaussian_policy

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = Encoder(config, ob_space)

        self.fc = MLP(
            config, self.encoder.output_dim, config.policy_mlp_dim[-1], config.policy_mlp_dim[:-1]
        )

        self.fcs = nn.ModuleDict()
        self._dists = {}
        for k, v in ac_space.spaces.items():
            if isinstance(v, gym.spaces.Box): # and self._gaussian:  # for convenience to transfer bc policy
                self.fcs.update(
                    {k: MLP(config, config.policy_mlp_dim[-1], gym.spaces.flatdim(v) * 2)}
                )
            else:
                self.fcs.update(
                    {k: MLP(config, config.policy_mlp_dim[-1], gym.spaces.flatdim(v))}
                )

            if isinstance(v, gym.spaces.Box):
                if self._gaussian:
                    self._dists[k] = lambda m, s: FixedNormal(m, s)
                else:
                    self._dists[k] = lambda m, s: Identity(m)
            else:
                self._dists[k] = lambda m, s: FixedCategorical(logits=m)

    @property
    def info(self):
        return {}

    def forward(self, ob: dict, detach_conv=False):
        out = self.encoder(ob, detach_conv=detach_conv)
        out = self._activation_fn(self.fc(out))

        means, stds = OrderedDict(), OrderedDict()
        for k, v in self._ac_space.spaces.items():
            if isinstance(v, gym.spaces.Box): # and self._gaussian:
                mean, log_std = self.fcs[k](out).chunk(2, dim=-1)
                log_std_min, log_std_max = -10, 2
                log_std = torch.tanh(log_std)
                log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
                std = log_std.exp()
            else:
                mean, std = self.fcs[k](out), None

            means[k] = mean
            stds[k] = std

        return means, stds

    def act(self, ob, deterministic=False, activations=None, return_log_prob=False, detach_conv=False):
        """ Samples action for rollout. """
        means, stds = self.forward(ob, detach_conv=detach_conv)

        dists = OrderedDict()
        for k in means.keys():
            dists[k] = self._dists[k](means[k], stds[k])

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
    def __init__(self, config, ob_space, ac_space=None, encoder=None):
        super().__init__()
        self._config = config

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = Encoder(config, ob_space)

        input_dim = self.encoder.output_dim
        if ac_space is not None:
            input_dim += gym.spaces.flatdim(ac_space)

        self.fcs = nn.ModuleList()

        for _ in range(config.critic_ensemble):
            self.fcs.append(MLP(config, input_dim, 1, config.critic_mlp_dim))

    def forward(self, ob, ac=None, detach_conv=False):
        out = self.encoder(ob, detach_conv=detach_conv)

        if ac is not None:
            out = torch.cat([out, flatten_ac(ac)], dim=-1)
        assert len(out.shape) == 2

        out = [fc(out) for fc in self.fcs]
        if len(out) == 1:
            return out[0]
        return out
