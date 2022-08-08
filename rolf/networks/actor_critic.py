from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym.spaces

from .distributions import Categorical, Normal, TanhNormal, MixedDistribution
from .distributions import Identity, TanhIdentity
from .utils import MLP, flatten_ac, get_activation
from .encoder import Encoder
from ..utils import Logger


class Actor(nn.Module):
    def __init__(self, cfg, ob_space, ac_space, tanh_policy, encoder=None):
        super().__init__()
        self._cfg = cfg
        self._ac_space = ac_space
        self._activation = get_activation(cfg.policy_activation)
        self._tanh = tanh_policy
        self._gaussian = cfg.gaussian_policy

        # bias to set initial std to `cfg.target_init_std`
        if cfg.use_log_std_bias:
            self._raw_init_std = np.log(np.exp(cfg.target_init_std) - 1)
        else:
            self._raw_init_std = 0
        self._min_std = cfg.min_std
        self._max_std = cfg.max_std

        self.encoder = encoder or Encoder(cfg, ob_space)

        self.fc = MLP(
            self.encoder.output_dim,
            cfg.policy_mlp_dim[-1],
            cfg.policy_mlp_dim[:-1],
            self._activation,
            small_weight=True,
        )

        self.fcs = nn.ModuleDict()
        self._dists = {}
        for k, v in ac_space.spaces.items():
            output_dim = gym.spaces.flatdim(v)
            if isinstance(v, gym.spaces.Box):
                output_dim = gym.spaces.flatdim(v) * 2
            self.fcs[k] = MLP(cfg.policy_mlp_dim[-1], output_dim, [], self._activation)

            if isinstance(v, gym.spaces.Box):
                if self._gaussian:
                    if self._tanh:
                        self._dists[k] = lambda m, s: TanhNormal(m, s, event_dim=1)
                    else:
                        self._dists[k] = lambda m, s: Normal(m, s, event_dim=1)
                else:
                    if self._tanh:
                        self._dists[k] = lambda m, _: TanhIdentity(m)
                    else:
                        self._dists[k] = lambda m, _: Identity(m)
            else:
                self._dists[k] = lambda m, _: Categorical(logits=m)

    @property
    def info(self):
        return {}

    def forward(self, ob: dict, detach_conv=False):
        out = self.encoder(ob, detach_conv=detach_conv)
        out = self._activation(self.fc(out))

        means, stds = OrderedDict(), OrderedDict()
        for k, v in self._ac_space.spaces.items():
            if isinstance(v, gym.spaces.Box):
                mean, std = self.fcs[k](out).chunk(2, dim=-1)
                std = F.softplus(std + self._raw_init_std) + self._min_std
                std = torch.clamp(std, max=self._max_std)
            else:
                mean, std = self.fcs[k](out), None

            means[k] = mean
            stds[k] = std

        return means, stds

    def act(
        self, ob, deterministic=False, ac=None, return_log_prob=False, detach_conv=False
    ):
        """Samples action for rollout."""
        means, stds = self.forward(ob, detach_conv=detach_conv)

        mixed_dist = MixedDistribution(
            OrderedDict([(k, self._dists[k](means[k], stds[k])) for k in means])
        )
        if ac is None:
            ac = mixed_dist.mode() if deterministic else mixed_dist.rsample()
        log_prob = mixed_dist.log_prob(ac) if return_log_prob else None
        entropy = mixed_dist.entropy() if return_log_prob else None

        return ac, log_prob, entropy


class Critic(nn.Module):
    def __init__(self, cfg, ob_space, ac_space=None, encoder=None):
        super().__init__()
        self._cfg = cfg
        self._ac_space = ac_space

        self.encoder = encoder or Encoder(cfg, ob_space)

        input_dim = self.encoder.output_dim
        if ac_space is not None:
            input_dim += gym.spaces.flatdim(ac_space)

        self.fcs = nn.ModuleList()
        for _ in range(cfg.critic_ensemble):
            self.fcs.append(
                MLP(input_dim, 1, cfg.critic_mlp_dim, cfg.policy_activation)
            )

    def forward(self, ob, ac=None, detach_conv=False):
        out = self.encoder(ob, detach_conv=detach_conv)

        if ac is not None:
            out = torch.cat([out, flatten_ac(ac)], dim=-1)
        assert len(out.shape) == 2

        out = [fc(out) for fc in self.fcs]
        if len(out) == 1:
            return out[0]
        return out
