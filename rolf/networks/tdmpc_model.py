import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym.spaces

from .utils import MLP, get_activation, weight_init
from .distributions import TanhNormal


class TDMPCModel(nn.Module):
    """Task-Oriented Latent Dynamics model."""

    def __init__(self, cfg, ob_space, ac_dim, dtype):
        super().__init__()
        self._cfg = cfg
        self._ob_space = ob_space

        self.encoder = Encoder(cfg.encoder, ob_space, cfg.state_dim)
        self.dynamics = MLP(
            cfg.state_dim + ac_dim,
            cfg.state_dim,
            [cfg.num_units] * cfg.num_layers,
            cfg.dense_act,
        )
        self.reward = Critic(
            cfg.state_dim + ac_dim,
            [cfg.num_units] * cfg.num_layers,
            1,
            cfg.dense_act,
        )
        input_dim = (
            gym.spaces.flatdim(ob_space)
            if hasattr(cfg, "sac") and cfg.sac
            else cfg.state_dim
        )
        self.critic = Critic(
            input_dim + ac_dim,
            [cfg.num_units] * cfg.num_layers,
            2,
            cfg.dense_act,
        )

    def imagine_step(self, state, ac):
        out = torch.cat([state, ac], dim=-1)
        return self.dynamics(out), self.reward(out).squeeze(-1)

    def load_state_dict(self, ckpt):
        try:
            super().load_state_dict(ckpt, strict=False)
        except:
            from collections import OrderedDict

            ckpt = OrderedDict(
                [(k, v) for k, v in ckpt.items() if not k.startswith("critic")]
            )
            super().load_state_dict(ckpt, strict=False)


class Encoder(nn.Module):
    def __init__(self, cfg, ob_space, state_dim):
        super().__init__()
        self._ob_space = ob_space
        self.encoders = nn.ModuleDict()
        enc_dim = 0
        for k, v in ob_space.spaces.items():
            if len(v.shape) == 3:
                self.encoders[k] = ConvEncoder(
                    cfg.image_shape,
                    cfg.kernel_size,
                    cfg.stride,
                    cfg.conv_dim,
                    cfg.cnn_act,
                )
            elif len(v.shape) == 1:
                self.encoders[k] = DenseEncoder(
                    gym.spaces.flatdim(v),
                    cfg.embed_dim,
                    cfg.hidden_dims,
                    cfg.dense_act,
                )
            else:
                raise ValueError("Observations should be either vectors or RGB images")
            enc_dim += self.encoders[k].output_dim
        self.fc = MLP(enc_dim, state_dim, [], cfg.dense_act)
        self.act = get_activation(cfg.dense_act)
        self.output_dim = state_dim

    def forward(self, ob):
        embeddings = [self.act(self.encoders[k](v)) for k, v in ob.items()]
        return self.fc(torch.cat(embeddings, -1))
        # return torch.cat(embeddings, -1)


class DenseEncoder(nn.Module):
    def __init__(self, shape, embed_dim, hidden_dims, activation):
        super().__init__()
        self.fc = MLP(shape, embed_dim, hidden_dims, activation)
        self.output_dim = embed_dim

    def forward(self, ob):
        return self.fc(ob)


class ConvEncoder(nn.Module):
    def __init__(self, shape, kernel_size, stride, conv_dim, activation):
        super().__init__()
        convs = []
        activation = get_activation(activation)
        h, w, d_prev = shape
        for k, s, d in zip(kernel_size, stride, conv_dim):
            convs.append(nn.Conv2d(d_prev, d, k, s))
            convs.append(activation)
            d_prev = d
            h = int(np.floor((h - k) / s + 1))
            w = int(np.floor((w - k) / s + 1))

        self.convs = nn.Sequential(*convs)
        self.output_dim = h * w * d_prev
        self.apply(weight_init)

    def forward(self, ob):
        shape = list(ob.shape[:-3]) + [-1]
        x = ob.reshape([-1] + list(ob.shape[-3:]))
        x = x.permute(0, 3, 1, 2)
        x = self.convs(x)
        return x.reshape(shape)


class Critic(nn.Module):
    def __init__(self, input_dim, hidden_dims, ensemble, activation):
        super().__init__()
        self._ensemble = ensemble
        h = hidden_dims
        assert len(h) > 0
        self.fcs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, h[0]),
                    nn.LayerNorm(h[0]),
                    nn.Tanh(),
                    MLP(h[0], 1, h[1:], activation, small_weight=True),
                )
                for _ in range(ensemble)
            ]
        )

    def forward(self, state, ac=None):
        if ac is not None:
            state = torch.cat([state, ac], dim=-1)
        q = [fc(state).squeeze(-1) for fc in self.fcs]
        return q[0] if self._ensemble == 1 else q


class ActionDecoder(nn.Module):
    """MLP decoder returns tanh normal distribution of action."""

    def __init__(self, input_dim, ac_dim, hidden_dims, activation):
        super().__init__()
        self.fc = MLP(input_dim, ac_dim, hidden_dims, activation)

    def forward(self, state, std=0, return_dist=False):
        mean = self.fc(state)
        if std == 0:
            return torch.tanh(mean)
        std = std * torch.ones_like(mean)
        dist = TanhNormal(mean, std, 1)
        if return_dist:
            return dist
        return dist.rsample()

        # Original code uses Truncated Normal
        # mean = torch.tanh(self.fc(state))
        # if std > 0:
        #     std = std * torch.ones_like(mean)
        #     noise = std * torch.randn_like(std)
        #     noise = torch.clamp(noise, -0.3, 0.3)
        #     x = mean + noise
        #     clamped_x = torch.clamp(x, -1 + 1e-6, 1 - 1e-6)
        #     x = x - x.detach() + clamped_x.detach()
        #     return x
        # return mean

    def act(self, state, std=0, cond=None, deterministic=False, return_dist=False):
        """Samples action for rollout."""
        if cond is not None:
            state = torch.cat([state, cond], -1)
        dist = self.forward(state, std=std, return_dist=True)
        action = dist.mode() if deterministic else dist.rsample()
        if return_dist:
            return action, dist
        return action


class LSTMEncoder(nn.Module):
    """LSTM encoder returns tanh normal distribution of latents."""

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dim,
        num_layers,
        batch_first=True,
        log_std=True,
    ):
        super().__init__()

        init_std = 5
        self._raw_init_std = np.log(np.exp(init_std) - 1)
        self._min_std = 1e-4
        self._mean_scale = 5
        self._log_std = log_std

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=batch_first)
        self.output_layer = nn.Linear(hidden_dim, output_dim * 2)
        self.input_dim = input_dim

    def forward(self, input):
        out, _ = self.lstm(
            input.reshape(input.shape[0] * input.shape[1], -1, self.input_dim)
        )
        out = out[:, -1, :].view(input.shape[0], input.shape[1], -1)
        if self._log_std:
            mean, log_std = self.output_layer(out).chunk(2, dim=-1)
            std = torch.exp(torch.clamp(log_std, min=-10, max=2)) + self._min_std
        else:
            mean, std = self.output_layer(out).chunk(2, dim=-1)
            std = F.softplus(std + self._raw_init_std) + self._min_std
        mean = self._mean_scale * (mean / self._mean_scale).tanh()
        return TanhNormal(mean, std, event_dim=1)

    def act(self, state, cond=None, deterministic=False, return_dist=False):
        """Samples action for rollout."""
        if cond is not None:
            state = torch.cat([state, cond], -1)
        dist = self.forward(state)
        action = dist.mode() if deterministic else dist.rsample()
        if return_dist:
            return action, dist
        return action
