# SAC training code reference
# https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym.spaces

from .base_agent import BaseAgent
from .dataset import ReplayBuffer, RandomSampler
from ..networks import Actor, Critic
from ..utils.info_dict import Info
from ..utils.logger import logger
from ..utils.mpi import mpi_average, mpi_sum
from ..utils.pytorch import (
    optimizer_cuda,
    count_parameters,
    compute_gradient_norm,
    compute_weight_norm,
    sync_networks,
    sync_grads,
    to_tensor,
)


class SACAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        self._target_entropy = -gym.spaces.flatdim(ac_space)
        self._log_alpha = torch.zeros(1, requires_grad=True, device=config.device)
        self._alpha_optim = optim.Adam([self._log_alpha], lr=config.alpha_lr)

        # build up networks
        self._actor = Actor(config, ob_space, ac_space, config.tanh_policy)
        self._critic1 = Critic(config, ob_space, ac_space)
        self._critic2 = Critic(config, ob_space, ac_space)

        # build up target networks
        self._critic1_target = Critic(config, ob_space, ac_space)
        self._critic2_target = Critic(config, ob_space, ac_space)
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())
        self._network_cuda(config.device)

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.actor_lr)
        self._critic1_optim = optim.Adam(
            self._critic1.parameters(), lr=config.critic_lr
        )
        self._critic2_optim = optim.Adam(
            self._critic2.parameters(), lr=config.critic_lr
        )

        sampler = RandomSampler(image_crop_size=config.encoder_image_size)
        buffer_keys = ["ob", "ac", "done", "rew"]
        self._buffer = ReplayBuffer(
            buffer_keys, config.buffer_size, sampler.sample_func
        )

        self._log_creation()

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a SAC agent")
            logger.info("The actor has %d parameters", count_parameters(self._actor))
            logger.info(
                "The critic1 has %d parameters", count_parameters(self._critic1)
            )
            logger.info(
                "The critic2 has %d parameters", count_parameters(self._critic2)
            )

    def store_episode(self, rollouts):
        self._num_updates = mpi_sum(len(rollouts["ac"])) // self._config.num_workers // self._config.actor_update_freq
        self._buffer.store_episode(rollouts)

    def state_dict(self):
        return {
            "log_alpha": self._log_alpha.cpu().detach().numpy(),
            "actor_state_dict": self._actor.state_dict(),
            "critic1_state_dict": self._critic1.state_dict(),
            "critic2_state_dict": self._critic2.state_dict(),
            "alpha_optim_state_dict": self._alpha_optim.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "critic1_optim_state_dict": self._critic1_optim.state_dict(),
            "critic2_optim_state_dict": self._critic2_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        if "log_alpha" not in ckpt:
            missing = self._actor.load_state_dict(ckpt["actor_state_dict"], strict=False)
            for missing_key in missing.missing_keys:
                if 'stds' not in missing_key:
                    logger.warn("Missing key", missing_key)
            if len(missing.unexpected_keys) > 0:
                logger.warn("Unexpected keys", missing.unexpected_keys)
            self._network_cuda(self._config.device)
            return

        self._log_alpha.data = torch.tensor(
            ckpt["log_alpha"], requires_grad=True, device=self._config.device
        )
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._critic1.load_state_dict(ckpt["critic1_state_dict"])
        self._critic2.load_state_dict(ckpt["critic2_state_dict"])
        self._critic1_target.load_state_dict(self._critic1.state_dict())
        self._critic2_target.load_state_dict(self._critic2.state_dict())
        self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
        self._network_cuda(self._config.device)

        self._alpha_optim.load_state_dict(ckpt["alpha_optim_state_dict"])
        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self._critic1_optim.load_state_dict(ckpt["critic1_optim_state_dict"])
        self._critic2_optim.load_state_dict(ckpt["critic2_optim_state_dict"])
        optimizer_cuda(self._alpha_optim, self._config.device)
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic1_optim, self._config.device)
        optimizer_cuda(self._critic2_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._critic1.to(device)
        self._critic2.to(device)
        self._critic1_target.to(device)
        self._critic2_target.to(device)

    def sync_networks(self):
        sync_networks(self._actor)
        sync_networks(self._critic1)
        sync_networks(self._critic2)

    def train(self):
        train_info = Info()

        for _ in range(self._num_updates):
            transitions = self._buffer.sample(self._config.batch_size)
            _train_info = self._update_network(transitions)
            train_info.add(_train_info)

            self._soft_update_target_network(
                self._critic1_target, self._critic1, self._config.critic_soft_update_weight
            )
            self._soft_update_target_network(
                self._critic2_target, self._critic2, self._config.critic_soft_update_weight
            )

        train_info.add(
            {
                "actor_grad_norm": compute_gradient_norm(self._actor),
                "actor_weight_norm": compute_weight_norm(self._actor),
                "critic1_grad_norm": compute_gradient_norm(self._critic1),
                "critic1_weight_norm": compute_weight_norm(self._critic1),
                "critic2_grad_norm": compute_gradient_norm(self._critic2),
                "critic2_weight_norm": compute_weight_norm(self._critic2),
            }
        )
        return train_info.get_dict(only_scalar=True)

    def _update_network(self, transitions):
        info = Info()

        # pre-process observations
        o, o_next = transitions["ob"], transitions["ob_next"]
        o = self.normalize(o)
        o_next = self.normalize(o_next)

        bs = len(transitions["done"])
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions["ac"])
        done = _to_tensor(transitions["done"]).reshape(bs, 1)
        rew = _to_tensor(transitions["rew"]).reshape(bs, 1)

        # update alpha
        actions_real, _, log_pi, _ = self._actor.act(o, return_log_prob=True)
        alpha_loss = -(
            self._log_alpha.exp() * (log_pi + self._target_entropy).detach()
        ).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()
        alpha = self._log_alpha.exp()

        # the actor loss
        entropy_loss = (alpha * log_pi).mean()
        actor_loss = -torch.min(
            self._critic1(o, actions_real), self._critic2(o, actions_real)
        ).mean()
        info["entropy_alpha"] = alpha.cpu().item()
        info["entropy_loss"] = entropy_loss.cpu().item()
        info["actor_loss"] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # calculate the target Q value function
        with torch.no_grad():
            actions_next, _, log_pi_next, _ = self._actor.act(o_next, return_log_prob=True)
            q_next_value1 = self._critic1_target(o_next, actions_next)
            q_next_value2 = self._critic2_target(o_next, actions_next)
            q_next_value = torch.min(q_next_value1, q_next_value2) - alpha * log_pi_next
            target_q_value = (
                rew * self._config.reward_scale
                + (1 - done) * self._config.rl_discount_factor * q_next_value
            )
            target_q_value = target_q_value.detach()
            ## clip the q value
            # clip_return = 10 / (1 - self._config.rl_discount_factor)
            # target_q_value = torch.clamp(target_q_value, -clip_return, clip_return)

        # the q loss
        real_q_value1 = self._critic1(o, ac)
        real_q_value2 = self._critic2(o, ac)
        critic1_loss = 0.5 * (target_q_value - real_q_value1).pow(2).mean()
        critic2_loss = 0.5 * (target_q_value - real_q_value2).pow(2).mean()

        info["min_target_q"] = target_q_value.min().cpu().item()
        info["target_q"] = target_q_value.mean().cpu().item()
        info["min_real1_q"] = real_q_value1.min().cpu().item()
        info["min_real2_q"] = real_q_value2.min().cpu().item()
        info["real1_q"] = real_q_value1.mean().cpu().item()
        info["real2_q"] = real_q_value2.mean().cpu().item()
        info["critic1_loss"] = critic1_loss.cpu().item()
        info["critic2_loss"] = critic2_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._config.max_grad_norm)
        sync_grads(self._actor)
        self._actor_optim.step()

        # update the critic
        self._critic1_optim.zero_grad()
        critic1_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._critic1.parameters(), self._config.max_grad_norm)
        sync_grads(self._critic1)
        self._critic1_optim.step()

        self._critic2_optim.zero_grad()
        critic2_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._critic2.parameters(), self._config.max_grad_norm)
        sync_grads(self._critic2)
        self._critic2_optim.step()

        # include info from policy
        info.add(self._actor.info)

        return mpi_average(info.get_dict(only_scalar=True))
