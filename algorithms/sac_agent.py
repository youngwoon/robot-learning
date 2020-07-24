# SAC training code reference
# https://github.com/vitchyr/rlkit/blob/master/rlkit/torch/sac/sac.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym.spaces

from .base_agent import BaseAgent
from .dataset import ReplayBuffer, RandomSampler, ReplayBufferPerStep
from ..networks import Actor, Critic
from ..utils.info_dict import Info
from ..utils.logger import logger
from ..utils.mpi import mpi_average, mpi_sum
from ..utils.gym_env import spaces_to_shapes
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
    def __init__(self, config, ob_space, ac_space, env_ob_space):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        if config.target_entropy is not None:
            self._target_entropy = config.target_entropy
        else:
            self._target_entropy = -gym.spaces.flatdim(ac_space)
        self._log_alpha = torch.tensor(
            np.log(config.alpha_init_temperature),
            requires_grad=True,
            device=config.device,
        )

        # build up networks
        self._actor = Actor(config, ob_space, ac_space, config.tanh_policy)
        self._critic = Critic(config, ob_space, ac_space)

        # build up target networks
        self._critic_target = Critic(config, ob_space, ac_space)
        self._network_cuda(config.device)
        self._copy_target_network(self._critic_target, self._critic)
        self._actor.encoder.copy_conv_weights_from(self._critic.encoder)

        # optimizers
        self._alpha_optim = optim.Adam(
            [self._log_alpha], lr=config.alpha_lr, betas=(0.5, 0.999)
        )
        self._actor_optim = optim.Adam(
            self._actor.parameters(), lr=config.actor_lr, betas=(0.9, 0.999)
        )
        self._critic_optim = optim.Adam(
            self._critic.parameters(), lr=config.critic_lr, betas=(0.9, 0.999)
        )

        # per-episode replay buffer
        sampler = RandomSampler(image_crop_size=config.encoder_image_size)
        buffer_keys = ["ob", "ob_next", "ac", "done", "rew"]
        self._buffer = ReplayBuffer(
            buffer_keys, config.buffer_size, sampler.sample_func
        )

        # per-step replay buffer
        # shapes = {
        #     "ob": spaces_to_shapes(env_ob_space),
        #     "ob_next": spaces_to_shapes(env_ob_space),
        #     "ac": spaces_to_shapes(ac_space),
        #     "done": [1],
        #     "rew": [1],
        # }
        # self._buffer = ReplayBufferPerStep(
        #     shapes, config.buffer_size, config.encoder_image_size
        # )

        self._update_iter = 0

        self._log_creation()

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a SAC agent")
            logger.info("The actor has %d parameters", count_parameters(self._actor))
            logger.info("The critic has %d parameters", count_parameters(self._critic))

    def store_episode(self, rollouts):
        self._num_updates = (
            mpi_sum(len(rollouts["ac"]))
            // self._config.num_workers
            // self._config.actor_update_freq
        )
        self._buffer.store_episode(rollouts)

    def state_dict(self):
        return {
            "log_alpha": self._log_alpha.cpu().detach().numpy(),
            "actor_state_dict": self._actor.state_dict(),
            "critic_state_dict": self._critic.state_dict(),
            "alpha_optim_state_dict": self._alpha_optim.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "critic_optim_state_dict": self._critic_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        if "log_alpha" not in ckpt:
            missing = self._actor.load_state_dict(
                ckpt["actor_state_dict"], strict=False
            )
            for missing_key in missing.missing_keys:
                if "stds" not in missing_key:
                    logger.warn("Missing key", missing_key)
            if len(missing.unexpected_keys) > 0:
                logger.warn("Unexpected keys", missing.unexpected_keys)
            self._network_cuda(self._config.device)
            return

        self._log_alpha.data = torch.tensor(
            ckpt["log_alpha"], requires_grad=True, device=self._config.device
        )
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._critic.load_state_dict(ckpt["critic_state_dict"])
        self._copy_target_network(self._critic_target, self._critic)
        self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
        self._network_cuda(self._config.device)

        self._alpha_optim.load_state_dict(ckpt["alpha_optim_state_dict"])
        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self._critic_optim.load_state_dict(ckpt["critic_optim_state_dict"])
        optimizer_cuda(self._alpha_optim, self._config.device)
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._critic.to(device)
        self._critic_target.to(device)

    def sync_networks(self):
        sync_networks(self._actor)
        sync_networks(self._critic)

    def train(self):
        train_info = Info()

        self._num_updates = 1
        for _ in range(self._num_updates):
            transitions = self._buffer.sample(self._config.batch_size)
            _train_info = self._update_network(transitions)
            train_info.add(_train_info)

        # slow!
        # train_info.add(
        #    {
        #        "actor_grad_norm": compute_gradient_norm(self._actor),
        #        "actor_weight_norm": compute_weight_norm(self._actor),
        #        "critic_grad_norm": compute_gradient_norm(self._critic),
        #        "critic_weight_norm": compute_weight_norm(self._critic),
        #    }
        # )
        return mpi_average(train_info.get_dict(only_scalar=True))

    def _update_actor_and_alpha(self, o):
        info = Info()

        actions_real, _, log_pi, _ = self._actor.act(
            o, return_log_prob=True, detach_conv=True
        )
        alpha = self._log_alpha.exp()

        # the actor loss
        entropy_loss = (alpha.detach() * log_pi).mean()
        actor_loss = -torch.min(*self._critic(o, actions_real, detach_conv=True)).mean()
        info["entropy_alpha"] = alpha.cpu().item()
        info["entropy_loss"] = entropy_loss.cpu().item()
        info["actor_loss"] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self._actor)
        self._actor_optim.step()

        # update alpha
        alpha_loss = -(alpha * (log_pi + self._target_entropy).detach()).mean()
        self._alpha_optim.zero_grad()
        alpha_loss.backward()
        self._alpha_optim.step()

        return info

    def _update_critic(self, o, ac, rew, o_next, done):
        info = Info()

        # calculate the target Q value function
        with torch.no_grad():
            alpha = self._log_alpha.exp().detach()
            actions_next, _, log_pi_next, _ = self._actor.act(
                o_next, return_log_prob=True
            )
            q_next_value1, q_next_value2 = self._critic_target(o_next, actions_next)
            q_next_value = torch.min(q_next_value1, q_next_value2) - alpha * log_pi_next
            target_q_value = (
                rew * self._config.reward_scale
                + (1 - done) * self._config.rl_discount_factor * q_next_value
            )

        # the q loss
        real_q_value1, real_q_value2 = self._critic(o, ac)
        critic1_loss = F.mse_loss(target_q_value, real_q_value1)
        critic2_loss = F.mse_loss(target_q_value, real_q_value2)
        critic_loss = critic1_loss + critic2_loss

        # update the critic
        self._critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self._critic)
        self._critic_optim.step()

        info["min_target_q"] = target_q_value.min().cpu().item()
        info["target_q"] = target_q_value.mean().cpu().item()
        info["min_real1_q"] = real_q_value1.min().cpu().item()
        info["min_real2_q"] = real_q_value2.min().cpu().item()
        info["real1_q"] = real_q_value1.mean().cpu().item()
        info["real2_q"] = real_q_value2.mean().cpu().item()
        info["critic1_loss"] = critic1_loss.cpu().item()
        info["critic2_loss"] = critic2_loss.cpu().item()

        return info

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
        done = _to_tensor(transitions["done"]).reshape(bs, 1).float()
        rew = _to_tensor(transitions["rew"]).reshape(bs, 1)

        self._update_iter += 1

        critic_train_info = self._update_critic(o, ac, rew, o_next, done)
        info.add(critic_train_info)

        if self._update_iter % self._config.actor_update_freq == 0:
            actor_train_info = self._update_actor_and_alpha(o)
            info.add(actor_train_info)

        if self._update_iter % self._config.critic_target_update_freq == 0:
            for i, fc in enumerate(self._critic.fcs):
                self._soft_update_target_network(
                    self._critic_target.fcs[i],
                    fc,
                    self._config.critic_soft_update_weight,
                )
            self._soft_update_target_network(
                self._critic_target.encoder,
                self._critic.encoder,
                self._config.encoder_soft_update_weight,
            )

        return info.get_dict(only_scalar=True)
