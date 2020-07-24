import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym.spaces
from torch.optim.lr_scheduler import StepLR

from .base_agent import BaseAgent
from .dataset import ReplayBuffer, RandomSampler
from ..networks import Actor, Critic
from ..utils.info_dict import Info
from ..utils.logger import logger
from ..utils.mpi import mpi_average
from ..utils.pytorch import (
    optimizer_cuda,
    count_parameters,
    compute_gradient_norm,
    compute_weight_norm,
    sync_networks,
    sync_grads,
    to_tensor,
    scale_dict_tensor,
)


class DDPGAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, env_ob_space):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        # build up networks
        self._actor = Actor(config, ob_space, ac_space, config.tanh_policy)
        self._critic = Critic(config, ob_space, ac_space)

        # build up target networks
        self._actor_target = Actor(config, ob_space, ac_space, config.tanh_policy)
        self._critic_target = Critic(config, ob_space, ac_space)
        self._network_cuda(self._config.device)
        self._copy_target_network(self._actor_target, self._actor)
        self._copy_target_network(self._critic_target, self._critic)
        self._actor.encoder.copy_conv_weights_from(self._critic.encoder)
        self._actor_target.encoder.copy_conv_weights_from(self._critic_target.encoder)

        # build optimizers
        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.actor_lr)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=config.critic_lr)

        # build learning rate scheduler
        self._actor_lr_scheduler = StepLR(
            self._actor_optim, step_size=self._config.max_global_step // 5, gamma=0.5,
        )

        # per-episode replay buffer
        sampler = RandomSampler(image_crop_size=config.encoder_image_size)
        buffer_keys = ["ob", "ob_next", "ac", "done", "done_mask", "rew"]
        self._buffer = ReplayBuffer(
            buffer_keys, config.buffer_size, sampler.sample_func
        )

        self._update_iter = 0
        self._predict_reward = None

        self._log_creation()

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a DDPG agent")
            logger.info("The actor has %d parameters", count_parameters(self._actor))
            logger.info("The critic has %d parameters", count_parameters(self._critic))

    def act(self, ob, is_train=True):
        """ Returns action and the actor's activation given an observation @ob. """
        ac, activation = super().act(ob, is_train=is_train)

        if not is_train:
            return ac, activation

        if self._config.epsilon_greedy:
            if np.random.uniform() < self._config.epsilon_greedy_eps:
                for k, v in self._ac_space.spaces.items():
                    ac[k] = v.sample()
                return ac, activation

        for k, v in self._ac_space.spaces.items():
            if isinstance(v, gym.spaces.Box):
                ac[k] += self._config.policy_exploration_noise * np.random.randn(
                    *ac[k].shape
                )
                ac[k] = np.clip(ac[k], v.low, v.high)

        return ac, activation

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts)

    def state_dict(self):
        return {
            "update_iter": self._update_iter,
            "actor_state_dict": self._actor.state_dict(),
            "critic_state_dict": self._critic.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "critic_optim_state_dict": self._critic_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        if "critic_state_dict" not in ckpt:
            missing = self._actor.load_state_dict(
                ckpt["actor_state_dict"], strict=False
            )
            self._copy_target_network(self._actor_target, self._actor)
            self._network_cuda(self._config.device)
            return

        self._update_iter = ckpt["update_iter"]
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._critic.load_state_dict(ckpt["critic_state_dict"])
        self._copy_target_network(self._actor_target, self._actor)
        self._copy_target_network(self._critic_target, self._critic)
        self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
        self._network_cuda(self._config.device)

        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self._critic_optim.load_state_dict(ckpt["critic_optim_state_dict"])
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._critic.to(device)
        self._actor_target.to(device)
        self._critic_target.to(device)

    def sync_networks(self):
        sync_networks(self._actor)
        sync_networks(self._critic)
        sync_networks(self._actor_target)
        sync_networks(self._critic_target)

    def train(self):
        train_info = Info()

        self._num_updates = 1
        for _ in range(self._num_updates):
            self._actor_lr_scheduler.step()
            transitions = self._buffer.sample(self._config.batch_size)
            train_info.add(self._update_network(transitions))

        return train_info.get_dict()

    def _update_actor(self, o, mask):
        info = Info()

        # the actor loss
        actions_real, _, _, _ = self._actor.act(
            o, return_log_prob=False, detach_conv=True
        )

        q_pred = self._critic(o, actions_real, detach_conv=True)
        if self._config.critic_ensemble > 1:
            q_pred = q_pred[0]

        if self._config.absorbing_state:
            # do not update the actor for absorbing states
            a_mask = 1.0 - torch.clamp(-mask, min=0)  # 0 absorbing, 1 done/not done
            actor_loss = -(q_pred * a_mask).sum()
            if a_mask.sum() > 1e-8:
                actor_loss /= a_mask.sum()
        else:
            actor_loss = -q_pred.mean()
        info["actor_loss"] = actor_loss.cpu().item()

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        if self._config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor.parameters(), self._config.max_grad_norm
            )
        sync_grads(self._actor)
        self._actor_optim.step()

        return info

    def _update_critic(self, o, ac, rew, o_next, mask):
        info = Info()

        # calculate the target Q value function
        with torch.no_grad():
            actions_next, _, _, _ = self._actor_target.act(
                o_next, return_log_prob=False
            )

            # TD3 adds noise to action
            if self._config.critic_ensemble > 1:
                for k in self._ac_space.spaces.keys():
                    noise = (
                        torch.randn_like(actions_next[k]) * self._config.policy_noise
                    ).clamp(
                        -self._config.policy_noise_clip, self._config.policy_noise_clip
                    )
                    actions_next[k] = (actions_next[k] + noise).clamp(-1, 1)

                if self._config.absorbing_state:
                    a_mask = torch.clamp(mask, min=0)  # 0 absorbing/done, 1 not done
                    masked_actions_next = scale_dict_tensor(actions_next, a_mask)
                    q_next_values = self._critic_target(o_next, masked_actions_next)
                else:
                    q_next_values = self._critic_target(o_next, actions_next)

                q_next_value = torch.min(*q_next_values)

            else:
                q_next_value = self._critic_target(o_next, actions_next)

            # For IL, use IL reward
            if self._predict_reward is not None:
                rew_il = self._predict_reward(o, ac)
                rew = (
                    1 - self._config.gail_env_reward
                ) * rew_il + self._config.gail_env_reward * rew

            if self._config.absorbing_state:
                target_q_value = (
                    rew + self._config.rl_discount_factor * q_next_value
                )
            else:
                target_q_value = (
                    rew + mask * self._config.rl_discount_factor * q_next_value
                )

        # the q loss
        if self._config.critic_ensemble == 1:
            real_q_value = self._critic(o, ac)
            critic_loss = F.mse_loss(target_q_value, real_q_value)
        else:
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

        if self._config.critic_ensemble == 1:
            info["min_real1_q"] = real_q_value.min().cpu().item()
            info["real1_q"] = real_q_value.mean().cpu().item()
            info["critic1_loss"] = critic_loss.cpu().item()
        else:
            info["min_real1_q"] = real_q_value1.min().cpu().item()
            info["min_real2_q"] = real_q_value2.min().cpu().item()
            info["real1_q"] = real_q_value1.mean().cpu().item()
            info["real2_q"] = real_q_value2.mean().cpu().item()
            info["critic1_loss"] = critic1_loss.cpu().item()
            info["critic2_loss"] = critic2_loss.cpu().item()

        return info

    def _update_network(self, transitions):
        info = Info()

        # pre-process the observation
        o, o_next = transitions["ob"], transitions["ob_next"]
        o = self.normalize(o)
        o_next = self.normalize(o_next)
        bs = len(transitions["done"])

        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        o_next = _to_tensor(o_next)
        ac = _to_tensor(transitions["ac"])
        mask = _to_tensor(transitions["done_mask"]).reshape(bs, 1)
        rew = _to_tensor(transitions["rew"]).reshape(bs, 1)

        self._update_iter += 1

        critic_train_info = self._update_critic(o, ac, rew, o_next, mask)
        info.add(critic_train_info)

        if (
            self._update_iter % self._config.actor_update_freq == 0
            and self._update_iter > self._config.actor_update_delay
        ):
            actor_train_info = self._update_actor(o, mask)
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

        if (
            self._update_iter % self._config.actor_target_update_freq == 0
            and self._update_iter > self._config.actor_update_delay
        ):
            self._soft_update_target_network(
                self._actor_target.fc,
                self._actor.fc,
                self._config.actor_soft_update_weight,
            )
            for k, fc in self._actor.fcs.items():
                self._soft_update_target_network(
                    self._actor_target.fcs[k],
                    fc,
                    self._config.actor_soft_update_weight,
                )
            self._soft_update_target_network(
                self._actor_target.encoder,
                self._actor.encoder,
                self._config.encoder_soft_update_weight,
            )

        return info.get_dict(only_scalar=True)
