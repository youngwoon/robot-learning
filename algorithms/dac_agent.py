from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
from torch.optim.lr_scheduler import StepLR
import gym.spaces

from .base_agent import BaseAgent
from .dataset import ReplayBuffer, RandomSampler
from .expert_dataset import ExpertDataset
from ..networks import Actor, Critic
from ..networks.discriminator import Discriminator
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


class DACAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, env_ob_space):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        self._target_entropy = -gym.spaces.flatdim(ac_space)
        self._log_alpha = torch.tensor(
            np.log(config.alpha_init_temperature),
            requires_grad=True,
            device=config.device,
        )

        # build up networks
        self._actor = Actor(config, ob_space, ac_space, config.tanh_policy)
        self._critic = Critic(config, ob_space, ac_space)
        self._discriminator = Discriminator(
            config, ob_space, ac_space if not config.gail_no_action else None
        )
        self._discriminator_loss = nn.BCEWithLogitsLoss()

        # build up target networks
        self._critic_target = Critic(config, ob_space, ac_space)
        self._network_cuda(config.device)
        self._copy_target_network(self._critic_target, self._critic)
        self._actor.encoder.copy_conv_weights_from(self._critic.encoder)

        # build optimizers
        self._alpha_optim = optim.Adam(
            [self._log_alpha], lr=config.alpha_lr, betas=(0.5, 0.999)
        )
        self._actor_optim = optim.Adam(
            self._actor.parameters(), lr=config.actor_lr, betas=(0.9, 0.999)
        )
        self._critic_optim = optim.Adam(
            self._critic.parameters(), lr=config.critic_lr, betas=(0.9, 0.999)
        )
        self._discriminator_optim = optim.Adam(
            self._discriminator.parameters(), lr=config.discriminator_lr
        )

        self._discriminator_lr_scheduler = StepLR(
            self._discriminator_optim,
            step_size=self._config.max_global_step // self._config.rollout_length // 5,
            gamma=0.5,
        )

        # expert dataset
        self._dataset = ExpertDataset(config.demo_path, config.demo_subsample_interval)
        self._data_loader = torch.utils.data.DataLoader(
            self._dataset, batch_size=self._config.batch_size, shuffle=True
        )
        self._data_iter = iter(self._data_loader)

        # per-episode replay buffer
        sampler = RandomSampler(image_crop_size=config.encoder_image_size)
        buffer_keys = ["ob", "ac", "done", "rew"]
        self._buffer = ReplayBuffer(
            buffer_keys, config.buffer_size, sampler.sample_func
        )

        self._update_iter = 0

        self._log_creation()

    def _predict_reward(self, ob, ac):
        if self._config.gail_no_action:
            ac = None
        with torch.no_grad():
            ret = self._discriminator(ob, ac)
            eps = 1e-20
            s = torch.sigmoid(ret)
            if self._config.gail_vanilla_reward:
                reward = -(1 - s + eps).log()
            else:
                reward = (s + eps).log() - (1 - s + eps).log()
        return reward

    def predict_reward(self, ob, ac=None):
        ob = self.normalize(ob)
        ob = to_tensor(ob, self._config.device)
        if self._config.gail_no_action:
            ac = None
        if ac is not None:
            ac = to_tensor(ac, self._config.device)

        reward = self._predict_reward(ob, ac)
        return reward.cpu().item()

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a DAC agent")
            logger.info("The actor has %d parameters", count_parameters(self._actor))
            logger.info("The critic has %d parameters", count_parameters(self._critic))
            logger.info(
                "The discriminator has %d parameters",
                count_parameters(self._discriminator),
            )

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
            "discriminator_state_dict": self._discriminator.state_dict(),
            "alpha_optim_state_dict": self._alpha_optim.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "critic_optim_state_dict": self._critic_optim.state_dict(),
            "discriminator_optim_state_dict": self._discriminator_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        if "critic_state_dict" not in ckpt:
            # BC initialization
            logger.warn("Load only actor from BC initialization")
            self._actor.load_state_dict(ckpt["actor_state_dict"], strict=False)
            self._network_cuda(self._config.device)
            self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
            return

        self._log_alpha.data = torch.tensor(
            ckpt["log_alpha"], requires_grad=True, device=self._config.device
        )
        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._critic.load_state_dict(ckpt["critic_state_dict"])
        self._copy_target_network(self._critic_target, self._critic)
        self._discriminator.load_state_dict(ckpt["discriminator_state_dict"])
        self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
        self._network_cuda(self._config.device)

        self._alpha_optim.load_state_dict(ckpt["alpha_optim_state_dict"])
        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self._critic_optim.load_state_dict(ckpt["critic_optim_state_dict"])
        self._discriminator_optim.load_state_dict(
            ckpt["discriminator_optim_state_dict"]
        )
        optimizer_cuda(self._alpha_optim, self._config.device)
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic_optim, self._config.device)
        optimizer_cuda(self._discriminator_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._critic.to(device)
        self._critic_target.to(device)
        self._discriminator.to(device)

    def sync_networks(self):
        sync_networks(self._actor)
        sync_networks(self._critic)
        sync_networks(self._discriminator)

    def train(self):
        train_info = Info()

        self._discriminator_lr_scheduler.step()

        self._num_updates = 1
        for _ in range(self._num_updates):
            transitions = self._buffer.sample(self._config.batch_size)
            _train_info = self._update_policy(transitions)
            train_info.add(_train_info)

        if self._update_iter % self._config.discriminator_update_freq == 0:
            for _ in range(self._num_updates):
                policy_data = self._buffer.sample(self._config.batch_size)
                try:
                    expert_data = next(self._data_iter)
                except StopIteration:
                    self._data_iter = iter(self._data_loader)
                    expert_data = next(self._data_iter)
                _train_info = self._update_discriminator(policy_data, expert_data)
                train_info.add(_train_info)

        # train_info.add(
        #     {
        #         "actor_grad_norm": compute_gradient_norm(self._actor),
        #         "actor_weight_norm": compute_weight_norm(self._actor),
        #         "critic_grad_norm": compute_gradient_norm(self._critic),
        #         "critic_weight_norm": compute_weight_norm(self._critic),
        #     }
        # )
        return train_info.get_dict(only_scalar=True)

    def _update_discriminator(self, policy_data, expert_data):
        info = Info()

        _to_tensor = lambda x: to_tensor(x, self._config.device)
        # pre-process observations
        p_o = policy_data["ob"]
        p_o = self.normalize(p_o)

        p_bs = len(policy_data["ac"])
        p_o = _to_tensor(p_o)
        if self._config.gail_no_action:
            p_ac = None
        else:
            p_ac = _to_tensor(policy_data["ac"])

        e_o = expert_data["ob"]
        e_o = self.normalize(e_o)

        e_bs = len(expert_data["ac"])
        e_o = _to_tensor(e_o)
        if self._config.gail_no_action:
            e_ac = None
        else:
            e_ac = _to_tensor(expert_data["ac"])

        p_logit = self._discriminator(p_o, p_ac)
        e_logit = self._discriminator(e_o, e_ac)

        p_output = torch.sigmoid(p_logit)
        e_output = torch.sigmoid(e_logit)

        p_loss = self._discriminator_loss(
            p_logit, torch.zeros_like(p_logit).to(self._config.device)
        )
        e_loss = self._discriminator_loss(
            e_logit, torch.ones_like(e_logit).to(self._config.device)
        )

        logits = torch.cat([p_logit, e_logit], dim=0)
        entropy = torch.distributions.Bernoulli(logits).entropy().mean()
        entropy_loss = -self._config.gail_entropy_loss_coeff * entropy

        gail_loss = p_loss + e_loss + entropy_loss

        # update the discriminator
        self._discriminator.zero_grad()
        gail_loss.backward()
        # torch.nn.utils.clip_grad_norm_(self._actor.parameters(), self._config.max_grad_norm)
        sync_grads(self._discriminator)
        self._discriminator_optim.step()

        info["gail_policy_output"] = p_output.mean().detach().cpu().item()
        info["gail_expert_output"] = e_output.mean().detach().cpu().item()
        info["gail_entropy"] = entropy.detach().cpu().item()
        info["gail_policy_loss"] = p_loss.detach().cpu().item()
        info["gail_expert_loss"] = e_loss.detach().cpu().item()
        info["gail_entropy_loss"] = entropy_loss.detach().cpu().item()

        return mpi_average(info.get_dict(only_scalar=True))

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

    def _update_policy(self, transitions):
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
        rew_env = _to_tensor(transitions["rew"]).reshape(bs, 1)
        rew_il = self._predict_reward(o, ac)
        rew = (
            1 - self._config.gail_env_reward
        ) * rew_il + self._config.gail_env_reward * rew_env

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
