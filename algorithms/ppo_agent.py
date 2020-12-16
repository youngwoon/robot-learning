import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from ..networks import Actor, Critic
from ..utils.info_dict import Info
from ..utils.logger import logger
from ..utils.mpi import mpi_average
from ..utils.pytorch import (
    compute_gradient_norm,
    compute_weight_norm,
    count_parameters,
    obs2tensor,
    optimizer_cuda,
    sync_grads,
    sync_networks,
    to_tensor,
    center_crop_images
)
from .base_agent import BaseAgent
from .dataset import RandomSampler, ReplayBuffer


class PPOAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, env_ob_space):
        super().__init__(config, ob_space)

        self._ac_space = ac_space

        # build up networks
        self._actor = Actor(config, ob_space, ac_space, config.tanh_policy)
        self._old_actor = Actor(config, ob_space, ac_space, config.tanh_policy)
        self._critic = Critic(config, ob_space)
        self._network_cuda(config.device)

        self._actor_optim = optim.Adam(self._actor.parameters(), lr=config.actor_lr)
        self._critic_optim = optim.Adam(self._critic.parameters(), lr=config.critic_lr)

        self._actor_lr_scheduler = StepLR(
            self._actor_optim,
            step_size=self._config.max_global_step // self._config.rollout_length // 5,
            gamma=0.5,
        )
        self._critic_lr_scheduler = StepLR(
            self._critic_optim,
            step_size=self._config.max_global_step // self._config.rollout_length // 5,
            gamma=0.5,
        )

        sampler = RandomSampler(image_crop_size=self._config.encoder_image_size)
        self._buffer = ReplayBuffer(
            [
                "ob",
                "ob_next",
                "ac",
                "done",
                "rew",
                "ret",
                "adv",
                "ac_before_activation",
            ],
            config.rollout_length,
            sampler.sample_func,
        )

        self._update_iter = 0

        self._log_creation()

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a PPO agent")
            logger.info("The actor has %d parameters", count_parameters(self._actor))
            logger.info("The critic has %d parameters", count_parameters(self._critic))

    def store_episode(self, rollouts):
        self._compute_gae(rollouts)
        self._buffer.store_episode(rollouts)

    def _compute_gae(self, rollouts):
        T = len(rollouts["done"])
        ob = rollouts["ob"]
        ob = self.normalize(ob)
        ob = obs2tensor(ob, self._config.device)
        for k, v in ob.items():
            if self._config.encoder_type == "cnn" and len(v.shape) == 4:
                ob[k] = center_crop_images(v, self._config.encoder_image_size)

        ob_last = rollouts["ob_next"][-1:]
        ob_last = self.normalize(ob_last)
        ob_last = obs2tensor(ob_last, self._config.device)
        for k, v in ob_last.items():
            if self._config.encoder_type == "cnn" and len(v.shape) == 4:
                ob_last[k] = center_crop_images(v, self._config.encoder_image_size)

        done = rollouts["done"]
        rew = rollouts["rew"]

        vpred = self._critic(ob).detach().cpu().numpy()[:, 0]
        vpred_last = self._critic(ob_last).detach().cpu().numpy()[:, 0]
        vpred = np.append(vpred, vpred_last)
        assert len(vpred) == T + 1

        if hasattr(self, "_predict_reward"):
            ob = rollouts["ob"]
            ob = self.normalize(ob)
            ob = obs2tensor(ob, self._config.device)
            ac = obs2tensor(rollouts["ac"], self._config.device)
            rew_il = self._predict_reward(ob, ac).cpu().numpy().squeeze()
            rew = (1 - self._config.gail_env_reward) * rew_il[
                :T
            ] + self._config.gail_env_reward * np.array(rew)
            assert rew.shape == (T,)

        adv = np.empty((T,), "float32")
        lastgaelam = 0
        for t in reversed(range(T)):
            nonterminal = 1 - done[t]
            delta = (
                rew[t]
                + self._config.rl_discount_factor * vpred[t + 1] * nonterminal
                - vpred[t]
            )
            adv[t] = lastgaelam = (
                delta
                + self._config.rl_discount_factor
                * self._config.gae_lambda
                * nonterminal
                * lastgaelam
            )

        ret = adv + vpred[:-1]

        assert np.isfinite(adv).all()
        assert np.isfinite(ret).all()

        # update rollouts
        if self._config.advantage_norm:
            rollouts["adv"] = ((adv - adv.mean()) / (adv.std() + 1e-5)).tolist()
        else:
            rollouts["adv"] = adv.tolist()

        rollouts["ret"] = ret.tolist()

    def state_dict(self):
        return {
            "actor_state_dict": self._actor.state_dict(),
            "critic_state_dict": self._critic.state_dict(),
            "actor_optim_state_dict": self._actor_optim.state_dict(),
            "critic_optim_state_dict": self._critic_optim.state_dict(),
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

        self._actor.load_state_dict(ckpt["actor_state_dict"])
        self._critic.load_state_dict(ckpt["critic_state_dict"])
        self._ob_norm.load_state_dict(ckpt["ob_norm_state_dict"])
        self._network_cuda(self._config.device)

        self._actor_optim.load_state_dict(ckpt["actor_optim_state_dict"])
        self._critic_optim.load_state_dict(ckpt["critic_optim_state_dict"])
        optimizer_cuda(self._actor_optim, self._config.device)
        optimizer_cuda(self._critic_optim, self._config.device)

    def _network_cuda(self, device):
        self._actor.to(device)
        self._old_actor.to(device)
        self._critic.to(device)

    def sync_networks(self):
        sync_networks(self._actor)
        sync_networks(self._critic)

    def train(self):
        train_info = Info()

        self._copy_target_network(self._old_actor, self._actor)

        num_batches = (
            self._config.ppo_epoch
            * self._config.rollout_length
            // self._config.batch_size
        )
        assert num_batches > 0

        for _ in range(num_batches):
            transitions = self._buffer.sample(self._config.batch_size)
            _train_info = self._update_network(transitions)
            train_info.add(_train_info)

        self._buffer.clear()

        self._actor_lr_scheduler.step()
        self._critic_lr_scheduler.step()

        logger.info(
            "Actor lr %f, Critic lr %f, PPO Clip Frac %f",
            self._actor_lr_scheduler.get_lr()[0],
            self._critic_lr_scheduler.get_lr()[0],
            np.mean(train_info["ppo_clip_frac"])
        )

        # slow!
        # train_info.add(
        #     {
        #         "actor_grad_norm": compute_gradient_norm(self._actor),
        #         "actor_weight_norm": compute_weight_norm(self._actor),
        #         "critic_grad_norm": compute_gradient_norm(self._critic),
        #         "critic_weight_norm": compute_weight_norm(self._critic),
        #     }
        # )
        return mpi_average(train_info.get_dict(only_scalar=True))

    def _update_actor(self, o, a_z, adv):
        info = Info()

        _, _, log_pi, ent = self._actor.act(
            o, activations=a_z, return_log_prob=True
        )
        _, _, old_log_pi, _ = self._old_actor.act(
            o, activations=a_z, return_log_prob=True
        )
        if old_log_pi.min() < -100:
            logger.error("sampling an action with a probability of 1e-100")
            import ipdb
            ipdb.set_trace()

        # the actor loss
        entropy_loss = -self._config.entropy_loss_coeff * ent.mean()
        ratio = torch.exp(log_pi - old_log_pi)
        surr1 = ratio * adv
        surr2 = (
            torch.clamp(ratio, 1.0 - self._config.ppo_clip, 1.0 + self._config.ppo_clip)
            * adv
        )
        actor_loss = -torch.min(surr1, surr2).mean()

        ppo_clip_frac = torch.gt(torch.abs(ratio - 1.0), self._config.ppo_clip).float().mean()

        if (
            not np.isfinite(ratio.cpu().detach()).all()
            or not np.isfinite(adv.cpu().detach()).all()
        ):
            import ipdb

            ipdb.set_trace()
        info["ppo_clip_frac"] = ppo_clip_frac.cpu().item()
        info["entropy_loss"] = entropy_loss.cpu().item()
        info["actor_loss"] = actor_loss.cpu().item()
        actor_loss += entropy_loss

        # update the actor
        self._actor_optim.zero_grad()
        actor_loss.backward()
        if self._config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._actor.parameters(), self._config.max_grad_norm
            )
        sync_grads(self._actor)
        self._actor_optim.step()

        # include info from policy
        info.add(self._actor.info)

        return info

    def _update_critic(self, o, ret):
        info = Info()

        # the q loss
        value_pred = self._critic(o)
        value_loss = self._config.value_loss_coeff * (ret - value_pred).pow(2).mean()

        # update the critic
        self._critic_optim.zero_grad()
        value_loss.backward()
        if self._config.max_grad_norm:
            torch.nn.utils.clip_grad_norm_(
                self._critic.parameters(), self._config.max_grad_norm
            )
        sync_grads(self._critic)
        self._critic_optim.step()

        info["value_target"] = ret.mean().cpu().item()
        info["value_predicted"] = value_pred.mean().cpu().item()
        info["value_loss"] = value_loss.cpu().item()

        return info

    def _update_network(self, transitions):
        info = Info()

        # pre-process observations
        o = transitions["ob"]
        o = self.normalize(o)

        bs = len(transitions["done"])
        _to_tensor = lambda x: to_tensor(x, self._config.device)
        o = _to_tensor(o)
        ac = _to_tensor(transitions["ac"])
        a_z = _to_tensor(transitions["ac_before_activation"])
        ret = _to_tensor(transitions["ret"]).reshape(bs, 1)
        adv = _to_tensor(transitions["adv"]).reshape(bs, 1)

        self._update_iter += 1

        critic_train_info = self._update_critic(o, ret)
        info.add(critic_train_info)

        if self._update_iter % self._config.actor_update_freq == 0:
            actor_train_info = self._update_actor(o, a_z, adv)
            info.add(actor_train_info)

        return info
