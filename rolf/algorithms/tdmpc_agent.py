import numpy as np
import torch
import torch.nn as nn
import gym.spaces

from . import BaseAgent
from .dataset import ReplayBufferEpisode, SeqSampler
from ..utils import Logger, Info, Once, StopWatch, LinearDecay
from ..utils.pytorch import optimizer_cuda, count_parameters
from ..utils.pytorch import copy_network, soft_copy_network
from ..utils.pytorch import to_tensor, RandomShiftsAug, AdamAMP

from ..networks.tdmpc_model import TDMPCModel, ActionDecoder


class TDMPCAgent(BaseAgent):
    """TD-MPC algorithm.

    Comments on design choices from Nicklas Hansen.
    - PER is only necessary for Finger Turn Hard (hard exploration, sparse reward).
    - Image shifting augmentation is essential.
    - Larger networks did not improve performances.
    """

    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._use_amp = cfg.precision == 16
        self._dtype = torch.float16 if self._use_amp else torch.float32
        self._std = LinearDecay(cfg.max_std, cfg.min_std, cfg.std_step)
        self._horizon = LinearDecay(1, cfg.horizon, cfg.horizon_step)
        self._update_iter = 0

        self._build_networks()
        self._build_optims()
        self._build_buffers()
        self._log_creation()

    def _build_networks(self):
        cfg = self._cfg
        self._ac_dim = ac_dim = gym.spaces.flatdim(self._ac_space)
        self.model = TDMPCModel(cfg, self._ob_space, ac_dim, self._dtype)
        self.model_target = TDMPCModel(cfg, self._ob_space, ac_dim, self._dtype)
        copy_network(self.model_target, self.model)
        self.actor = ActionDecoder(
            cfg.state_dim, ac_dim, [cfg.num_units] * 2, cfg.dense_act
        )
        self._aug = RandomShiftsAug()
        self.to(self._device)

    def _build_optims(self):
        cfg = self._cfg
        adam_amp = lambda model, lr: AdamAMP(
            model, lr, cfg.weight_decay, cfg.grad_clip, self._device, self._use_amp
        )
        self._model_optim = adam_amp(self.model, cfg.model_lr)
        self._actor_optim = adam_amp(self.actor, cfg.actor_lr)

    def _build_buffers(self):
        cfg = self._cfg
        # Per-episode replay buffer
        sampler = SeqSampler(cfg.horizon)
        buffer_keys = ["ob", "ac", "rew", "done"]
        self._buffer = ReplayBufferEpisode(
            buffer_keys, cfg.buffer_size, sampler.sample_func_one_more_ob, cfg.precision
        )

    def _log_creation(self):
        Logger.info("Creating a TD-MPC agent")
        Logger.info(f"The actor has {count_parameters(self.actor)} parameters")
        Logger.info(f"The model has {count_parameters(self.model)} parameters")

    def is_off_policy(self):
        return True

    def store_episode(self, rollout):
        self._buffer.store_episode(rollout)

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "model_target": self.model_target.state_dict(),
            "actor": self.actor.state_dict(),
            "ob_norm": self._ob_norm.state_dict(),
            "model_optim": self._model_optim.state_dict(),
            "actor_optim": self._actor_optim.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.model.load_state_dict(ckpt["model"])
        self.model_target.load_state_dict(ckpt["model_target"])
        self.actor.load_state_dict(ckpt["actor"])
        self.to(self._device)

        self._model_optim.load_state_dict(ckpt["model_optim"])
        self._actor_optim.load_state_dict(ckpt["actor_optim"])
        optimizer_cuda(self._model_optim, self._device)
        optimizer_cuda(self._actor_optim, self._device)

    @property
    def ac_space(self):
        return self._ac_space

    @torch.no_grad()
    def estimate_value(self, state, ac, horizon):
        value, discount = 0, 1
        for t in range(horizon):
            state, reward = self.model.imagine_step(state, ac[t])
            value += discount * reward
            discount *= self._cfg.rl_discount
        value += discount * torch.min(
            *self.model.critic(state, self.actor(state, self._cfg.min_std))
        )
        return value

    @torch.no_grad()
    def plan(self, ob, prev_mean=None, is_train=True):
        """Plan given an observation `ob`."""
        cfg = self._cfg
        horizon = int(self._horizon(self._step))

        state = self.model.encoder(ob)

        # Sample policy trajectories
        z = state.repeat(cfg.num_policy_traj, 1)
        policy_ac = []
        for t in range(horizon):
            policy_ac.append(self.actor(z, cfg.min_std))
            z, _ = self.model.imagine_step(z, policy_ac[t])
        policy_ac = torch.stack(policy_ac, dim=0)

        # CEM optimization
        z = state.repeat(cfg.num_policy_traj + cfg.num_sample_traj, 1)
        mean = torch.zeros(horizon, self._ac_dim, device=self._device)
        std = 2.0 * torch.ones(horizon, self._ac_dim, device=self._device)
        if prev_mean is not None and horizon > 1 and prev_mean.shape[0] == horizon:
            mean[:-1] = prev_mean[1:]

        for _ in range(cfg.cem_iter):
            sample_ac = mean.unsqueeze(1) + std.unsqueeze(1) * torch.randn(
                horizon, cfg.num_sample_traj, self._ac_dim, device=self._device
            )
            sample_ac = torch.clamp(sample_ac, -1, 1)

            ac = torch.cat([sample_ac, policy_ac], dim=1)

            imagine_return = self.estimate_value(z, ac, horizon).squeeze(-1)
            _, idxs = imagine_return.sort(dim=0)
            idxs = idxs[-cfg.num_elites :]
            elite_value = imagine_return[idxs]
            elite_action = ac[:, idxs]

            # Weighted aggregation of elite plans
            score = torch.exp(cfg.cem_temperature * (elite_value - elite_value.max()))
            score = (score / score.sum()).view(1, -1, 1)
            new_mean = (score * elite_action).sum(dim=1)
            new_std = torch.sqrt(
                torch.sum(score * (elite_action - new_mean.unsqueeze(1)) ** 2, dim=1)
            )

            mean = cfg.cem_momentum * mean + (1 - cfg.cem_momentum) * new_mean
            std = torch.clamp(new_std, self._std(self._step), 2)

        # Sample action for MPC
        score = score.squeeze().cpu().numpy()
        ac = elite_action[0, np.random.choice(np.arange(cfg.num_elites), p=score)]
        if is_train:
            ac += std[0] * torch.randn_like(std[0])
        return torch.clamp(ac, -1, 1), mean

    @torch.no_grad()
    def act(self, ob, mean=None, is_train=True):
        """Returns action and the actor's activation given an observation `ob`."""
        ob = ob.copy()
        for k, v in ob.items():
            ob[k] = np.expand_dims(v, axis=0).copy()

        self.model.eval()
        self.actor.eval()
        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            ob = to_tensor(ob, self._device, self._dtype)
            ob = self.preprocess(ob)
            ac, mean = self.plan(ob, mean, is_train)
            ac = ac.cpu().numpy()
            ac = gym.spaces.unflatten(self._ac_space, ac)
        self.model.train()
        self.actor.train()
        return ac, mean

    def preprocess(self, ob, aug=None):
        ob = ob.copy()
        for k, v in ob.items():
            if len(v.shape) >= 4:
                ob[k] = ob[k] / 255.0 - 0.5
                # ob[k] = ob[k] / 255.0  # CHECK IT
                if aug:
                    ob[k] = aug(ob[k])
        return ob

    def update(self):
        train_info = Info()
        sw_data, sw_train = StopWatch(), StopWatch()
        train_steps = self._cfg.train_steps
        if self.warm_up_training():
            train_steps += self._cfg.warm_up_step
        for _ in range(train_steps):
            sw_data.start()
            batch = self._buffer.sample(self._cfg.batch_size)
            # ob: {k: BxTx`ob_dim[k]`}, ac: BxTx`ac_dim`, rew: BxTx1
            # batch["ob"] = to_tensor(batch["ob"], self._device, self._dtype)
            # batch["ac"] = to_tensor(batch["ac"], self._device, self._dtype)
            # batch["rew"] = to_tensor(batch["rew"], self._device, self._dtype)
            # batch["done"] = to_tensor(batch["done"], self._device, self._dtype)  # BxTx1
            sw_data.stop()

            sw_train.start()
            _train_info = self._update_network(batch)
            train_info.add(_train_info)
            sw_train.stop()
        Logger.info(f"Data: {sw_data.average():.3f}  Train: {sw_train.average():.3f}")
        return train_info.get_dict()

    def _update_network(self, batch):
        cfg = self._cfg
        info = Info()
        mse = nn.MSELoss(reduction="none")

        # o = to_tensor(batch["ob"], self._device, self._dtype)  # {k: BxTx`ob_dim[k]`}
        # ac = to_tensor(batch["ac"], self._device, self._dtype)  # BxTx`ac_dim`
        # rew = to_tensor(batch["rew"], self._device, self._dtype)  # BxTx1
        # done = to_tensor(batch["done"], self._device, self._dtype)  # BxTx1
        o = batch["ob"]
        ac = batch["ac"]
        rew = batch["rew"]
        # done = batch["done"]
        o = self.preprocess(o, aug=self._aug)
        ac = torch.cat([v for v in ac.values()], -1)

        # Flip dimensions, BxT -> TxB
        def flip(x, l=None):
            if isinstance(x, dict):
                return [{k: v[:, t] for k, v in x.items()} for t in range(l)]
            else:
                return x.transpose(0, 1)

        o = flip(o, cfg.horizon + 1)
        ac = flip(ac)
        rew = flip(rew)
        # done = flip(done)

        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            z = z_next_pred = self.model.encoder(o[0])
            zs = [z.detach()]

            consistency_loss = 0
            reward_loss = 0
            value_loss = 0
            for t in range(cfg.horizon):
                z = z_next_pred
                q_pred = self.model.critic(z, ac[t])
                z_next_pred, reward_pred = self.model.imagine_step(z, ac[t])
                with torch.no_grad():
                    # `z` for contrastive learning
                    z_next = self.model_target.encoder(o[t + 1])

                    # `z` for `q_target`
                    z_next_q = self.model.encoder(o[t + 1])
                    ac_next = self.actor(z_next_q, cfg.min_std)
                    q_next = torch.min(*self.model_target.critic(z_next_q, ac_next))
                    # q_target = rew[t] + (1 - done[t]) * cfg.rl_discount * q_next
                    q_target = rew[t] + cfg.rl_discount * q_next
                zs.append(z_next_pred.detach())

                rho = cfg.rho**t
                consistency_loss += rho * mse(z_next_pred, z_next).mean(dim=1)
                reward_loss += rho * mse(reward_pred, rew[t])
                value_loss += rho * (
                    mse(q_pred[0], q_target) + mse(q_pred[1], q_target)
                )
            model_loss = (
                cfg.consistency_coef * consistency_loss.clamp(max=1e4)
                + cfg.reward_coef * reward_loss.clamp(max=1e4)
                + cfg.value_coef * value_loss.clamp(max=1e4)
            ).mean()
            model_loss.register_hook(lambda grad: grad * (1 / cfg.horizon))  # CHECK
        model_grad_norm = self._model_optim.step(model_loss)

        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            # self.model.critic.requires_grad_(False)  # CHECK
            actor_loss = 0
            for t, z in enumerate(zs):
                a = self.actor(z, cfg.min_std)
                rho = cfg.rho**t
                actor_loss += -rho * torch.min(*self.model.critic(z, a)).mean()
        actor_grad_norm = self._actor_optim.step(actor_loss)
        # self.model.critic.requires_grad_(True)  # CHECK

        self._update_iter += 1
        if self._update_iter % cfg.target_update_freq == 0:
            soft_copy_network(self.model_target, self.model, cfg.target_update_tau)

        info["min_q_target"] = q_target.min().item()
        info["q_target"] = q_target.mean().item()
        info["min_q_pred1"] = q_pred[0].min().item()
        info["min_q_pred2"] = q_pred[1].min().item()
        info["q_pred1"] = q_pred[0].mean().item()
        info["q_pred2"] = q_pred[1].mean().item()
        info["model_grad_norm"] = model_grad_norm.item()
        info["actor_grad_norm"] = actor_grad_norm.item()
        info["actor_loss"] = actor_loss.mean().item()
        info["model_loss"] = model_loss.mean().item()
        info["consistency_loss"] = consistency_loss.mean().item()
        info["reward_loss"] = reward_loss.mean().item()
        info["value_loss"] = value_loss.mean().item()

        return info.get_dict()
