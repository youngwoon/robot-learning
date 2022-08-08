# Dreamer code reference:
# https://github.com/danijar/dreamer/blob/master/dreamer.py

import numpy as np
import torch
import gym.spaces

from .base_agent import BaseAgent
from .dataset import ReplayBufferEpisode, SeqSampler
from .dreamer_rollout import DreamerRolloutRunner
from ..networks.dreamer import DreamerModel, DenseDecoder1, ActionDecoder
from ..utils import Logger, Once, Info, StopWatch
from ..utils.pytorch import optimizer_cuda, count_parameters
from ..utils.pytorch import to_tensor, RequiresGrad, AdamAMP
from ..utils.dreamer import static_scan, lambda_return


class DreamerAgent(BaseAgent):
    def __init__(self, cfg, ob_space, ac_space):
        super().__init__(cfg, ob_space)
        self._ob_space = ob_space
        self._ac_space = ac_space
        self._ac_dim = ac_dim = gym.spaces.flatdim(ac_space)
        self._use_amp = cfg.precision == 16
        self._dtype = torch.float16 if self._use_amp else torch.float32
        state_dim = cfg.deter_dim + cfg.stoch_dim

        # Build up networks
        self.model = DreamerModel(cfg, ob_space, ac_dim, self._dtype)
        self.actor = ActionDecoder(
            state_dim, ac_dim, [cfg.num_units] * 4, cfg.dense_act
        )
        self.critic = DenseDecoder1(state_dim, 1, [cfg.num_units] * 3, cfg.dense_act)
        self.to(self._device)

        # Optimizers
        adam_amp = lambda model, lr: AdamAMP(
            model, lr, cfg.weight_decay, cfg.grad_clip, self._device, self._use_amp
        )
        self.model_optim = adam_amp(self.model, cfg.model_lr)
        self.actor_optim = adam_amp(self.actor, cfg.actor_lr)
        self.critic_optim = adam_amp(self.critic, cfg.critic_lr)

        # Per-episode replay buffer
        sampler = SeqSampler(cfg.batch_length)
        buffer_keys = ["ob", "ac", "rew", "done"]
        self._buffer = ReplayBufferEpisode(
            buffer_keys, cfg.buffer_size, sampler.sample_func, cfg.precision
        )

        self._log_creation()

        # Freeze modules. Only updated modules will be unfrozen.
        self.requires_grad_(False)

    @property
    def ac_space(self):
        return self._ac_space

    def _log_creation(self):
        Logger.info("Creating a Dreamer agent")
        Logger.info(f"The actor has {count_parameters(self.actor)} parameters")
        Logger.info(f"The critic has {count_parameters(self.critic)} parameters")
        Logger.info(f"The model has {count_parameters(self.model)} parameters")

    @torch.no_grad()
    def act(self, ob, state, is_train=True):
        """Returns action and the actor's activation given an observation `ob`."""
        ob = ob.copy()
        for k, v in ob.items():
            ob[k] = np.expand_dims(v, axis=0).copy()

        self.model.eval()
        self.actor.eval()
        with torch.autocast(self._cfg.device, enabled=self._use_amp):
            ob = to_tensor(ob, self._device, self._dtype)
            ac, state_next = self._policy(ob, state, is_train)
            ac = ac.cpu().numpy().squeeze(0)
            ac = gym.spaces.unflatten(self._ac_space, ac)
        self.model.train()
        self.actor.train()

        return ac, state_next

    def get_runner(self, cfg, env, env_eval):
        """Returns rollout runner."""
        return DreamerRolloutRunner(cfg, env, env_eval, self)

    def is_off_policy(self):
        return True

    def store_episode(self, rollouts):
        self._buffer.store_episode(rollouts, include_last_ob=False)

    def state_dict(self):
        return {
            "model": self.model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "model_optim": self.model_optim.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "ob_norm_state_dict": self._ob_norm.state_dict(),
        }

    def load_state_dict(self, ckpt):
        self.model.load_state_dict(ckpt["model"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.to(self._device)

        self.model_optim.load_state_dict(ckpt["model_optim"])
        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.critic_optim.load_state_dict(ckpt["critic_optim"])
        optimizer_cuda(self.model_optim, self._device)
        optimizer_cuda(self.actor_optim, self._device)
        optimizer_cuda(self.critic_optim, self._device)

    def update(self):
        train_info = Info()
        log_once = Once()
        sw_data = StopWatch()
        sw_train = StopWatch()
        for _ in range(self._cfg.train_iter):
            sw_data.start()
            batch = self._buffer.sample(self._cfg.batch_size)
            sw_data.stop()

            sw_train.start()
            _train_info = self._update_network(batch, log_image=log_once())
            train_info.add(_train_info)
            sw_train.stop()
        Logger.info(f"Data: {sw_data.average():.3f}  Train: {sw_train.average():.3f}")
        # return train_info.get_dict()

        info = train_info.get_dict()
        Logger.info(
            f"model_grad: {info['model_grad_norm']:.1f} / actor_grad: {info['actor_grad_norm']:.1f} / critic_grad: {info['critic_grad_norm']:.1f} / model_loss: {info['model_loss']:.1f} / actor_loss: {info['actor_loss']:.1f} / critic_loss: {info['critic_loss']:.1f} / prior_ent: {info['prior_entropy']:.1f} / post_ent: {info['posterior_entropy']:.1f} / reward_loss: {info['reward_loss']:.1f} / div: {info['kl_loss']:.1f} / actor_ent: {info['actor_entropy']:.1f}"
        )
        return info

    def _update_network(self, batch, log_image=False):
        info = Info()
        cfg = self._cfg

        o = to_tensor(batch["ob"], self._device, self._dtype)
        ac = to_tensor(batch["ac"], self._device, self._dtype)
        rew = to_tensor(batch["rew"], self._device, self._dtype)
        o = self.preprocess(o)

        # Compute model loss
        with RequiresGrad(self.model):
            with torch.autocast(cfg.device, enabled=self._use_amp):
                embed = self.model.encoder(o)
                post, prior = self.model.observe(embed, ac)
                feat = self.model.get_feat(post)

                ob_pred = self.model.decoder(feat)
                recon_losses = {k: -ob_pred[k].log_prob(v).mean() for k, v in o.items()}
                recon_loss = sum(recon_losses.values())

                reward_pred = self.model.reward(feat)
                reward_loss = -reward_pred.log_prob(rew.unsqueeze(-1)).mean()

                prior_dist = self.model.get_dist(prior)
                post_dist = self.model.get_dist(post)

                # Clipping KL divergence after taking mean (from official code)
                div = torch.distributions.kl.kl_divergence(post_dist, prior_dist).mean()
                div_clipped = torch.clamp(div, min=cfg.free_nats)
                model_loss = cfg.kl_scale * div_clipped + recon_loss + reward_loss
            model_grad_norm = self.model_optim.step(model_loss)

        # Compute actor loss with imaginary rollout
        with RequiresGrad(self.actor):
            with torch.autocast(cfg.device, enabled=self._use_amp):
                post = {k: v.detach() for k, v in post.items()}
                imagine_feat = self._imagine_ahead(post)
                imagine_reward = (
                    self.model.reward(imagine_feat).mode().squeeze(-1).float()
                )
                imagine_value = self.critic(imagine_feat).mode().squeeze(-1).float()
                pcont = cfg.rl_discount * torch.ones_like(imagine_reward)
                imagine_return = lambda_return(
                    imagine_reward[:-1],
                    imagine_value[:-1],
                    pcont[:-1],
                    bootstrap=imagine_value[-1],
                    lambda_=cfg.gae_lambda,
                )
                with torch.no_grad():
                    discount = torch.cumprod(
                        torch.cat([torch.ones_like(pcont[:1]), pcont[:-2]], 0), 0
                    )
                actor_loss = -(discount * imagine_return).mean()
            actor_grad_norm = self.actor_optim.step(actor_loss)

        # Compute critic loss
        with RequiresGrad(self.critic):
            with torch.autocast(cfg.device, enabled=self._use_amp):
                value_pred = self.critic(imagine_feat.detach()[:-1])
                target = imagine_return.detach().unsqueeze(-1)
                critic_loss = -(discount * value_pred.log_prob(target)).mean()
            critic_grad_norm = self.critic_optim.step(critic_loss)

        # Log scalar
        for k, v in recon_losses.items():
            info[f"recon_loss_{k}"] = v.item()
        info["reward_loss"] = reward_loss.item()
        info["prior_entropy"] = prior_dist.entropy().mean().item()
        info["posterior_entropy"] = post_dist.entropy().mean().item()
        info["kl_loss"] = div_clipped.item()
        info["model_loss"] = model_loss.item()
        info["actor_loss"] = actor_loss.item()
        info["critic_loss"] = critic_loss.item()
        info["value_target"] = imagine_return.mean().item()
        info["value_predicted"] = value_pred.mode().mean().item()
        info["model_grad_norm"] = model_grad_norm.item()
        info["actor_grad_norm"] = actor_grad_norm.item()
        info["critic_grad_norm"] = critic_grad_norm.item()

        if log_image:
            with torch.no_grad(), torch.autocast(cfg.device, enabled=self._use_amp):
                info["actor_entropy"] = self.actor(feat).entropy().mean().item()

                # 5 timesteps for each of 4 samples
                init, _ = self.model.observe(embed[:4, :5], ac[:4, :5])
                init = {k: v[:, -1] for k, v in init.items()}
                prior = self.model.imagine(ac[:4, 5:], init)
                openloop = self.model.decoder(self.model.get_feat(prior)).mode()
                for k, v in o.items():
                    if len(v.shape) != 5:
                        continue
                    truth = o[k][:4] + 0.5
                    recon = ob_pred[k].mode()[:4]
                    model = torch.cat([recon[:, :5] + 0.5, openloop[k] + 0.5], 1)
                    error = (model - truth + 1) / 2
                    openloop = torch.cat([truth, model, error], 2)
                    img = openloop.detach().cpu().numpy() * 255
                    info[f"recon_{k}"] = img.transpose(0, 1, 4, 2, 3).astype(np.uint8)

        if cfg.maze_visualize:
            with torch.no_grad():
                init, _ = self.model.observe(embed[:1, :1], ac[:1, :1])
                init = {k: v[:, -1] for k, v in init.items()}
                prior = self.model.imagine(ac[:1, 1:], init)
                openloop = self.model.decoder(self.model.get_feat(prior)).mode()
                for k, v in o.items():
                    recon = ob_pred[k].mode()[:1]
                    model = torch.cat([o["ob"][:1, :1], openloop[k]], 1)

            info["skill_rollout"] = self._visualize(
                o["ob"][:1].detach().cpu().numpy(),
                model.detach().cpu().numpy(),
            )

        return info.get_dict()

    def _imagine_ahead(self, post):
        """Computes imagination rollouts.
        Args:
            post: BxTx(`stoch_dim` + `deter_dim`) stochastic states.
        """
        flatten = lambda x: x.reshape([-1] + list(x.shape[2:]))
        start = {k: flatten(v) for k, v in post.items()}
        policy = lambda state: self.actor.act(self.model.get_feat(state).detach())
        imagine_states = static_scan(
            lambda prev, _: self.model.imagine_step(prev, policy(prev)),
            [torch.arange(self._cfg.horizon)],
            start,
        )
        imagine_feat = self.model.get_feat(imagine_states)
        return imagine_feat

    def _policy(self, ob, state, is_train):
        """Computes actions given `ob` and `state`.

        Args:
            ob: list of B observations (tensors)
            state: (previous_latent_state, previous_action)
        """
        latent, action = state or self.initial_state(ob)
        embed = self.model.encoder(self.preprocess(ob))
        latent = self.model.obs_step(latent, action, embed)
        feat = self.model.get_feat(latent)
        action = self.actor.act(feat, deterministic=not is_train)
        if is_train:
            action = action + torch.randn_like(action) * self._cfg.expl_noise
            action = torch.clamp(action, -1, 1)
        state = (latent, action)
        return action, state

    def initial_state(self, ob):
        batch_size = len(list(ob.values())[0])
        latent = self.model.initial(batch_size)
        action = torch.zeros(
            [batch_size, self._ac_dim], dtype=self._dtype, device=self._device
        )
        return latent, action

    def preprocess(self, ob):
        ob = ob.copy()
        for k, v in ob.items():
            if len(v.shape) >= 4:
                ob[k] = ob[k] / 255.0 - 0.5
        return ob

    def _visualize(self, ob_gt, ob_pred):
        """Visualize the prediction and ground truth states."""
        from matplotlib import pyplot as plt
        import wandb
        import imageio

        n_vis = 1
        self._overlay = imageio.imread("envs/assets/maze_40.png")

        extent = (0, 40, 0, 40)
        ob_gt = ob_gt[:, :, :2]
        ob_gt = np.concatenate([ob_gt[i] for i in range(n_vis)], 0)
        ob_pred = ob_pred[:, :, :2]
        ob_pred = np.clip(ob_pred, 0, 40)
        ob_pred = np.concatenate([ob_pred[i] for i in range(n_vis)], 0)

        fig, axs = plt.subplots(1, 3, clear=True)
        for ax in axs.reshape(-1):
            ax.imshow(self._overlay, alpha=0.3, extent=extent)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlim(0, 40)
            ax.set_ylim(0, 40)
            ax.axis("off")

        def render(ax, data, cmap, title, legend=True):
            s = ax.scatter(
                40 - data[:, 1],
                data[:, 0],
                s=3,
                c=np.arange(len(data)),
                cmap=cmap,
            )
            ax.set_title(title)
            if legend:
                cbar = fig.colorbar(s, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=6)

        render(axs[0], ob_gt, "summer", "Ground truth")
        render(axs[1], ob_pred, "copper", "Predicted Rollout")

        render(axs[2], ob_gt, "summer", "All", False)
        render(axs[2], ob_pred, "copper", "All", False)

        img = wandb.Image(fig)
        plt.close(fig)
        return img
