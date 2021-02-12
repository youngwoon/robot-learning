from collections import OrderedDict
import os

import numpy as np
import torch

from .base_agent import BaseAgent
from .gail_agent import GAILAgent
from .dataset import RandomSampler, ReplayBuffer
from ..networks.discriminator import Discriminator
from ..utils.info_dict import Info
from ..utils.logger import logger
from ..utils.mpi import mpi_average
from ..utils.pytorch import get_ckpt_path


class PolicySequencingAgent(BaseAgent):
    def __init__(self, config, ob_space, ac_space, env_ob_space):
        super().__init__(config, ob_space)

        self._ob_space = ob_space
        self._ac_space = ac_space

        ckpts = config.ps_ckpts.split("/")
        self._num_agents = len(ckpts)
        self._rl_agents = [
            GAILAgent(config, ob_space, ac_space, env_ob_space)
            for _ in ckpts
        ]
        for i, ckpt in enumerate(ckpts):
            ckpt_dir = os.path.join(config.ps_dir, ckpt)
            ckpt_path, ckpt_num = get_ckpt_path(ckpt_dir, ckpt_num=None)
            assert ckpt_path is not None

            logger.warn("Load checkpoint %s", ckpt_path)
            self._rl_agents[i].load_state_dict(
                torch.load(ckpt_path, map_location=self._config.device)["agent"]
            )

        if config.is_train:
            self._discriminators = []
            self._discriminator_losses = []
            for i in range(self._num_agents):
                self._discriminators[i] = Discriminator(config, ob_space, None)
                self._discriminator_losses[i] = nn.BCEWithLogitsLoss()
            self._network_cuda(config.device)

            self._discriminator_optims = []
            self._discriminator_lr_schedulers = []
            for i in range(self._num_agents):
                # build optimizers
                self._discriminator_optims[i]= optim.Adam(
                    self._discriminators[i].parameters(), lr=config.discriminator_lr
                )

                # build learning rate scheduler
                self._discriminator_lr_schedulers[i] = StepLR(
                    self._discriminator_optims[i],
                    step_size=self._config.max_global_step // self._config.rollout_length // 5,
                    gamma=0.5,
                )

        # expert dataset
        if config.is_train:
            self._datasets = []
            self._data_loaders = []
            self._data_iters = []
            for i in range(self._num_agents):
                self._datasets[i] = ExpertDataset(
                    config.demo_path,
                    config.demo_subsample_interval,
                    ac_space,
                    use_low_level=config.demo_low_level,
                    sample_range_start=config.demo_sample_range_start,
                    sample_range_end=config.demo_sample_range_end,
                )
                self._data_loaders[i] = torch.utils.data.DataLoader(
                    self._datasets[i],
                    batch_size=self._config.batch_size,
                    shuffle=True,
                    drop_last=True,
                )
                self._data_iters[i] = iter(self._data_loaders[i])

            # policy dataset
            sampler = RandomSampler()
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

            for i in range(self._num_agents):
                self._rl_agents[i].set_buffer(self._buffer)

        self._log_creation()

    def __getitem__(self, key):
        return self._rl_agents[key]

    def _log_creation(self):
        if self._config.is_chef:
            logger.info("Creating a policy sequencing agent")

    def is_off_policy(self):
        return False

    def store_episode(self, rollouts):
        pass

    def state_dict(self):
        return {
            "rl_agents": [agent.state_dict() for agent in self._rl_agents],
            "discriminators_state_dict": [d.state_dict() for d in self._discriminators],
            "discriminator_optims_state_dict": [o.state_dict() for o in self._discriminator_optims],
        }

    def load_state_dict(self, ckpt):
        for i in range(self._num_agents):
            self._rl_agents[i].load_state_dict(ckpt["rl_agents"][i])
            self._discriminators[i].load_state_dict(ckpt["discriminators_state_dict"][i])
            self._discriminator_optims[i].load_state_dict(
                ckpt["discriminator_optims_state_dict"][i]
            )
            optimizer_cuda(self._discriminator_optims[i], self._config.device)
        self._network_cuda(self._config.device)

    def _network_cuda(self, device):
        for i in range(self._num_agents):
            self._discriminators[i].to(device)

    def sync_networks(self):
        for i in range(self._num_agents):
            self._rl_agents[i].sync_networks()
            sync_networks(self._discriminators[i])

    def update_normalizer(self, obs=None):
        pass

    def train(self):
        train_info = Info()

        num_batches = (
            self._config.rollout_length
            // self._config.batch_size
            // self._config.discriminator_update_freq
        )
        assert num_batches > 0

        for i in range(self._num_agents):
            self._discriminator_lr_schedulers[i].step()

            for _ in range(num_batches):
                policy_data = self._buffer.sample(self._config.batch_size)
                try:
                    expert_data = next(self._data_iter)
                except StopIteration:
                    self._data_iter = iter(self._data_loader)
                    expert_data = next(self._data_iter)

                _train_info = self._update_discriminator(i, policy_data, expert_data)
                train_info.add(_train_info)

            _train_info = self._rl_agents[i].train()
            train_info.add(_train_info)

            for _ in range(num_batches):
                try:
                    expert_data = next(self._data_iter)
                except StopIteration:
                    self._data_iter = iter(self._data_loader)
                    expert_data = next(self._data_iter)
                self.update_normalizer(expert_data["ob"])

        return train_info.get_dict(only_scalar=True)

    def _update_discriminator(self, i, policy_data, expert_data):
        info = Info()

        _to_tensor = lambda x: to_tensor(x, self._config.device)
        # pre-process observations
        p_o = policy_data["ob"]
        p_o = self.normalize(p_o)
        p_o = _to_tensor(p_o)

        e_o = expert_data["ob"]
        e_o = self.normalize(e_o)
        e_o = _to_tensor(e_o)

        p_ac = None
        e_ac = None

        p_logit = self._discriminators[i](p_o, p_ac)
        e_logit = self._discriminators[i](e_o, e_ac)

        p_output = torch.sigmoid(p_logit)
        e_output = torch.sigmoid(e_logit)

        p_loss = self._discriminator_loss(
            p_logit, torch.zeros_like(p_logit).to(self._config.device)
        )
        e_loss = self._discriminator_loss(
            e_logit, torch.ones_like(e_logit).to(self._config.device)
        )

        logits = torch.cat([p_logit, e_logit], dim=0)
        entropy = torch.distributions.Bernoulli(logits=logits).entropy().mean()
        entropy_loss = -self._config.gail_entropy_loss_coeff * entropy

        gail_loss = p_loss + e_loss + entropy_loss

        # update the discriminator
        self._discriminators[i].zero_grad()
        gail_loss.backward()
        sync_grads(self._discriminators[i])
        self._discriminator_optims[i].step()

        info["gail_policy_output"] = p_output.mean().detach().cpu().item()
        info["gail_expert_output"] = e_output.mean().detach().cpu().item()
        info["gail_entropy"] = entropy.detach().cpu().item()
        info["gail_policy_loss"] = p_loss.detach().cpu().item()
        info["gail_expert_loss"] = e_loss.detach().cpu().item()
        info["gail_entropy_loss"] = entropy_loss.detach().cpu().item()
        info["gail_loss"] = gail_loss.detach().cpu().item()

        return mpi_average(info.get_dict(only_scalar=True))
