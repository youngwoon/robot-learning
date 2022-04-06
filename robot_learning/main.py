""" Launch RL/IL training and evaluation. """

import sys
import signal
import os
import json
import logging

import numpy as np
import torch
import wandb
from six.moves import shlex_quote
from mpi4py import MPI

from .trainer import Trainer
from .utils.logger import logger
from .utils.mpi import mpi_sync


class Run(object):
    """ Class for experiment run. """

    def __init__(self, parser):
        self._config, unparsed = parser.parse_known_args()
        config = self._config
        if len(unparsed):
            logger.error("Unparsed argument is detected:\n%s", unparsed)
            sys.exit()

        self._set_run_name()
        self._set_log_path()

        rank = MPI.COMM_WORLD.Get_rank()
        config.rank = rank
        config.is_chef = rank == 0
        config.num_workers = MPI.COMM_WORLD.Get_size()

        config.seed += rank
        if hasattr(config, "port"):
            config.port += rank * 2  # training env + evaluation env

        if config.is_chef:
            logger.warning("Run a base worker.")
            self._make_log_files()
        else:
            logger.warning("Run worker %d and disable logger.", config.rank)
            logger.setLevel(logging.CRITICAL)

        # syncronize all processes
        mpi_sync()

        def shutdown(signal, frame):
            logger.warning("Received signal %s: exiting", signal)
            sys.exit(128 + signal)

        signal.signal(signal.SIGHUP, shutdown)
        signal.signal(signal.SIGINT, shutdown)
        signal.signal(signal.SIGTERM, shutdown)

        # set global seed
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

        np.set_printoptions(precision=3)
        np.set_printoptions(suppress=True)

        if config.gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(config.gpu)
            assert torch.cuda.is_available()
            config.device = torch.device("cuda")
        else:
            config.device = torch.device("cpu")

    def _set_run_name(self):
        """ Sets run name. """
        config = self._config
        config.run_name = "{}.{}.{}.{}".format(
            config.env, config.algo, config.run_prefix, config.seed
        )

    def _set_log_path(self):
        """ Sets paths to log directories. """
        config = self._config
        config.log_dir = os.path.join(config.log_root_dir, config.run_name)
        config.record_dir = os.path.join(config.log_dir, "video")
        config.demo_dir = os.path.join(config.log_dir, "demo")

    def _make_log_files(self):
        """ Sets up log directories and saves git diff and command line. """
        config = self._config

        logger.info("Create log directory: %s", config.log_dir)
        os.makedirs(config.log_dir, exist_ok=True)

        logger.info("Create video directory: %s", config.record_dir)
        os.makedirs(config.record_dir, exist_ok=True)

        logger.info("Create demo directory: %s", config.demo_dir)
        os.makedirs(config.demo_dir, exist_ok=True)

        if config.is_train:
            # log git diff
            git_path = os.path.join(config.log_dir, "git.txt")
            cmd_path = os.path.join(config.log_dir, "cmd.sh")
            cmds = [
                "echo `git rev-parse HEAD` >> {}".format(git_path),
                "git diff --submodule=diff >> {}".format(git_path),
                "echo 'python -m rl {}' >> {}".format(
                    " ".join([shlex_quote(arg) for arg in sys.argv[1:]]), cmd_path
                ),
            ]
            os.system("\n".join(cmds))

            # log config
            param_path = os.path.join(config.log_dir, "params.json")
            logger.info("Store parameters in %s", param_path)
            with open(param_path, "w") as fp:
                json.dump(config.__dict__, fp, indent=4, sort_keys=True)

            # setup wandb
            exclude = ["device"]

            wandb.init(
                resume=config.run_name,
                project=config.wandb_project,
                config={k: v for k, v in config.__dict__.items() if k not in exclude},
                dir=config.log_dir,
                entity=config.wandb_entity,
                notes=config.notes,
                mode="online" if config.wandb else "disabled",
            )
            wandb.save(git_path)

    def _get_trainer(self):
        return Trainer(self._config)

    def run(self):
        """ Runs Trainer. """
        trainer = self._get_trainer()
        if self._config.is_train:
            logger.info("Start training")
            trainer.train()
            logger.info("Finish training")
        else:
            logger.info("Start evaluating")
            trainer.evaluate()
            logger.info("Finish evaluating")


if __name__ == "__main__":
    # default arguments
    from .config import create_parser

    parser = create_parser()

    # change default values
    parser.set_defaults(wandb_entity="youngwoon")
    parser.set_defaults(wandb_project="robot-learning")

    # execute training code
    Run(parser).run()
