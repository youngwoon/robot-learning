# Robot Learning Framework for Research


## Reinforcement learning (RL) algorithms
* PPO
* DDPG
* TD3
* SAC


## Imitation learning (IL) algorithms
* BC
* GAIL
* DAC


## Directories
* `robot_learning`:
  * `main.py`: sets up experiment and runs training using `trainer.py`
  * `trainer.py`: contains training and evaluation code
  * `algorithms/`: implementation of all RL and IL algorithms
  * `config/`: hyperparameters in `config/__init__.py`
  * `environments/`: registers environments (OpenAI Gym and Deepmind Control Suite)
  * `networks/`: implementation of networks, such as policy and value function
  * `utils/`: contains helper functions


## Prerequisites
* Ubuntu 18.04 or above
* Python 3.6
* Mujoco 2.1


## Installation

1. Install mujoco 2.1 and add the following environment variables into `~/.bashrc` or `~/.zshrc`
```bash
# download mujoco 2.1
$ mkdir ~/.mujoco
$ wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco_linux.tar.gz
$ tar -xvzf mujoco_linux.tar.gz -C ~/.mujoco/
$ rm mujoco_linux.tar.gz

# add mujoco to LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin

# for GPU rendering
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

# only for a headless server
$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

2. Install python dependencies
```bash
$ sudo apt-get install cmake libopenmpi-dev libgl1-mesa-dev libgl1-mesa-glx libosmesa6-dev patchelf libglew-dev

# software rendering
$ sudo apt-get install libgl1-mesa-glx libosmesa6 patchelf

# window rendering
$ sudo apt-get install libglfw3 libglew2.0

$ pip install -r requirements.txt
```


## Usage

Use following commands to run RL/IL algorithms. Each experiment is represented as `[ENV].[ALGORITHM].[RUN_PREFIX].[SEED]` and checkpoints and videos are stored in `log/[ENV].[ALGORITHM].[RUN_PREFIX].[SEED]`. `--run_prefix` can be used to differentiate runs with different hyperparameters.


### PPO
```bash
$ python -m robot_learning.main --run_prefix test --algo ppo --env "Hopper-v2"
```

### DDPG
```bash
$ python -m robot_learning.main --run_prefix test --algo ddpg --env "Hopper-v2"
```

### TD3
```bash
$ python -m robot_learning.main --run_prefix test --algo td3 --env "Hopper-v2"
```

### SAC
```bash
$ python -m robot_learning.main --run_prefix test --algo sac --env "Hopper-v2"
```

### BC
1. Generate demo using PPO
```bash
# train ppo expert agent
$ python -m robot_learning.main --run_prefix test --algo ppo --env "Hopper-v2"
# collect expert trajectories using ppo expert policy
$ python -m robot_learning.main --run_prefix test --algo ppo --env "Hopper-v2" --is_train False --record_video False --record_demo True --num_eval 100
# 100 trajectories are stored in log/Hopper-v2.ppo.test.123/demo/Hopper-v2.ppo.test.123_step_00001000000_100.pkl
```

2. Run BC
```bash
$ python -m robot_learning.main --run_prefix test --algo bc --env "Hopper-v2" --demo_path log/Hopper-v2.ppo.test.123/demo/Hopper-v2.ppo.test.123_step_00001000000_100.pkl
```

### GAIL
```bash
$ python -m robot_learning.main --run_prefix test --algo gail --env "Hopper-v2" --demo_path log/Hopper-v2.ppo.test.123/demo/Hopper-v2.ppo.test.123_step_00001000000_100.pkl

# initialize with BC policy
$ python -m robot_learning.main --run_prefix test --algo gail --env "Hopper-v2" --demo_path log/Hopper-v2.ppo.test.123/demo/Hopper-v2.ppo.test.123_step_00001000000_100.pkl --init_ckpt_path log/Hopper-v2.bc.test.123/ckpt_00000020.pt
```


## Implement your own algorithm
Implement your own `run.py` for experiment setup, `your_config.py` for configuration, `your_trainer.py` for training/evaluation loop, `your_agent.py` for algorithm, `your_rollout.py` for rollout, `your_network.py` for models.

Please refer to [`skill-chaining` repository](https://github.com/clvrai/skill-chaining) for an example. It implements `run.py` for experiment setup, `policy_sequencing_config.py` for configuration, `policy_sequencing_trainer.py` for training/evaluation loop, `policy_sequencing_agent.py` for algorithm, `policy_sequencing_rollout.py` for rollout.


## To dos
* Ray
* HER
* Skill coordination
* Configuration with hydra


## Papers using this code
* [Adversarial Skill Chaining for Long-Horizon Robot Manipulation via Terminal State Regularization (CoRL 2021)](https://clvrai.com/skill-chaining)
* [Policy Transfer across Visual and Dynamics Domain Gaps via Iterative Grounding (RSS 2021)](https://clvrai.com/idapt)
* [IKEA Furniture Assembly Environment for Long-Horizon Complex Manipulation Tasks (ICRA 2021)](https://clvrai.com/furniture)
