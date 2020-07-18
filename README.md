# Robot Learning Framework for Research


## RL algorithms
* PPO
* DDPG
* TD3
* SAC


## IL algorithms
* BC
* GAIL
* DAC


## Directories
* `run.py`: simply launches `main.py`
* `main.py`: sets up experiment and runs training using `trainer.py`
* `trainer.py`: contains training and evaluation code
* `algorithms/`: implementation of all RL and IL algorithms
* `config/`: hyper-parameters in `config/__init__.py`
* `environments/`: registers environments (OpenAI Gym and Deepmind Control Suite)
* `networks/`: implementation of networks, such as policy and value function
* `utils/`: contains helper functions


## Prerequisites
* Ubuntu 18.04 or above
* Python 3.6
* Mujoco 2.0


## Installation

1. Install mujoco 2.0 and add the following environment variables into `~/.bashrc` or `~/.zshrc`
```bash
# download mujoco 2.0
$ wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip
$ unzip mujoco.zip -d ~/.mujoco
$ cp -r ~/.mujoco/mujoco200_linux ~/.mujoco/mujoco200

# copy mujoco license key `mjkey.txt` to `~/.mujoco`

# add mujoco to LD_LIBRARY_PATH
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin

# for GPU rendering (replace 418 with your nvidia driver version or you can make a dummy directory /usr/lib/nvidia-000)
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-418

# only for a headless server
$ export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-418/libGL.so
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

### PPO
```bash
$ python -m run --run_prefix test --algo ppo --env "Hopper-v2"
```

### DDPG
```bash
$ python -m run --run_prefix test --algo ddpg --env "Hopper-v2"
```

### TD3
```bash
$ python -m run --run_prefix test --algo td3 --env "Hopper-v2"
```

### SAC
```bash
$ python -m run --run_prefix test --algo sac --env "Hopper-v2"
```

### BC
1. Generate demo using PPO
```bash
# train ppo expert agent
$ python -m run --run_prefix test --algo ppo --env "Hopper-v2"
# collect expert trajectories using ppo expert policy
$ python -m run --run_prefix test --algo ppo --env "Hopper-v2" --is_train False --record_video False --record_demo True --num_eval 100
# 100 trajectories are stored in log/Hopper-v2.ppo.test.123/demo/Hopper-v2.ppo.test.123_step_00001000000_100.pkl
```

2. Run BC
```bash
$ python -m run --run_prefix test --algo bc --env "Hopper-v2" --demo_path log/Hopper-v2.ppo.test.123/demo/Hopper-v2.ppo.test.123_step_00001000000_100.pkl
```

### GAIL
```bash
$ python -m run --run_prefix test --algo gail --env "Hopper-v2" --demo_path log/Hopper-v2.ppo.test.123/demo/Hopper-v2.ppo.test.123_step_00001000000_100.pkl

# initialize with BC policy
$ python -m run --run_prefix test --algo gail --env "Hopper-v2" --demo_path log/Hopper-v2.ppo.test.123/demo/Hopper-v2.ppo.test.123_step_00001000000_100.pkl --init_ckpt_path log/Hopper-v2.bc.test.123/ckpt_00000020.pt
```


## To dos
* BC intialization for all algorithms
* Ray
* HER
* Skill coordination

