from .base_agent import BaseAgent

# RL algorithms
from .sac_agent import SACAgent
from .ppo_agent import PPOAgent
from .ddpg_agent import DDPGAgent
from .dreamer_agent import DreamerAgent
from .tdmpc_agent import TDMPCAgent

# IL algorithms
from .bc_agent import BCAgent
from .gail_agent import GAILAgent
from .dac_agent import DACAgent


RL_ALGOS = {
    "sac": SACAgent,
    "ppo": PPOAgent,
    "ddpg": DDPGAgent,
    "td3": DDPGAgent,
    "dreamer": DreamerAgent,
    "tdmpc": TDMPCAgent,
}


IL_ALGOS = {
    "bc": BCAgent,
    "gail": GAILAgent,
    "dac": DACAgent,
}
