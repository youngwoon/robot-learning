# RL algorithms
from .sac_agent import SACAgent
from .ppo_agent import PPOAgent
from .ddpg_agent import DDPGAgent

# IL algorithms
from .bc_agent import BCAgent
from .gail_agent import GAILAgent
from .dac_agent import DACAgent


RL_ALGOS = {
    "sac": SACAgent,
    "ppo": PPOAgent,
    "ddpg": DDPGAgent,
    "td3": DDPGAgent,
}


IL_ALGOS = {
    "bc": BCAgent,
    "gail": GAILAgent,
    "dac": DACAgent,
}


def get_agent_by_name(algo):
    """
    Returns RL or IL agent.
    """
    if algo in RL_ALGOS:
        return RL_ALGOS[algo]
    elif algo in IL_ALGOS:
        return IL_ALGOS[algo]
    else:
        raise ValueError("--algo %s is not supported" % algo)
