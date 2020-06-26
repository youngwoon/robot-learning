from . import make_env
from ..config import argparser


config, unparsed = argparser()

env = make_env(config.env, config)

ob = env.reset()

while True:
    ob, reward, done, info = env.step(env.action_space.sample())
    print(reward)
    if done:
        break
