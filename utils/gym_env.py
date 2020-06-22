import gym


class DictWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self._is_ob_dict = isinstance(env.observation_space, gym.spaces.Dict)
        if not self._is_ob_dict:
            self.observation_space = gym.spaces.Dict({"ob": env.observation_space})
        else:
            self.observation_space = env.observation_space

        self._is_ac_dict = isinstance(env.action_space, gym.spaces.Dict)
        if not self._is_ac_dict:
            self.action_space = gym.spaces.Dict({"ac": env.action_space})
        else:
            self.action_space = env.action_space

    def reset(self):
        ob = super().reset()
        if not self._is_ob_dict:
            ob = {"ob": ob}
        return ob

    def step(self, ac):
        if not self._is_ac_dict:
            ac = ac["ac"]
        ob, reward, done, info = self.env.step(ac)
        if not self._is_ob_dict:
            ob = {"ob": ob}
        return ob, reward, done, info
