from collections import defaultdict, deque
from functools import partial

import numpy as np

from ..utils.pytorch import random_crop
from ..utils.logger import logger


# Methods to support recursive dictionary observations and actions.
def _make_buffer(shapes, buffer_size):
    buffer = {}
    for k, v in shapes.items():
        if isinstance(v, dict):
            buffer[k] = _make_buffer(v, buffer_size)
        else:
            if len(v) >= 3:
                buffer[k] = np.empty((buffer_size, *v), dtype=np.uint8)
            else:
                buffer[k] = np.empty((buffer_size, *v), dtype=np.float32)
    return buffer


def _add_rollout(buffer, rollout, idx: int):
    if isinstance(rollout, list):
        rollout = rollout[0]

    if isinstance(rollout, dict):
        for k in rollout.keys():
            _add_rollout(buffer[k], rollout[k], idx)
    else:
        np.copyto(buffer[idx], rollout)


def _get_batch(buffer: dict, idxs):
    batch = {}
    for k in buffer.keys():
        if isinstance(buffer[k], dict):
            batch[k] = _get_batch(buffer[k], idxs)
        else:
            batch[k] = buffer[k][idxs]
    return batch


def _augment_ob(batch, image_crop_size):
    for k, v in batch.items():
        if isinstance(batch[k], dict):
            _augment_ob(batch[k], image_crop_size)
        elif len(batch[k].shape) > 3:
            batch[k] = random_crop(batch[k], image_crop_size)


class ReplayBuffer(object):
    """ Replay buffer to store trainsitions in list (deque). """

    def __init__(self, keys, buffer_size, sample_func):
        self._capacity = buffer_size
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = keys
        self.clear()

    @property
    def size(self):
        return self._capacity

    @property
    def last_saved_idx(self):
        return self._last_saved_idx

    def clear(self):
        self._idx = 0
        self._current_size = 0
        self._last_saved_idx = -1
        self._buffer = defaultdict(partial(deque, maxlen=self._capacity))

    def store_episode(self, rollout):
        """ @rollout can be any length of transitions. """
        for k in self._keys:
            self._buffer[k].append(rollout[k])

        self._idx += 1
        if self._current_size < self._capacity:
            self._current_size += 1

    def sample(self, batch_size):
        transitions = self._sample_func(self._buffer, batch_size)
        return transitions

    def state_dict(self):
        """ Returns new transitions in replay buffer. """
        assert self._idx - self._last_saved_idx - 1 <= self._capacity
        state_dict = {}
        s = (self._last_saved_idx + 1) % self._capacity
        e = (self._idx - 1) % self._capacity
        for k in self._keys:
            state_dict[k] = list(self._buffer[k])
            if s < e:
                state_dict[k] = state_dict[k][s : e + 1]
            else:
                state_dict[k] = state_dict[k][s:] + state_dict[k][: e + 1]
            assert len(state_dict[k]) == self._idx - self._last_saved_idx - 1
        self._last_saved_idx = self._idx - 1
        logger.info("Store %d states", len(state_dict["ac"]))
        return state_dict

    def append_state_dict(self, state_dict):
        """ Adds transitions to replay buffer. """
        for k in self._keys:
            self._buffer[k].extend(state_dict[k])

        n = len(state_dict["ac"])
        self._last_saved_idx += n
        self._idx += n
        logger.info("Load %d states", n)

    def load_state_dict(self, state_dict):
        n = len(self._buffer["ac"])
        self._buffer = state_dict
        self._current_size = min(self._current_size + n, self._capacity)
        self._last_saved_idx += n
        self._idx += n


class ReplayBufferPerStep(object):
    """ Replay buffer to store transitions in numpy array. """

    def __init__(
        self, shapes: dict, buffer_size: int, image_crop_size=84, absorbing_state=False
    ):
        self._capacity = buffer_size

        if absorbing_state:
            shapes["ob"]["absorbing_state"] = [1]
            shapes["ob_next"]["absorbing_state"] = [1]

        self._shapes = shapes
        self._keys = list(shapes.keys())
        self._image_crop_size = image_crop_size
        self._absorbing_state = absorbing_state

        self._buffer = _make_buffer(shapes, buffer_size)
        self._idx = 0
        self._full = False

    def clear(self):
        self._idx = 0
        self._full = False

    def store_episode(self, rollout):
        for k in self._keys:
            _add_rollout(self._buffer[k], rollout[k], self._idx)

        self._idx = (self._idx + 1) % self._capacity
        self._full = self._full or self._idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(
            0, self._capacity if self._full else self._idx, size=batch_size
        )
        batch = _get_batch(self._buffer, idxs)

        # apply random crop to image
        _augment_ob(batch, self._image_crop_size)

        return batch

    def state_dict(self):
        return {"buffer": self._buffer, "idx": self._idx, "full": self._full}

    def load_state_dict(self, state_dict):
        self._buffer = state_dict["buffer"]
        self._idx = state_dict["idx"]
        self._full = state_dict["full"]


class ReplayBufferEpisode(object):
    """ Replay buffer to store episodes in list. """

    def __init__(self, keys, buffer_size, sample_func):
        self._capacity = buffer_size
        self._sample_func = sample_func

        # create the buffer to store info
        self._keys = keys
        self.clear()

    def clear(self):
        self._idx = 0
        self._current_size = 0
        self._new_episode = True
        self._buffer = defaultdict(list)

    def store_episode(self, rollout):
        if self._new_episode:
            self._new_episode = False
            for k in self._keys:
                if self._current_size < self._capacity:
                    self._buffer[k].append(rollout[k])
                else:
                    self._buffer[k][self._idx] = rollout[k]
        else:
            for k in self._keys:
                self._buffer[k][self._idx].extend(rollout[k])

        if rollout["done"][-1]:
            self._idx = (self._idx + 1) % self._capacity
            if self._current_size < self._capacity:
                self._current_size += 1
            self._new_episode = True

    def sample(self, batch_size):
        transitions = self._sample_func(self._buffer, batch_size)
        return transitions

    def state_dict(self):
        return self._buffer

    def load_state_dict(self, state_dict):
        self._buffer = state_dict
        self._current_size = len(self._buffer["ac"])


class RandomSampler(object):
    def __init__(self, image_crop_size=84):
        self._image_crop_size = image_crop_size

    def sample_func(self, episode_batch, batch_size_in_transitions):
        key_len = "ac" if "ac" in episode_batch else "ob"
        rollout_batch_size = len(episode_batch[key_len])
        batch_size = batch_size_in_transitions

        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = [
            np.random.randint(len(episode_batch[key_len][episode_idx]))
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        if "ob_next" in episode_batch:
            transitions["ob_next"] = [
                episode_batch["ob_next"][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        for k, v in new_transitions["ob"].items():
            if len(v.shape) in [4, 5]:
                new_transitions["ob"][k] = random_crop(v, self._image_crop_size)

        if "ob_next" in episode_batch:
            for k, v in new_transitions["ob_next"].items():
                if len(v.shape) in [4, 5]:
                    new_transitions["ob_next"][k] = random_crop(
                        v, self._image_crop_size
                    )

        return new_transitions


class SeqSampler(object):
    def __init__(self, seq_length, image_crop_size=84):
        self._seq_length = seq_length
        self._image_crop_size = image_crop_size

    def sample_func(self, episode_batch, batch_size_in_transitions):
        rollout_batch_size = len(episode_batch["ac"])
        batch_size = batch_size_in_transitions

        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = [
            np.random.randint(len(episode_batch["ac"][episode_idx]))
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        transitions["ob_next"] = [
            episode_batch["ob_next"][episode_idx][t]
            for episode_idx, t in zip(episode_idxs, t_samples)
        ]

        # Create a key that stores the specified future fixed length of sequences, pad last states if necessary

        print(episode_idxs)
        print(t_samples)

        # List of dictionaries is created here..., flatten it out?
        transitions["following_sequences"] = [
            episode_batch["ob"][episode_idx][t : t + self._seq_length]
            for episode_idx, t in zip(episode_idxs, t_samples)
        ]

        # something's wrong here... should use index episode_idx to episode_batch, not transitions

        # # Pad last states
        # for episode_idx in episode_idxs:
        #     # curr_ep = episode_batch["ob"][episode_idx]
        #     # curr_ep.extend(curr_ep[-1:] * (self._seq_length - len(curr_ep)))
        #
        #     #all list should have 10 dictionaries now
        #     if isinstance(transitions["following_sequences"][episode_idx], dict):
        #         continue
        #     transitions["following_sequences"][episode_idx].extend(transitions["following_sequences"][episode_idx][-1:] * (self._seq_length - len(transitions["following_sequences"][episode_idx])))
        #
        #     #turn transitions["following_sequences"] to a dictionary
        #     fs_list = transitions["following_sequences"][episode_idx]
        #     container = {}
        #     container["ob"] = []
        #     for i in fs_list:
        #         container["ob"].extend(i["ob"])
        #     container["ob"] = np.array(container["ob"])
        #     transitions["following_sequences"][episode_idx] = container

        # Pad last states
        for i in range(len(transitions["following_sequences"])):
            # curr_ep = episode_batch["ob"][episode_idx]
            # curr_ep.extend(curr_ep[-1:] * (self._seq_length - len(curr_ep)))

            # all list should have 10 dictionaries now
            if isinstance(transitions["following_sequences"][i], dict):
                continue
            transitions["following_sequences"][i].extend(
                transitions["following_sequences"][i][-1:]
                * (self._seq_length - len(transitions["following_sequences"][i]))
            )

            # turn transitions["following_sequences"] to a dictionary
            fs_list = transitions["following_sequences"][i]
            container = {}
            container["ob"] = []
            for j in fs_list:
                container["ob"].extend(j["ob"])
            container["ob"] = np.array(container["ob"])
            transitions["following_sequences"][i] = container

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        for k, v in new_transitions["ob"].items():
            if len(v.shape) in [4, 5]:
                new_transitions["ob"][k] = random_crop(v, self._image_crop_size)

        for k, v in new_transitions["ob_next"].items():
            if len(v.shape) in [4, 5]:
                new_transitions["ob_next"][k] = random_crop(v, self._image_crop_size)

        return new_transitions


class HERSampler(object):
    def __init__(self, replay_strategy, replace_future, reward_func=None):
        self.replay_strategy = replay_strategy
        if self.replay_strategy == "future":
            self.future_p = replace_future
        else:
            self.future_p = 0
        self.reward_func = reward_func

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        rollout_batch_size = len(episode_batch["ac"])
        batch_size = batch_size_in_transitions

        # select which rollouts and which timesteps to be used
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size)
        t_samples = [
            np.random.randint(len(episode_batch["ac"][episode_idx]))
            for episode_idx in episode_idxs
        ]

        transitions = {}
        for key in episode_batch.keys():
            transitions[key] = [
                episode_batch[key][episode_idx][t]
                for episode_idx, t in zip(episode_idxs, t_samples)
            ]

        transitions["ob_next"] = [
            episode_batch["ob"][episode_idx][t + 1]
            for episode_idx, t in zip(episode_idxs, t_samples)
        ]
        transitions["r"] = np.zeros((batch_size,))

        # hindsight experience replay
        for i, (episode_idx, t) in enumerate(zip(episode_idxs, t_samples)):
            replace_goal = np.random.uniform() < self.future_p
            if replace_goal:
                future_t = np.random.randint(
                    t + 1, len(episode_batch["ac"][episode_idx]) + 1
                )
                future_ag = episode_batch["ag"][episode_idx][future_t]
                if (
                    self.reward_func(
                        episode_batch["ag"][episode_idx][t], future_ag, None
                    )
                    < 0
                ):
                    transitions["g"][i] = future_ag
            transitions["r"][i] = self.reward_func(
                episode_batch["ag"][episode_idx][t + 1], transitions["g"][i], None
            )

        new_transitions = {}
        for k, v in transitions.items():
            if isinstance(v[0], dict):
                sub_keys = v[0].keys()
                new_transitions[k] = {
                    sub_key: np.stack([v_[sub_key] for v_ in v]) for sub_key in sub_keys
                }
            else:
                new_transitions[k] = np.stack(v)

        return new_transitions
