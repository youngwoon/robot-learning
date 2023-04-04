import torch
import torch.nn as nn
import tensorflow as tf


class EMANorm(nn.Module):
    """Normalizes values to [0, 1] using EMA."""

    def __init__(
        self,
        update_weight=0.95,
        min_scale=1.0,
        clip_percentage_low=0.05,
        clip_percentage_high=0.95,
    ):
        super().__init__()
        self._update_weight = update_weight
        self._min_scale = min_scale
        self._clip_perc_low = clip_percentage_low
        self._clip_perc_high = clip_percentage_high
        self.low = nn.parameter.Parameter(torch.tensor([0.0]), requires_grad=False)
        self.high = nn.parameter.Parameter(torch.tensor([0.0]), requires_grad=False)

    def __call__(self, v, update=True):
        if update:
            self.low.data.lerp_(
                torch.quantile(v, self._clip_perc_low), 1 - self._update_weight
            )
            self.high.data.lerp_(
                torch.quantile(v, self._clip_perc_high), 1 - self._update_weight
            )

        offset = self.low.data
        scale = torch.clamp(self.high.data - self.low.data, min=self._min_scale)
        return (v - offset) / scale


def static_scan(fn, inputs, start, reverse=False):
    """Applies `fn(state, input[t])` sequentially `T=len(inputs)` times and returns intermediate states.

    Args:
        fn: Function to apply.
        inputs: A list of `T` inputs.
        start: A list of `M` initial states.

    Returns `M` lists of `T` states.
    """
    last = start
    outputs = [[] for _ in tf.nest.flatten(start)]
    indices = range(len(tf.nest.flatten(inputs)[0]))
    if reverse:
        indices = reversed(indices)
    for index in indices:
        inp = tf.nest.map_structure(lambda x: x[index], inputs)
        last = fn(last, inp)
        [o.append(l) for o, l in zip(outputs, tf.nest.flatten(last))]
    if reverse:
        outputs = [list(reversed(x)) for x in outputs]
    outputs = [torch.stack(x) for x in outputs]
    return tf.nest.pack_sequence_as(start, outputs)


def lambda_return(reward, value, pcont, bootstrap, lambda_):
    # Setting lambda=1 gives a discounted Monte Carlo return.
    # Setting lambda=0 gives a fixed 1-step return.
    assert len(reward.shape) == len(value.shape), (reward.shape, value.shape)
    if isinstance(pcont, (int, float)):
        pcont = pcont * torch.ones_like(reward)
    next_values = torch.cat([value[1:], bootstrap[None]], 0)
    inputs = reward + pcont * next_values * (1 - lambda_)
    returns = static_scan(
        lambda agg, cur: cur[0] + cur[1] * lambda_ * agg,
        (inputs, pcont),
        bootstrap,
        reverse=True,
    )
    return returns
