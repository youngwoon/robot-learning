"""
Code reference:
  https://github.com/MishaLaskin/rad/blob/master/encoder.py
"""

import gym.spaces
import torch
import torch.nn as nn

from .utils import CNN, MLP, flatten_ac


class Encoder(nn.Module):
    def __init__(self, config, ob_space):
        super().__init__()

        self._encoder_type = config.encoder_type
        self._ob_space = ob_space

        self.base = nn.ModuleDict()
        encoder_output_dim = 0
        for k, v in ob_space.spaces.items():
            if len(v.shape) in [3, 4]:
                if self._encoder_type == "mlp":
                    self.base[k] = None
                    encoder_output_dim += gym.spaces.flatdim(v)
                else:
                    if len(v.shape) == 3:
                        image_dim = v.shape[0]
                    elif len(v.shape) == 4:
                        image_dim = v.shape[0] * v.shape[1]
                    self.base[k] = CNN(config, image_dim)
                    encoder_output_dim += self.base[k].output_dim
            elif len(v.shape) == 1:
                self.base[k] = None
                encoder_output_dim += gym.spaces.flatdim(v)
            else:
                raise ValueError("Check the shape of observation %s (%s)" % (k, v))

        self.output_dim = encoder_output_dim

    def forward(self, ob, detach_conv=False):
        encoder_outputs = []
        for k, v in ob.items():
            if self.base[k] is not None:
                if isinstance(self.base[k], CNN):
                    if v.max() > 1.0:
                        v = v.float() / 255.0
                encoder_outputs.append(
                    self.base[k](v, detach_conv=detach_conv)
                )
            else:
                encoder_outputs.append(v.flatten(start_dim=1))
        out = torch.cat(encoder_outputs, dim=-1)
        assert len(out.shape) == 2
        return out

    def copy_conv_weights_from(self, source):
        """ Tie convolutional layers """
        for k in self.base.keys():
            if self.base[k] is not None:
                self.base[k].copy_conv_weights_from(source.base[k])
