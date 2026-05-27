"""Diagnostic latent-to-observation decoder (visualization probe only).

This module is NOT part of the RL objective. It reconstructs observations from
frozen world-model encoder latents so we can inspect what information the latent
state retains. Training must keep WM/encoder gradients disabled.
"""

from typing import Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal


def encoder_spatial_pyramid(
    height: int,
    width: int,
    num_pools: int = 3,
    pool_size: int = 3,
    pool_stride: int = 3,
) -> Tuple[Tuple[int, int], ...]:
    """Spatial sizes after each encoder max-pool (matches ActorCriticConvWorldModel.encode)."""
    h, w = height, width
    sizes = []
    for _ in range(num_pools):
        h = (h - pool_size) // pool_stride + 1
        w = (w - pool_size) // pool_stride + 1
        sizes.append((h, w))
    return tuple(sizes)


class LatentObsDecoder(nn.Module):
    """Flat latent -> observation decoder for diagnostic visualization."""

    layer_width: int
    obs_shape: Sequence[int]  # (H, W, C) NHWC, matching Craftax pixel envs
    bottleneck_channels: int = 32

    def setup(self):
        height, width, _ = self.obs_shape
        self.spatial_pyramid = encoder_spatial_pyramid(height, width)
        bottleneck_h, bottleneck_w = self.spatial_pyramid[-1]
        bottleneck_dim = (
            bottleneck_h * bottleneck_w * self.bottleneck_channels
        )

        self.dec_fc = nn.Dense(
            bottleneck_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="dec_fc",
        )

        upsample_targets = list(reversed(self.spatial_pyramid[:-1])) + [
            (height, width)
        ]
        self.upsample_targets = upsample_targets
        for idx in range(len(upsample_targets)):
            setattr(
                self,
                f"dec_up{idx}_conv",
                nn.Conv(
                    features=self.bottleneck_channels,
                    kernel_size=(3, 3),
                    kernel_init=orthogonal(2),
                    bias_init=constant(0.0),
                    name=f"dec_up{idx}_conv",
                ),
            )

        _, _, channels = self.obs_shape
        self.dec_out = nn.Conv(
            features=channels,
            kernel_size=(3, 3),
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="dec_out",
        )

    def __call__(self, latent: jnp.ndarray) -> jnp.ndarray:
        x = self.dec_fc(latent)
        x = nn.relu(x)

        bottleneck_h, bottleneck_w = self.spatial_pyramid[-1]
        x = x.reshape(x.shape[0], bottleneck_h, bottleneck_w, self.bottleneck_channels)

        for idx, (target_h, target_w) in enumerate(self.upsample_targets):
            x = jax.image.resize(
                x,
                (x.shape[0], target_h, target_w, x.shape[-1]),
                method="nearest",
            )
            conv = getattr(self, f"dec_up{idx}_conv")
            x = conv(x)
            x = nn.relu(x)

        x = self.dec_out(x)
        return nn.sigmoid(x)
