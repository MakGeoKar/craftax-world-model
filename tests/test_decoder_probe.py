import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.latent_obs_decoder import LatentObsDecoder, encoder_spatial_pyramid


def test_encoder_spatial_pyramid_matches_63_input():
    sizes = encoder_spatial_pyramid(63, 63)
    assert sizes == ((21, 21), (7, 7), (2, 2))


def test_decoder_output_shape():
    obs_shape = (63, 63, 3)
    layer_width = 64
    decoder = LatentObsDecoder(layer_width=layer_width, obs_shape=obs_shape)
    rng = jax.random.PRNGKey(0)
    z = jnp.zeros((4, layer_width))
    params = decoder.init(rng, z)
    obs_hat = decoder.apply(params, z)
    assert obs_hat.shape == (4, *obs_shape)


def test_wm_params_receive_no_gradients():
    from analysis.train_decoder_probe import verify_wm_frozen
    from models.actor_critic import ActorCriticConvWorldModel

    obs_shape = (63, 63, 3)
    action_dim = 17
    layer_width = 64
    wm_network = ActorCriticConvWorldModel(action_dim, layer_width=layer_width)
    decoder = LatentObsDecoder(layer_width=layer_width, obs_shape=obs_shape)

    rng = jax.random.PRNGKey(0)
    rng, wm_rng, dec_rng = jax.random.split(rng, 3)
    init_obs = jnp.zeros((1, *obs_shape))
    init_action = jnp.zeros((1,), dtype=jnp.int32)
    wm_params = wm_network.init(
        wm_rng, init_obs, init_action, init_obs, method=wm_network.init_all
    )
    z = wm_network.apply(
        wm_params, init_obs, method=wm_network.encode
    )
    decoder_params = decoder.init(dec_rng, z)

    verify_wm_frozen(
        decoder.apply,
        wm_network,
        wm_params,
        decoder_params,
        init_obs,
        repr_alpha=0.0,
    )
