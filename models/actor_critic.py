import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

import distrax


class ActorCriticConvSymbolicCraftax(nn.Module):
    action_dim: Sequence[int]
    map_obs_shape: Sequence[int]
    layer_width: int

    @nn.compact
    def __call__(self, obs):
        # Split into map and flat obs
        flat_map_obs_shape = (
            self.map_obs_shape[0] * self.map_obs_shape[1] * self.map_obs_shape[2]
        )
        image_obs = obs[:, :flat_map_obs_shape]
        image_dim = self.map_obs_shape
        image_obs = image_obs.reshape((image_obs.shape[0], *image_dim))

        flat_obs = obs[:, flat_map_obs_shape:]

        # Convolutions on map
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_obs)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = nn.Conv(features=32, kernel_size=(2, 2))(image_embedding)
        image_embedding = nn.relu(image_embedding)
        image_embedding = nn.max_pool(
            image_embedding, window_shape=(2, 2), strides=(1, 1)
        )
        image_embedding = image_embedding.reshape(image_embedding.shape[0], -1)
        # image_embedding = jnp.concatenate([image_embedding, obs[:, : CraftaxEnv.get_flat_map_obs_shape()]], axis=-1)

        # Combine embeddings
        embedding = jnp.concatenate([image_embedding, flat_obs], axis=-1)
        embedding = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticConv(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, obs):
        x = nn.Conv(features=32, kernel_size=(5, 5))(obs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = nn.Conv(features=32, kernel_size=(5, 5))(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        embedding = x.reshape(x.shape[0], -1)

        actor_mean = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticConvWorldModel(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    def setup(self):
        self.enc_conv1 = nn.Conv(
            features=32, kernel_size=(5, 5), name="enc_conv1"
        )
        self.enc_conv2 = nn.Conv(
            features=32, kernel_size=(5, 5), name="enc_conv2"
        )
        self.enc_conv3 = nn.Conv(
            features=32, kernel_size=(5, 5), name="enc_conv3"
        )
        self.enc_latent = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="enc_latent",
        )

        self.actor_fc1 = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="actor_fc1",
        )
        self.actor_fc2 = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_fc2",
        )
        self.actor_out = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="actor_out",
        )

        self.critic_fc1 = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="critic_fc1",
        )
        self.critic_out = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="critic_out",
        )

        self.wm_fc1 = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="wm_fc1",
        )
        self.wm_fc2 = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="wm_fc2",
        )
        self.wm_next_latent = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="wm_next_latent",
        )
        self.wm_reward = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="wm_reward",
        )
        self.wm_done = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
            name="wm_done",
        )

        self.inv_fc1 = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
            name="inv_fc1",
        )
        self.inv_action = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            name="inv_action",
        )

    def encode(self, obs):
        x = self.enc_conv1(obs)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = self.enc_conv2(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))
        x = self.enc_conv3(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(3, 3))

        x = x.reshape(x.shape[0], -1)
        latent = self.enc_latent(x)
        latent = nn.relu(latent)
        return latent

    def __call__(self, obs):
        latent = self.encode(obs)

        actor_x = self.actor_fc1(latent)
        actor_x = nn.relu(actor_x)
        actor_x = self.actor_fc2(actor_x)
        actor_x = nn.relu(actor_x)
        actor_logits = self.actor_out(actor_x)
        pi = distrax.Categorical(logits=actor_logits)

        critic_x = self.critic_fc1(latent)
        critic_x = nn.relu(critic_x)
        value = self.critic_out(critic_x)

        return pi, jnp.squeeze(value, axis=-1), latent

    def world_model(self, latent, action):
        action_onehot = jax.nn.one_hot(action, self.action_dim)
        x = jnp.concatenate([latent, action_onehot], axis=-1)
        x = self.wm_fc1(x)
        x = nn.relu(x)
        x = self.wm_fc2(x)
        x = nn.relu(x)

        pred_next_latent = self.wm_next_latent(x)
        pred_reward = self.wm_reward(x)
        pred_done_logit = self.wm_done(x)

        return (
            pred_next_latent,
            jnp.squeeze(pred_reward, axis=-1),
            jnp.squeeze(pred_done_logit, axis=-1),
        )

    def inverse_model(self, latent, next_latent):
        x = jnp.concatenate([latent, next_latent], axis=-1)
        x = self.inv_fc1(x)
        x = nn.relu(x)
        pred_action_logits = self.inv_action(x)
        return pred_action_logits

    def init_all(self, obs, action, next_obs):
        pi, value, latent = self(obs)
        _, _, next_latent = self(next_obs)
        self.world_model(latent, action)
        self.inverse_model(latent, next_latent)
        return pi, value, latent


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


class ActorCriticWithEmbedding(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_emb = activation(actor_emb)

        actor_emb = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_emb)
        actor_emb = activation(actor_emb)

        actor_emb = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_emb)
        actor_emb = activation(actor_emb)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_emb)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic)
        critic = activation(critic)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1), actor_emb
