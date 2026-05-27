"""
Visualize a trained WM-PPO agent with a latent (not pixel-generative) world model.

The world model operates in encoder latent space:
    pred_next_latent = world_model(encoder(obs_t), action_t)
and is compared against:
    next_latent = encoder(obs_{t+1})

This script does NOT decode predicted latents back to pixels.
It plots latent prediction error over time and highlights high-error transitions.
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from flax.serialization import from_bytes

# Allow running as: python analysis/visualize_world_model.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.actor_critic import ActorCriticConvWorldModel


def obs_to_image(obs):
    """Convert an observation array to an HWC image suitable for imshow."""
    image = np.asarray(obs)

    while image.ndim > 3:
        image = np.squeeze(image, axis=0)

    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (
        1,
        3,
        4,
    ):
        image = np.transpose(image, (1, 2, 0))

    if image.dtype == np.uint8:
        return image

    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def load_params_if_available(network, init_args, checkpoint_path):
    """Initialize network params; optionally load a trained checkpoint."""
    rng, init_obs, init_action, init_next_obs = init_args
    params = network.init(
        rng, init_obs, init_action, init_next_obs, method=network.init_all
    )

    if checkpoint_path is None:
        print("No checkpoint_path provided; using randomly initialized params.")
        return params

    checkpoint_file = Path(checkpoint_path)
    with checkpoint_file.open("rb") as f:
        bytes_data = f.read()
    params = from_bytes(params, bytes_data)
    print(f"Loaded trained params from {checkpoint_file.resolve()}")
    return params


def save_prediction_error_curve(errors, output_path):
    steps = np.arange(len(errors))
    plt.figure(figsize=(10, 4))
    plt.plot(steps, errors, linewidth=1.5)
    plt.xlabel("Step")
    plt.ylabel("Latent prediction MSE")
    plt.title("Latent world-model prediction error over rollout")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_topk_transition_grid(transitions, topk, output_path):
    n_rows = min(topk, len(transitions))
    fig, axes = plt.subplots(n_rows, 2, figsize=(6, 2.5 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, transition in enumerate(transitions[:n_rows]):
        obs_t_img = obs_to_image(transition["obs_t"])
        obs_next_img = obs_to_image(transition["obs_next"])

        axes[row, 0].imshow(obs_t_img)
        axes[row, 0].set_title("obs_t")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(obs_next_img)
        axes[row, 1].set_title("obs_{t+1}")
        axes[row, 1].axis("off")

        fig.text(
            0.5,
            1.0 - (row + 0.5) / n_rows,
            (
                f"step={transition['step']} action={transition['action']} "
                f"reward={transition['reward']:.3f} done={transition['done']} "
                f"error={transition['pred_error']:.6f}"
            ),
            ha="center",
            va="center",
            fontsize=9,
        )

    fig.suptitle("Top latent prediction-error transitions (obs_t | obs_{t+1})", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_craftax_env_from_name(args.env_name, auto_reset=True)
    env_params = env.default_params
    action_dim = env.action_space(env_params).n

    network = ActorCriticConvWorldModel(action_dim, layer_width=args.layer_size)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, rollout_rng = jax.random.split(rng, 3)

    init_obs = jnp.zeros((1, *env.observation_space(env_params).shape))
    init_action = jnp.zeros((1,), dtype=jnp.int32)
    init_next_obs = init_obs
    params = load_params_if_available(
        network,
        (init_rng, init_obs, init_action, init_next_obs),
        args.checkpoint_path,
    )

    obs, env_state = env.reset(rollout_rng, env_params)

    errors = []
    actions = []
    rewards = []
    dones = []
    transition_records = []

    rng = rollout_rng
    for step in range(args.num_steps):
        obs_t = obs
        rng, act_rng, step_rng = jax.random.split(rng, 3)

        pi, _, latent_t = network.apply(params, obs_t[None, ...])
        action = int(pi.sample(seed=act_rng))

        obs_next, env_state, reward, done, info = env.step(
            step_rng, env_state, action, env_params
        )

        _, _, next_latent = network.apply(params, obs_next[None, ...])
        pred_next_latent, _, _ = network.apply(
            params,
            latent_t,
            jnp.asarray([action], dtype=jnp.int32),
            method=network.world_model,
        )
        pred_error = float(
            jnp.mean(
                jnp.square(pred_next_latent - jax.lax.stop_gradient(next_latent))
            )
        )

        errors.append(pred_error)
        actions.append(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        transition_records.append(
            {
                "step": step,
                "obs_t": np.asarray(obs_t),
                "obs_next": np.asarray(obs_next),
                "action": action,
                "reward": float(reward),
                "done": bool(done),
                "pred_error": pred_error,
            }
        )

        obs = obs_next

    errors_arr = np.asarray(errors, dtype=np.float32)
    actions_arr = np.asarray(actions, dtype=np.int32)
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    dones_arr = np.asarray(dones, dtype=np.bool_)

    curve_path = output_dir / "prediction_error_curve.png"
    topk_path = output_dir / "top_prediction_error_transitions.png"
    npz_path = output_dir / "wm_errors.npz"

    save_prediction_error_curve(errors_arr, curve_path)

    top_indices = np.argsort(-errors_arr)[: args.topk]
    top_transitions = [transition_records[i] for i in top_indices]
    save_topk_transition_grid(top_transitions, args.topk, topk_path)

    np.savez(
        npz_path,
        errors=errors_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        dones=dones_arr,
    )

    print(f"Saved prediction error curve: {curve_path}")
    print(f"Saved top-{args.topk} transition grid: {topk_path}")
    print(f"Saved raw rollout data: {npz_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize latent world-model prediction quality for WM-PPO."
    )
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Pixels-v1")
    parser.add_argument("--num_steps", type=int, default=256)
    parser.add_argument("--topk", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="wm_visualizations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layer_size", type=int, default=256)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    cli_args = parser.parse_args()
    main(cli_args)
