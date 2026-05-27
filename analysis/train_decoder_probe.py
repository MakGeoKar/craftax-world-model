"""
Train a diagnostic latent->observation decoder on frozen WM-PPO encoder latents.

This script is separate from RL training. It does NOT update the world model,
encoder, actor, or critic. Latents are computed with stop-gradient; only the
decoder is optimized.

JAX/Flax equivalent of the PyTorch sanity-check pattern:
    z = stop_gradient(encode(obs))
    obs_hat = decoder(z)
    loss = L1(obs_hat, obs)
    grad(loss, decoder_params) only
"""

import argparse
import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
from craftax.craftax_env import make_craftax_env_from_name
from flax.serialization import from_bytes, to_bytes
from flax.training.train_state import TrainState

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.actor_critic import ActorCriticConvWorldModel
from models.latent_obs_decoder import LatentObsDecoder


def normalize_obs(obs: jnp.ndarray) -> jnp.ndarray:
    """Convert observations to float32 in [0, 1], keeping NHWC layout."""
    obs = jnp.asarray(obs, dtype=jnp.float32)
    if obs.ndim == 3:
        obs = obs[None, ...]
    if obs.max() > 1.0:
        obs = obs / 255.0
    return jnp.clip(obs, 0.0, 1.0)


def obs_to_hwc_image(obs: np.ndarray) -> np.ndarray:
    """Convert a single observation to HWC for matplotlib."""
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


def load_wm_params(network, init_args, checkpoint_path: str | None):
    rng, init_obs, init_action, init_next_obs = init_args
    params = network.init(
        rng, init_obs, init_action, init_next_obs, method=network.init_all
    )
    if checkpoint_path is None:
        print("WARNING: no checkpoint_path; using randomly initialized WM params.")
        return params
    with Path(checkpoint_path).open("rb") as f:
        params = from_bytes(params, f.read())
    print(f"Loaded WM params from {Path(checkpoint_path).resolve()}")
    return params


def collect_observations(env, env_params, rng, num_steps: int) -> jnp.ndarray:
    obs_list = []
    reset_rng, step_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)
    obs_list.append(np.asarray(obs))

    step_rng = rng
    for _ in range(num_steps - 1):
        step_rng, act_rng, env_rng = jax.random.split(step_rng, 3)
        action = int(jax.random.randint(act_rng, (), 0, env.action_space(env_params).n))
        obs, env_state, _, _, _ = env.step(env_rng, env_state, action, env_params)
        obs_list.append(np.asarray(obs))

    return jnp.asarray(np.stack(obs_list, axis=0))


def compute_metrics(obs, obs_hat):
    l1 = float(jnp.mean(jnp.abs(obs_hat - obs)))
    mse = float(jnp.mean(jnp.square(obs_hat - obs)))
    per_channel = None
    if obs.shape[-1] > 1:
        per_channel = [
            float(jnp.mean(jnp.abs(obs_hat[..., c] - obs[..., c])))
            for c in range(obs.shape[-1])
        ]
    return {"l1": l1, "mse": mse, "per_channel_l1": per_channel}


def make_train_step(decoder_apply, wm_network, wm_params, repr_alpha: float):
    wm_params_frozen = jax.lax.stop_gradient(wm_params)

    def train_step(decoder_state, batch_obs):
        obs = normalize_obs(batch_obs)

        def loss_fn(decoder_params):
            z = wm_network.apply(
                wm_params_frozen, obs, method=wm_network.encode
            )
            z = jax.lax.stop_gradient(z)
            obs_hat = decoder_apply(decoder_params, z)

            pixel_l1 = jnp.mean(jnp.abs(obs_hat - obs))
            loss = pixel_l1

            if repr_alpha > 0.0:
                z_hat = wm_network.apply(
                    wm_params_frozen, obs_hat, method=wm_network.encode
                )
                repr_loss = jnp.mean(jnp.square(z_hat - z))
                loss = pixel_l1 + repr_alpha * repr_loss

            return loss, (pixel_l1, obs_hat)

        (loss, (pixel_l1, obs_hat)), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(decoder_state.params)
        decoder_state = decoder_state.apply_gradients(grads=grads)
        metrics = compute_metrics(obs, obs_hat)
        metrics["loss"] = float(loss)
        metrics["pixel_l1"] = float(pixel_l1)
        return decoder_state, metrics

    return jax.jit(train_step)


def verify_wm_frozen(
    decoder_apply,
    wm_network,
    wm_params,
    decoder_params,
    sample_obs,
    repr_alpha: float = 0.0,
):
    """Sanity check: decoder loss must not produce WM parameter gradients."""
    obs = normalize_obs(sample_obs[: min(4, sample_obs.shape[0])])
    wm_params_frozen = jax.lax.stop_gradient(wm_params)

    def scalar_loss(dec_p):
        z = wm_network.apply(
            wm_params_frozen, obs, method=wm_network.encode
        )
        z = jax.lax.stop_gradient(z)
        obs_hat = decoder_apply(dec_p, z)
        pixel_l1 = jnp.mean(jnp.abs(obs_hat - obs))
        loss = pixel_l1
        if repr_alpha > 0.0:
            z_hat = wm_network.apply(
                wm_params_frozen, obs_hat, method=wm_network.encode
            )
            loss = pixel_l1 + repr_alpha * jnp.mean(jnp.square(z_hat - z))
        return loss

    def wm_only_loss(wp):
        frozen = jax.lax.stop_gradient(wp)
        z = wm_network.apply(frozen, obs, method=wm_network.encode)
        z = jax.lax.stop_gradient(z)
        obs_hat = decoder_apply(decoder_params, z)
        pixel_l1 = jnp.mean(jnp.abs(obs_hat - obs))
        loss = pixel_l1
        if repr_alpha > 0.0:
            z_hat = wm_network.apply(frozen, obs_hat, method=wm_network.encode)
            loss = pixel_l1 + repr_alpha * jnp.mean(jnp.square(z_hat - z))
        return loss

    wm_grads = jax.grad(wm_only_loss)(wm_params)
    decoder_grads = jax.grad(scalar_loss)(decoder_params)

    wm_grad_norm = float(
        jnp.sqrt(
            sum(jnp.sum(jnp.square(g)) for g in jax.tree_util.tree_leaves(wm_grads))
        )
    )
    dec_grad_norm = float(
        jnp.sqrt(
            sum(
                jnp.sum(jnp.square(g))
                for g in jax.tree_util.tree_leaves(decoder_grads)
            )
        )
    )

    print("Gradient sanity check:")
    print(f"  ||grad_wm|| = {wm_grad_norm:.6e} (expected ~0)")
    print(f"  ||grad_decoder|| = {dec_grad_norm:.6e} (expected >0 after training init)")
    if wm_grad_norm > 1e-9:
        raise RuntimeError(
            "WM/encoder received non-zero gradients; decoder probe is miswired."
        )
    print("  OK: only decoder parameters receive gradients.")


def save_reconstruction_grid(
    obs_batch,
    obs_hat_batch,
    output_path: Path,
    num_samples: int,
):
    obs_np = np.asarray(obs_batch)
    obs_hat_np = np.asarray(obs_hat_batch)
    n = min(num_samples, obs_np.shape[0])

    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(n):
        orig = obs_to_hwc_image(obs_np[row])
        recon = obs_to_hwc_image(obs_hat_np[row])
        err = np.mean(np.abs(recon.astype(np.float32) - orig.astype(np.float32)), axis=-1)

        axes[row, 0].imshow(orig)
        axes[row, 0].set_title("original")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(recon)
        axes[row, 1].set_title("reconstructed")
        axes[row, 1].axis("off")

        im = axes[row, 2].imshow(err, cmap="magma")
        axes[row, 2].set_title("abs error")
        axes[row, 2].axis("off")
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)

    fig.suptitle("Decoder probe: original | reconstruction | abs error", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = make_craftax_env_from_name(args.env_name, auto_reset=True)
    env_params = env.default_params
    obs_shape = tuple(env.observation_space(env_params).shape)
    action_dim = env.action_space(env_params).n

    wm_network = ActorCriticConvWorldModel(action_dim, layer_width=args.layer_size)
    decoder = LatentObsDecoder(layer_width=args.layer_size, obs_shape=obs_shape)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, collect_rng, train_rng = jax.random.split(rng, 4)

    init_obs = jnp.zeros((1, *obs_shape))
    init_action = jnp.zeros((1,), dtype=jnp.int32)
    wm_params = load_wm_params(
        wm_network,
        (init_rng, init_obs, init_action, init_obs),
        args.checkpoint_path,
    )
    wm_params = jax.lax.stop_gradient(wm_params)

    dummy_obs = normalize_obs(init_obs)
    dummy_z = wm_network.apply(wm_params, dummy_obs, method=wm_network.encode)
    print(f"Observation shape (NHWC): {obs_shape}")
    print(f"Latent shape: {tuple(dummy_z.shape)} (flat vector)")
    print(
        f"Observation value range in sample rollout will be normalized to [0, 1] "
        f"for decoder training."
    )

    decoder_params = decoder.init(train_rng, dummy_z)
    tx = optax.adam(args.lr)
    decoder_state = TrainState.create(
        apply_fn=decoder.apply, params=decoder_params, tx=tx
    )

    if args.run_grad_sanity_check:
        verify_wm_frozen(
            decoder.apply,
            wm_network,
            wm_params,
            decoder_params,
            dummy_obs,
            repr_alpha=args.repr_alpha,
        )

    print(f"Collecting {args.num_collect_steps} observations...")
    obs_dataset = collect_observations(
        env, env_params, collect_rng, args.num_collect_steps
    )
    obs_dataset = normalize_obs(obs_dataset)

    train_step = make_train_step(
        decoder.apply, wm_network, wm_params, args.repr_alpha
    )

    metrics_path = output_dir / "decoder_metrics.csv"
    with metrics_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["epoch", "loss", "l1", "mse", "per_channel_l1"],
        )
        writer.writeheader()

        for epoch in range(1, args.train_epochs + 1):
            perm = np.random.permutation(obs_dataset.shape[0])
            epoch_l1, epoch_mse, epoch_loss = [], [], []

            for start in range(0, obs_dataset.shape[0], args.batch_size):
                batch_idx = perm[start : start + args.batch_size]
                batch = obs_dataset[batch_idx]
                decoder_state, batch_metrics = train_step(decoder_state, batch)
                epoch_l1.append(batch_metrics["l1"])
                epoch_mse.append(batch_metrics["mse"])
                epoch_loss.append(batch_metrics["loss"])

            row = {
                "epoch": epoch,
                "loss": float(np.mean(epoch_loss)),
                "l1": float(np.mean(epoch_l1)),
                "mse": float(np.mean(epoch_mse)),
                "per_channel_l1": batch_metrics["per_channel_l1"],
            }
            writer.writerow(row)
            per_ch = row["per_channel_l1"]
            per_ch_str = f", per_channel_l1={per_ch}" if per_ch is not None else ""
            print(
                f"epoch {epoch:03d}: loss={row['loss']:.6f}, "
                f"l1={row['l1']:.6f}, mse={row['mse']:.6f}{per_ch_str}"
            )

    viz_obs = obs_dataset[: args.num_viz_samples]
    viz_z = wm_network.apply(wm_params, viz_obs, method=wm_network.encode)
    viz_z = jax.lax.stop_gradient(viz_z)
    viz_hat = decoder.apply(decoder_state.params, viz_z)

    grid_path = output_dir / "decoder_reconstructions.png"
    save_reconstruction_grid(viz_obs, viz_hat, grid_path, args.num_viz_samples)

    decoder_ckpt = output_dir / "decoder_params.msgpack"
    with decoder_ckpt.open("wb") as f:
        f.write(to_bytes(decoder_state.params))

    print(f"Saved metrics: {metrics_path.resolve()}")
    print(f"Saved visualizations: {grid_path.resolve()}")
    print(f"Saved decoder params: {decoder_ckpt.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a diagnostic latent->observation decoder probe."
    )
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Pixels-v1")
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_collect_steps", type=int, default=4096)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--repr_alpha",
        type=float,
        default=0.0,
        help="Optional representation consistency weight (0 disables).",
    )
    parser.add_argument("--num_viz_samples", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="outputs/decoder_probe")
    parser.add_argument(
        "--run_grad_sanity_check",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    cli_args = parser.parse_args()
    main(cli_args)
