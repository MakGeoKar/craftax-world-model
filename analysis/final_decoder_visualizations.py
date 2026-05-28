"""
Report-ready visualizations for the WM decoder probe.

Standalone script — does not modify or run RL/decoder training.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import jax
import jax.tree_util

# Compatibility patch for newer JAX versions (gymnax/craftax on Kaggle).
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree_util.tree_map

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name
from flax.serialization import from_bytes

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.actor_critic import ActorCriticConvWorldModel
from models.latent_obs_decoder import LatentObsDecoder

TOP_K = 7
INFER_BATCH_SIZE = 64


def normalize_obs(obs: jnp.ndarray) -> jnp.ndarray:
    """Normalize Craftax pixel observations to [0, 1] in a JIT-safe way."""
    if jnp.issubdtype(obs.dtype, jnp.integer):
        return obs.astype(jnp.float32) / 255.0
    return jnp.clip(obs.astype(jnp.float32), 0.0, 1.0)


def obs_to_hwc_image(obs: np.ndarray) -> np.ndarray:
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


def abs_error_map(orig: np.ndarray, other: np.ndarray) -> np.ndarray:
    a = obs_to_hwc_image(orig).astype(np.float32)
    b = obs_to_hwc_image(other).astype(np.float32)
    return np.mean(np.abs(b - a), axis=-1)


def load_wm_params(network, init_args, checkpoint_path: str):
    rng, init_obs, init_action, init_next_obs = init_args
    params = network.init(
        rng, init_obs, init_action, init_next_obs, method=network.init_all
    )
    with Path(checkpoint_path).open("rb") as f:
        params = from_bytes(params, f.read())
    print(f"Loaded WM params from {Path(checkpoint_path).resolve()}")
    return params


def load_decoder_params(decoder, init_rng, dummy_z, checkpoint_path: str):
    params = decoder.init(init_rng, dummy_z)
    with Path(checkpoint_path).open("rb") as f:
        params = from_bytes(params, f.read())
    print(f"Loaded decoder params from {Path(checkpoint_path).resolve()}")
    return params


def collect_observations(env, env_params, rng, num_steps: int) -> jnp.ndarray:
    obs_list = []
    reset_rng, step_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)
    obs_list.append(np.asarray(obs))

    for _ in range(num_steps - 1):
        step_rng, act_rng, env_rng = jax.random.split(step_rng, 3)
        action = int(jax.random.randint(act_rng, (), 0, env.action_space(env_params).n))
        obs, env_state, _, _, _ = env.step(env_rng, env_state, action, env_params)
        obs_list.append(np.asarray(obs))

    return jnp.asarray(np.stack(obs_list, axis=0))


def collect_transitions(env, env_params, rng, num_steps: int):
    obs_t_list, action_list, obs_next_list = [], [], []
    reset_rng, step_rng = jax.random.split(rng)
    obs, env_state = env.reset(reset_rng, env_params)

    for _ in range(num_steps):
        step_rng, act_rng, env_rng = jax.random.split(step_rng, 3)
        action = int(jax.random.randint(act_rng, (), 0, env.action_space(env_params).n))
        obs_next, env_state, _, _, _ = env.step(env_rng, env_state, action, env_params)
        obs_t_list.append(np.asarray(obs))
        action_list.append(action)
        obs_next_list.append(np.asarray(obs_next))
        obs = obs_next

    return (
        jnp.asarray(np.stack(obs_t_list, axis=0)),
        jnp.asarray(action_list, dtype=jnp.int32),
        jnp.asarray(np.stack(obs_next_list, axis=0)),
    )


def batched_reconstruct(wm_network, decoder, wm_params, decoder_params, obs):
    obs_norm = normalize_obs(obs)
    hats = []
    for start in range(0, obs_norm.shape[0], INFER_BATCH_SIZE):
        batch = obs_norm[start : start + INFER_BATCH_SIZE]
        z = wm_network.apply(wm_params, batch, method=wm_network.encode)
        hats.append(decoder.apply(decoder_params, z))
    return jnp.concatenate(hats, axis=0)


def per_sample_l1(obs_norm: jnp.ndarray, obs_hat: jnp.ndarray) -> np.ndarray:
    return np.asarray(
        jax.device_get(jnp.mean(jnp.abs(obs_hat - obs_norm), axis=(1, 2, 3)))
    )


def shuffled_indices(num_items: int, k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_items)
    return perm[: min(k, num_items)]


def save_three_panel_grid(
    obs_norm,
    obs_hat,
    indices,
    l1_values,
    output_path: Path,
    suptitle: str,
    index_labels=None,
):
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(9, 2.3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    obs_np = jax.device_get(obs_norm)
    hat_np = jax.device_get(obs_hat)

    for row, idx in enumerate(indices):
        orig = obs_to_hwc_image(obs_np[idx])
        recon = obs_to_hwc_image(hat_np[idx])
        err = abs_error_map(orig, recon)
        l1 = l1_values[idx]

        label = ""
        if index_labels is not None:
            label = f"idx={index_labels[row]}  "
        row_title = f"{label}L1={l1:.5f}"

        axes[row, 0].imshow(orig)
        axes[row, 0].set_title("original")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(recon)
        axes[row, 1].set_title("reconstruction")
        axes[row, 1].axis("off")

        im = axes[row, 2].imshow(err, cmap="magma")
        axes[row, 2].set_title("abs error")
        axes[row, 2].axis("off")
        fig.colorbar(im, ax=axes[row, 2], fraction=0.046, pad=0.04)
        axes[row, 0].text(
            0.5,
            1.08,
            row_title,
            transform=axes[row, 0].transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(suptitle, y=1.01, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_four_panel_grid(
    obs_t_norm,
    obs_next_norm,
    pred_next,
    indices,
    l1_values,
    actions,
    output_path: Path,
    suptitle: str,
):
    n = len(indices)
    fig, axes = plt.subplots(n, 4, figsize=(11, 2.3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    obs_t_np = jax.device_get(obs_t_norm)
    obs_next_np = jax.device_get(obs_next_norm)
    pred_np = jax.device_get(pred_next)
    actions_np = np.asarray(jax.device_get(actions))

    for row, idx in enumerate(indices):
        ot = obs_to_hwc_image(obs_t_np[idx])
        on = obs_to_hwc_image(obs_next_np[idx])
        pn = obs_to_hwc_image(pred_np[idx])
        err = abs_error_map(on, pn)
        l1 = l1_values[idx]
        action = int(actions_np[idx])

        row_title = f"idx={idx}  action={action}  L1={l1:.5f}"

        axes[row, 0].imshow(ot)
        axes[row, 0].set_title("obs_t")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(on)
        axes[row, 1].set_title("real obs_{t+1}")
        axes[row, 1].axis("off")

        axes[row, 2].imshow(pn)
        axes[row, 2].set_title("predicted obs_{t+1}")
        axes[row, 2].axis("off")

        im = axes[row, 3].imshow(err, cmap="magma")
        axes[row, 3].set_title("abs error")
        axes[row, 3].axis("off")
        fig.colorbar(im, ax=axes[row, 3], fraction=0.046, pad=0.04)
        axes[row, 0].text(
            0.5,
            1.08,
            row_title,
            transform=axes[row, 0].transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.suptitle(suptitle, y=1.01, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_training_curve(metrics_path: Path, output_path: Path):
    epochs, l1s, mses = [], [], []
    with metrics_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            l1s.append(float(row["l1"]))
            mses.append(float(row["mse"]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    axes[0].plot(epochs, l1s, marker="o", linewidth=1.5, markersize=3)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("L1")
    axes[0].set_title("Decoder L1 vs epoch")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, mses, marker="o", linewidth=1.5, markersize=3, color="tab:orange")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE")
    axes[1].set_title("Decoder MSE vs epoch")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Decoder training curve", fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def save_histogram(values, output_path: Path, title: str, xlabel: str):
    values = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(values))
    median = float(np.median(values))

    plt.figure(figsize=(6, 3.5))
    plt.hist(values, bins=40, color="steelblue", alpha=0.85, edgecolor="white")
    plt.axvline(mean, color="tab:red", linestyle="--", linewidth=1.5, label=f"mean={mean:.5f}")
    plt.axvline(median, color="tab:green", linestyle=":", linewidth=1.5, label=f"median={median:.5f}")
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def batched_wm_one_step_predict(
    wm_network, decoder, wm_params, decoder_params, obs_t, actions, obs_next
):
    obs_t_norm = normalize_obs(obs_t)
    obs_next_norm = normalize_obs(obs_next)
    pred_next_list = []

    for start in range(0, obs_t_norm.shape[0], INFER_BATCH_SIZE):
        batch_t = obs_t_norm[start : start + INFER_BATCH_SIZE]
        batch_a = actions[start : start + INFER_BATCH_SIZE]
        z_t = wm_network.apply(wm_params, batch_t, method=wm_network.encode)
        pred_next_latent, _, _ = wm_network.apply(
            wm_params,
            z_t,
            batch_a,
            method=wm_network.world_model,
        )
        pred_next_list.append(decoder.apply(decoder_params, pred_next_latent))

    pred_next = jnp.concatenate(pred_next_list, axis=0)
    l1 = per_sample_l1(obs_next_norm, pred_next)
    return obs_t_norm, obs_next_norm, pred_next, l1


def distribution_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    env = make_craftax_env_from_name(args.env_name, auto_reset=True)
    env_params = env.default_params
    obs_shape = tuple(env.observation_space(env_params).shape)
    action_dim = env.action_space(env_params).n

    wm_network = ActorCriticConvWorldModel(action_dim, layer_width=args.layer_size)
    decoder = LatentObsDecoder(layer_width=args.layer_size, obs_shape=obs_shape)

    rng = jax.random.PRNGKey(args.seed)
    rng, init_rng, collect_rng, trans_rng, dec_init_rng = jax.random.split(rng, 5)

    init_obs = jnp.zeros((1, *obs_shape))
    init_action = jnp.zeros((1,), dtype=jnp.int32)
    wm_params = load_wm_params(
        wm_network,
        (init_rng, init_obs, init_action, init_obs),
        args.wm_checkpoint_path,
    )

    dummy_z = wm_network.apply(
        wm_params, normalize_obs(init_obs), method=wm_network.encode
    )
    decoder_params = load_decoder_params(
        decoder, dec_init_rng, dummy_z, args.decoder_checkpoint_path
    )

    print(f"Collecting {args.num_samples} observations...")
    obs_raw = collect_observations(env, env_params, collect_rng, args.num_samples)
    print(
        "Collected obs raw range:",
        float(jnp.min(obs_raw)),
        float(jnp.max(obs_raw)),
        obs_raw.dtype,
    )
    obs_norm = normalize_obs(obs_raw)
    obs_hat = batched_reconstruct(
        wm_network, decoder, wm_params, decoder_params, obs_raw
    )
    recon_l1 = per_sample_l1(obs_norm, obs_hat)

    random_idx = shuffled_indices(len(recon_l1), TOP_K, args.seed)
    path = output_dir / "random_reconstructions.png"
    save_three_panel_grid(
        obs_norm,
        obs_hat,
        random_idx,
        recon_l1,
        path,
        "Random reconstructions (decoder(encoder(obs)))",
        index_labels=random_idx,
    )
    saved_paths.append(path)

    best_idx = np.argsort(recon_l1)[:TOP_K]
    path = output_dir / "best_reconstructions_top7.png"
    save_three_panel_grid(
        obs_norm,
        obs_hat,
        best_idx,
        recon_l1,
        path,
        "Best reconstructions (lowest L1)",
        index_labels=best_idx,
    )
    saved_paths.append(path)

    worst_idx = np.argsort(-recon_l1)[:TOP_K]
    path = output_dir / "worst_reconstructions_top7.png"
    save_three_panel_grid(
        obs_norm,
        obs_hat,
        worst_idx,
        recon_l1,
        path,
        "Worst reconstructions (highest L1)",
        index_labels=worst_idx,
    )
    saved_paths.append(path)

    path = output_dir / "reconstruction_error_histogram.png"
    save_histogram(
        recon_l1,
        path,
        "Decoder reconstruction error distribution",
        "per-sample L1",
    )
    saved_paths.append(path)

    metrics_path = Path(args.decoder_metrics_path)
    if metrics_path.is_file():
        path = output_dir / "decoder_training_curve.png"
        save_training_curve(metrics_path, path)
        saved_paths.append(path)
    else:
        print(f"WARNING: decoder metrics not found at {metrics_path}; skipping training curve.")

    wm_l1 = None
    if not args.skip_wm_prediction:
        try:
            print(f"Collecting {args.num_transition_samples} transitions for WM one-step viz...")
            obs_t_raw, actions, obs_next_raw = collect_transitions(
                env, env_params, trans_rng, args.num_transition_samples
            )
            obs_t_norm, obs_next_norm, pred_next, wm_l1 = batched_wm_one_step_predict(
                wm_network,
                decoder,
                wm_params,
                decoder_params,
                obs_t_raw,
                actions,
                obs_next_raw,
            )

            random_trans_idx = shuffled_indices(len(wm_l1), TOP_K, args.seed + 1)
            path = output_dir / "one_step_wm_prediction_random.png"
            save_four_panel_grid(
                obs_t_norm,
                obs_next_norm,
                pred_next,
                random_trans_idx,
                wm_l1,
                actions,
                path,
                "One-step WM prediction (random transitions)",
            )
            saved_paths.append(path)

            miss_idx = np.argsort(-wm_l1)[:TOP_K]
            path = output_dir / "one_step_wm_prediction_top7_misses.png"
            save_four_panel_grid(
                obs_t_norm,
                obs_next_norm,
                pred_next,
                miss_idx,
                wm_l1,
                actions,
                path,
                "One-step WM prediction (top-7 misses)",
            )
            saved_paths.append(path)

            path = output_dir / "wm_prediction_error_histogram.png"
            save_histogram(
                wm_l1,
                path,
                "One-step WM prediction error distribution",
                "per-transition L1",
            )
            saved_paths.append(path)
        except Exception as exc:
            print(f"ERROR: WM one-step prediction visualization failed: {exc}")
            print("Reconstruction plots were still saved. Re-run with --skip_wm_prediction to suppress this block.")
            wm_l1 = None
    else:
        print("Skipping WM one-step prediction visualizations (--skip_wm_prediction).")

    recon_stats = distribution_stats(recon_l1)
    summary = {
        "reconstruction_l1_mean": recon_stats["mean"],
        "reconstruction_l1_median": recon_stats["median"],
        "reconstruction_l1_std": recon_stats["std"],
        "reconstruction_l1_min": recon_stats["min"],
        "reconstruction_l1_max": recon_stats["max"],
        "wm_one_step_l1_mean": None,
        "wm_one_step_l1_median": None,
        "wm_one_step_l1_std": None,
        "wm_one_step_l1_min": None,
        "wm_one_step_l1_max": None,
        "num_samples": int(args.num_samples),
        "num_transition_samples": int(args.num_transition_samples),
    }
    if wm_l1 is not None:
        wm_stats = distribution_stats(wm_l1)
        summary.update(
            {
                "wm_one_step_l1_mean": wm_stats["mean"],
                "wm_one_step_l1_median": wm_stats["median"],
                "wm_one_step_l1_std": wm_stats["std"],
                "wm_one_step_l1_min": wm_stats["min"],
                "wm_one_step_l1_max": wm_stats["max"],
            }
        )

    summary_path = output_dir / "summary_metrics.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    saved_paths.append(summary_path)

    print("\nSaved files:")
    for p in saved_paths:
        print(f"  {p.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate report-ready decoder probe visualizations."
    )
    parser.add_argument(
        "--wm_checkpoint_path",
        type=str,
        default="outputs/wm_params_seed42.msgpack",
    )
    parser.add_argument(
        "--decoder_checkpoint_path",
        type=str,
        default="outputs/decoder_probe_seed42_normfix/decoder_params.msgpack",
    )
    parser.add_argument(
        "--decoder_metrics_path",
        type=str,
        default="outputs/decoder_probe_seed42_normfix/decoder_metrics.csv",
    )
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Pixels-v1")
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=2048)
    parser.add_argument("--num_transition_samples", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default="outputs/final_visualizations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip_wm_prediction",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    cli_args = parser.parse_args()
    main(cli_args)
