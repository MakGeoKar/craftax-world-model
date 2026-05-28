# WM-PPO Code Walkthrough

A reading guide for the final world-model-augmented PPO agent. This document is
descriptive only; it does not change any behavior, architecture, or losses.

Key files:

- `models/actor_critic.py` — `ActorCriticConvWorldModel` (encoder + heads).
- `ppo_world_model.py` — training loop (rollout, GAE, PPO + WM losses, saving).
- `models/latent_obs_decoder.py` — diagnostic latent→pixel decoder (post-hoc).
- `analysis/train_decoder_probe.py` — trains the decoder on frozen latents.
- `analysis/final_decoder_visualizations.py` — report-ready figures.

---

## 1. `ActorCriticConvWorldModel`

One shared CNN encoder feeds four heads. All heads operate on a flat latent; the
world model never decodes back to pixels.

```
obs [B, H, W, C]
   │  encode()  (3× Conv5x5 + ReLU + max_pool3x3, flatten, Dense, ReLU)
   ▼
latent [B, layer_width]
   ├─ actor head   (actor_fc1 → actor_fc2 → actor_out)      → pi (Categorical)
   ├─ critic head  (critic_fc1 → critic_out)                → value (scalar)
   ├─ world_model(latent, action)                           → (pred_next_latent,
   │     (concat one-hot action; wm_fc1/wm_fc2; 3 output heads)   pred_reward,
   │                                                              pred_done_logit)
   └─ inverse_model(latent, next_latent)                    → action logits
         (concat; inv_fc1 → inv_action)
```

Methods:

| Method | Signature | Purpose |
|--------|-----------|---------|
| `encode` | `obs → latent` | CNN encoder |
| `__call__` | `obs → (pi, value, latent)` | PPO forward pass |
| `world_model` | `(latent, action) → (next_latent, reward, done_logit)` | latent dynamics |
| `inverse_model` | `(latent, next_latent) → action_logits` | inverse dynamics |
| `init_all` | `(obs, action, next_obs) → (pi, value, latent)` | touches all heads for `init` |

Call non-`__call__` heads via Flax method dispatch, e.g.
`network.apply(params, latent, action, method=network.world_model)`.

> Flax layer `name=` strings (`enc_conv1`, `actor_fc1`, `wm_fc1`, …) are part of
> the saved parameter tree. Renaming them breaks existing checkpoints.

Symbolic envs use the separate plain `ActorCritic` (no world model).

---

## 2. `ppo_world_model.py` flow

### Config setup (`make_train`)
Derives `NUM_UPDATES` and `MINIBATCH_SIZE` from timestep/env/step counts.

### Env wrappers
`make_craftax_env_from_name` → `LogWrapper` → either
`OptimisticResetVecEnvWrapper` (default) or `AutoResetEnvWrapper` + `BatchEnvWrapper`.

### Network + optimizer init
Pixel envs build `ActorCriticConvWorldModel` and init with `method=init_all`;
symbolic envs build `ActorCritic`. Optimizer is Adam with global-norm clipping
and optional linear LR anneal. State held in a Flax `TrainState`.

### `Transition`
Per-step rollout record. Important fields:

- `reward_e` — external (environment) reward.
- `reward_i` — intrinsic WM bonus (or ICM/E3B when enabled).
- `reward` — `reward_e + reward_i`; **this is what GAE/PPO optimize**.
- `info` — env-reported metrics (used for reporting only).

### Rollout collection (`_env_step`, scanned `NUM_STEPS` times)
1. **Action sampling:** `pi, value = policy_apply(params, obs)`, sample `action`,
   record `log_prob`.
2. **Env step:** get `next_obs, reward_e, done, info`.
3. **WM intrinsic reward** (pixel only):
   `pred_error = mean((world_model(encode(obs), action) − stopgrad(encode(next_obs)))²)`,
   masked to 0 on terminal steps, scaled by `WM_INTRINSIC_COEF`. Higher
   prediction error ⇒ larger exploration bonus.
4. `reward = reward_e + reward_i`.

### GAE
Standard GAE(λ) computed backward over the rollout using `transition.reward`
(external + intrinsic). Produces `advantages` and value `targets`.

### PPO + WM update (`_loss_fn`, scanned over epochs × minibatches)
Per minibatch, re-run the network on stored obs and compute:

- **Clipped value loss:** max of squared error vs. clipped-prediction squared
  error (`VF_COEF`).
- **Clipped actor loss:** PPO surrogate `min(ratio·Â, clip(ratio)·Â)` with
  normalized advantages.
- **Entropy bonus:** `−ENT_COEF · H[pi]`.
- **WM auxiliary losses** (pixel only, terminal steps masked):
  - forward: MSE on predicted next latent vs. `stopgrad(encode(next_obs))`,
  - reward: MSE on predicted external reward,
  - done: BCE on terminal flag,
  - inverse: cross-entropy recovering the action.
  Combined into `wm_loss` with `WM_*_COEF` weights.
- **Total:** `loss_actor + VF_COEF·value_loss − ENT_COEF·entropy + WM_COEF·wm_loss`.

The `stop_gradient` on the next-latent target means WM/inverse losses train the
WM heads (and encoder through `latent`) without pulling the target encoder.

### Saving trained params
With `--save_params_path`, the final params of the first repeat are pulled to
host and written as a flax `to_bytes` msgpack blob, consumed by the decoder /
visualization scripts. The wandb `--save_policy` path additionally writes an
Orbax checkpoint.

---

## 3. External reward vs. reported score

- **Training signal:** `reward = reward_e + reward_i`. The intrinsic WM bonus
  shapes exploration and is part of the GAE/PPO objective.
- **Reported metrics:** episode-averaged env `info` (e.g.
  `returned_episode_returns`, score). These are **external reward only** — the
  intrinsic bonus never enters the reported score. So a reported return is a
  clean measure of task performance, not inflated by curiosity.

---

## 4. Decoder diagnostics (post-hoc, no effect on RL)

The latent world model is not generative; to *inspect* what a latent encodes we
train a small decoder **after** RL:

- `models/latent_obs_decoder.py` — flat latent → reconstructed observation.
- `analysis/train_decoder_probe.py` — loads frozen WM params, computes
  `z = stopgrad(encode(obs))`, and trains only the decoder (L1 reconstruction,
  optional representation-consistency term). The encoder / WM / actor / critic
  are never updated here.
- `analysis/final_decoder_visualizations.py` — produces reconstruction grids,
  error histograms, and one-step WM prediction visualizations.

These scripts are purely diagnostic: they consume the saved checkpoint and do
not modify RL training or its objective.
