# Craftax-Classic Pixels Model-Based RL Experiments

## Setup

Environment: `Craftax-Classic-Pixels-v1`  
Budget: `1M environment steps`  
Observation type: pixels  
Training hardware: Kaggle GPU  
Base code: official Craftax PPO baseline  
Evaluation metric: original Craftax external score / achievements, not intrinsic reward  

Common config:

```bash
--total_timesteps 1000000
--num_envs 512
--num_steps 64
--num_minibatches 32
--update_epochs 4
--layer_size 256
--no-use_wandb
--no-debug
```

---

## Methods

### PPO baseline

Standard PPO with convolutional actor-critic.

```text
pixel obs -> CNN encoder -> actor head + critic head
```

The actor predicts a categorical action distribution.  
The critic predicts state value `V(s)`.

### WM-PPO

PPO with a shared latent world model.

```text
pixel obs -> CNN encoder -> shared latent

shared latent -> actor
shared latent -> critic

latent_t + action_t -> predicted latent_{t+1}
latent_t + action_t -> predicted reward
latent_t + action_t -> predicted done

latent_t + latent_{t+1} -> predicted action
```

Training objective:

```text
PPO loss
+ forward dynamics loss
+ reward prediction loss
+ done prediction loss
+ inverse dynamics loss
```

Exploration bonus:

```text
training_reward = external_reward + wm_intrinsic_coef * prediction_error
```

Evaluation uses only original Craftax external score.

---

## Results

| Run | Method | wm_coef | wm_intrinsic_coef | Score mean | Score last | Return mean | Return last | Time | SPS | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | PPO baseline | - | - | 1.291644 | 1.508824 | 2.673508 | 3.132766 | 415.32s | 2407.77 | Baseline |
| 1 | WM-PPO | 1.0 | 0.01 | 1.269974 | 1.547591 | 2.638449 | 3.240175 | 513.78s | 1946.37 | Slightly better final score/return |
| 2 | WM-PPO | 0.3 | 0.01 | TBD | TBD | TBD | TBD | TBD | TBD | Weaker WM auxiliary loss |
| 3 | WM-PPO | 1.0 | 0.003 | TBD | TBD | TBD | TBD | TBD | TBD | Weaker intrinsic reward |

---

## Key achievements comparison

### PPO baseline, last update

```text
collect_wood: 66.808510
place_table: 28.085106
make_wood_pickaxe: 3.404255
make_wood_sword: 2.978723
collect_stone: 1.276596
collect_drink: 18.723404
defeat_zombie: 7.234042
score: 1.508824
return: 3.132766
```

### WM-PPO, wm_coef=1.0, intrinsic=0.01, last update

```text
collect_wood: 69.868996
place_table: 31.877729
make_wood_pickaxe: 3.056769
make_wood_sword: 4.366812
collect_stone: 0.000000
collect_drink: 23.580786
defeat_zombie: 5.240175
score: 1.547591
return: 3.240175
```

---

## Preliminary observations

WM-PPO slightly improves final score and return over PPO under the same 1M environment-step budget.

The improvement appears mostly in early-game exploration and crafting-related achievements:

```text
place_table
collect_drink
make_wood_sword
collect_wood
```

However, deeper tech-tree achievements are still mostly absent:

```text
collect_iron
collect_diamond
make_iron_pickaxe
make_stone_pickaxe
```

This suggests that the current world model helps early exploration but does not yet solve long-horizon planning.

---

## Failure modes / limitations

1. The world model is currently used for representation learning and curiosity exploration, not for full imagined rollout training.
2. Pixel observations make the auxiliary world-model loss memory-heavy, so PPO minibatch size had to be reduced.
3. Intrinsic reward may distract the agent from external reward if `wm_intrinsic_coef` is too high.
4. Long-horizon achievements likely require stronger planning or recurrent memory.
