# Experiments

Environment: Craftax-Classic-Pixels-v1
Budget: 1M environment steps unless stated otherwise
Metric: score, last update
Final branch: final-clean-wm-ppo
Ablation branch: backup-experimental-wm

# 1. Summary

I implemented a compact latent world-model PPO agent.

The world model predicts:
- next latent state
- reward
- done flag
- inverse action

The world model is trained on real Craftax rollouts. Its latent prediction error is used as an intrinsic reward for exploration.

The final method is not a full Dreamer-style imagination agent. It is a world-model-augmented PPO agent with model-based exploration.

# 2. Main result

PPO, averaged over seeds default, 42, 239:
- mean score last: 1.458953
- mean return last: 3.063586

WM-PPO, averaged over seeds default, 42, 239:
- mean score last: 1.557813
- mean return last: 3.276904

Average improvement over PPO:
- score: +0.098860, +6.78%
- return: +0.213318, +6.96%

# 3. Matched seed comparison

Seed: default

PPO:
- score last: 1.508824
- return last: 3.132766

WM-PPO:
- score last: 1.553774
- return last: 3.276494

Gain:
- score absolute gain: +0.044950
- score relative gain: +2.98%


Seed: 42

PPO:
- score last: 1.427461
- return last: 3.014872

WM-PPO:
- score last: 1.606709
- return last: 3.360699

Gain:
- score absolute gain: +0.179248
- score relative gain: +12.56%


Seed: 239

PPO:
- score last: 1.440574
- return last: 3.043119

WM-PPO:
- score last: 1.512956
- return last: 3.193519

Gain:
- score absolute gain: +0.072382
- score relative gain: +5.02%


Best observed WM-PPO run:
- score last: 1.637527
- return last: 3.360699

# 4. Final method

Final method:
Latent World-Model PPO with prediction-error exploration.

Architecture:

obs -> CNN encoder -> latent

latent -> actor head -> action distribution
latent -> critic head -> value

latent + action -> world model -> predicted next latent, predicted reward, predicted done

latent + next_latent -> inverse model -> predicted action

Training objective:

total_loss =
    PPO actor loss
  + critic value loss
  - entropy bonus
  + wm_coef * world_model_loss

World-model loss:

world_model_loss =
    forward dynamics loss
  + reward prediction loss
  + done prediction loss
  + inverse dynamics loss

Forward dynamics:
pred_next_latent should match encoder(next_obs)

Reward prediction:
pred_reward should match reward

Done prediction:
pred_done should match done

Inverse dynamics:
pred_action should match action

Intrinsic reward:

prediction_error = MSE(pred_next_latent, stopgrad(next_latent))

reward_total = reward_external + wm_intrinsic_coef * prediction_error

Final config:
- wm_coef: 1.0
- wm_intrinsic_coef: 0.003

# 5. Ablations

Experiment: PPO baseline
- budget: 1M
- config: default seed
- score last: 1.508824
- return last: 3.132766
- outcome: baseline

Experiment: WM-PPO final
- budget: 1M
- config: wm_coef=1.0, intrinsic=0.003
- score last: 1.553774
- return last: 3.276494
- outcome: beats PPO

Experiment: WM-PPO seed 42
- budget: 1M
- config: wm_coef=1.0, intrinsic=0.003
- score last: 1.606709
- return last: 3.360699
- outcome: beats PPO seed 42

Experiment: WM-PPO seed 239
- budget: 1M
- config: wm_coef=1.0, intrinsic=0.003
- score last: 1.512956
- return last: 3.193519
- outcome: beats PPO seed 239

Experiment: Best observed WM-PPO
- budget: 1M
- config: wm_coef=1.0, intrinsic=0.003
- score last: 1.637527
- return last: 3.360699
- outcome: best single run

Experiment: Imagined value loss
- budget: 1M
- config: critic consistency through world model
- score last: nan / worse
- return last: nan / worse
- outcome: unstable

Experiment: Stabilized imagined value loss
- budget: 1M
- config: clipped targets, Huber loss, warmup
- score last: 1.500796
- return last: 3.133178
- outcome: worse than final WM-PPO

Experiment: Info-based achievement gate
- budget: 1M
- config: curiosity gate through info["Achievements/..."]
- score last: 1.602106
- return last: 3.352864
- outcome: good run, but the gate was semantically broken

Experiment: Dynamics disagreement
- budget: 100k
- config: second dynamics head as uncertainty signal
- score last: 1.182202
- return last: 2.553448
- outcome: worse screening

Experiment: Tech-tree bonus 0.02
- budget: 100k
- config: achievement delta reward
- score last: 1.034118
- return last: 2.054167
- outcome: too strong

Experiment: Tech-tree bonus 0.01
- budget: 100k
- config: achievement delta reward
- score last: 1.428848
- return last: 3.110000
- outcome: strong but noisy

Experiment: Tech-tree bonus 0.01
- budget: 1M
- config: achievement delta reward
- score last: 1.547845
- return last: 3.265939
- outcome: did not beat final WM-PPO

Experiment: Tech-tree bonus 0.005
- budget: 100k
- config: achievement delta reward
- score last: 1.220486
- return last: 2.562264
- outcome: too weak

Experiment: Normalized curiosity
- budget: 100k
- config: pred_error / mean(pred_error)
- score last: 1.080174
- return last: 2.283333
- outcome: intrinsic reward became too dominant

Experiment: Annealed curiosity
- budget: 1M
- config: intrinsic coefficient from 0.006 to 0.001
- score last: 1.572876
- return last: 3.294651
- outcome: better than PPO, worse than constant intrinsic

Experiment: Policy consistency
- budget: 1M
- config: imagined latent actor-logit consistency
- score last: 1.530997
- return last: 3.194606
- outcome: worse than final WM-PPO

Experiment: Imagined all-actions actor loss
- budget: 1M
- config: one-step model-based actor improvement
- score last: 1.533934
- return last: 3.282843
- outcome: worse than final WM-PPO

Experiment: CPC latent dynamics
- budget: 100k
- config: contrastive next-latent loss, coefficient 0.1
- score last: 1.127549
- return last: 2.317647
- outcome: worse screening

# 6. Notes on failed ideas

## Info-based achievement gate

I initially tried to gate curiosity using info["Achievements/..."].

Later inspection showed that Craftax reports these values only at episode end:

info["Achievements/x"] = state.achievements[x] * done * 100

Since prediction error was masked by (1 - done), this gate almost never affected non-terminal transitions.

I replaced it with a proper delta-achievement version, but it did not improve the 1M result.

## Imagined actor and value losses

I tested more explicit model-based objectives:
- one-step imagined value learning
- one-step all-actions model-based actor improvement

Both underperformed the final WM-PPO.

Likely reason:
the learned one-step model Q estimates were biased enough for actor/critic optimization to exploit model errors.

## CPC latent dynamics

I added a contrastive dynamics loss where the predicted next latent had to identify the true next latent among negatives from the minibatch.

It worsened 100k screening performance, so I did not scale it to 1M.

# 7. Conclusions

1. WM-PPO beat PPO on all three matched seeds.
2. The average score improvement over PPO was +6.78%.
3. The strongest component was prediction-error intrinsic reward from the latent world model.
4. Auxiliary world-model losses helped shape a dynamics-aware representation.
5. More aggressive model-based actor/critic objectives were less stable and did not improve the 1M score.
6. The final solution is intentionally minimal: PPO + latent dynamics/reward/done/inverse prediction + prediction-error exploration.