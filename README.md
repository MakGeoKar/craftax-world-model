Craftax World Model PPO

A compact research project exploring whether a lightweight action-conditioned world model can improve PPO in Craftax-Classic from pixel observations.

This project starts from the original Craftax baseline implementation and adds a latent world-model module to the PPO agent. The final method is not a full Dreamer reproduction. Instead, it keeps the PPO training loop and augments it with latent dynamics prediction, reward prediction, done prediction, inverse dynamics, and prediction-error exploration.

Motivation

Modern model-based RL agents use learned world models to reason about future trajectories. In this project I studied a smaller and more controlled question:

Can a simple latent world model improve a PPO agent in Craftax-Classic without replacing the whole RL pipeline?

The goal was to build a minimal world-model-augmented PPO system, test several model-based ideas, and understand which parts actually help under a small compute budget.

Environment

Environment: Craftax-Classic-Pixels-v1

Observation type: RGB pixels

Action space: discrete

Training budget: 1M environment steps

Evaluation seeds: 0, 42, 239

Main metric: final logged score and episode return

Method

The baseline is a convolutional PPO actor-critic.

The final WM-PPO agent uses a shared CNN encoder. The encoder maps pixel observations into a latent state. The actor and critic operate on this latent state. The world model receives the current latent state and the executed action, then predicts the next latent state, reward, and done flag. An inverse dynamics head predicts the action from the current and next latent states.

Architecture:

obs -> CNN encoder -> latent

latent -> actor head -> action distribution

latent -> critic head -> value

latent + action -> world model -> predicted next latent, reward, done

latent + next latent -> inverse model -> predicted action

The world model is trained only on real environment rollouts. Its one-step latent prediction error is also used as an intrinsic exploration bonus.

Training reward:

reward_total = reward_external + wm_intrinsic_coef * prediction_error

The reported score and episode return are still external environment metrics, so the final evaluation is not inflated by the intrinsic reward.

Results

PPO baseline over 3 matched seeds:

score last: 1.458953

return last: 3.063586

WM-PPO over 3 matched seeds:

score last: 1.557813

return last: 3.276904

Improvement:

score: +0.098860, +6.78%

return: +0.213318, +6.96%

Best observed WM-PPO run:

score last: 1.637527

return last: 3.360699

WM-PPO improved the PPO baseline on all three matched seeds. The strongest component was prediction-error exploration from the latent world model.

Ablations

I tested several additional model-based ideas.

Imagined value loss: unstable or worse than final WM-PPO.

Stabilized imagined value loss with clipping, Huber loss, and warmup: stable but worse than final WM-PPO.

Policy consistency on imagined latents: worse than final WM-PPO.

Imagined all-actions actor loss: worse than final WM-PPO.

CPC latent dynamics loss: worse in 100k screening.

Tech-tree bonus: sometimes promising at short budget, but did not beat final WM-PPO at 1M steps.

Annealed curiosity: better than PPO, but worse than constant intrinsic coefficient.

Conclusion: the simple WM-PPO setup with prediction-error exploration was stronger than more aggressive imagined actor-critic objectives in this implementation.

Key Findings

A lightweight latent world model can be integrated into PPO with a small architectural change.

Prediction-error intrinsic reward improved exploration and produced a consistent gain over PPO across three matched seeds.

More direct model-based actor and critic objectives were less stable and did not improve the final score.

The final method mostly improves early-game exploration and survival behavior.

Deeper progression such as iron and diamond collection remains unsolved.

Important Files

ppo.py — original PPO baseline.

ppo_world_model.py — final world-model-augmented PPO training loop.

ppo_rnn.py — recurrent PPO baseline.

models/actor_critic.py — actor-critic and world-model architectures.

models/latent_obs_decoder.py — diagnostic latent-to-pixel decoder.

analysis/train_decoder_probe.py — trains the decoder probe.

analysis/final_decoder_visualizations.py — produces decoder visualizations.

analysis/visualize_world_model.py — world-model rollout diagnostics.

experiments.md — full experiment log and ablations.

CODE_WALKTHROUGH.md — detailed implementation walkthrough.

Running

Install dependencies:

git clone https://github.com/MakGeoKar/craftax-world-model.git

cd craftax-world-model

pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

Run PPO baseline:

python ppo.py –env_name Craftax-Classic-Pixels-v1

Run WM-PPO:

python ppo_world_model.py –env_name Craftax-Classic-Pixels-v1 –wm_coef 1.0 –wm_intrinsic_coef 0.003

Run WM-PPO and save parameters:

python ppo_world_model.py –env_name Craftax-Classic-Pixels-v1 –wm_coef 1.0 –wm_intrinsic_coef 0.003 –save_params_path outputs/wm_params_seed42.msgpack

Run world-model visualization:

python analysis/visualize_world_model.py –env_name Craftax-Classic-Pixels-v1 –layer_size 256 –checkpoint_path outputs/wm_params_seed42.msgpack –num_steps 512 –max_horizon 4 –output_dir results/wm_viz_seed42

Limitations

This is not a full Dreamer-style agent. The policy is not primarily trained inside imagined latent rollouts. The project focuses on a minimal PPO-compatible world-model augmentation.

The world model is useful for exploration, but the imagined actor and critic objectives tested here did not improve performance.

The agent still does not reliably reach deeper Craftax progression such as iron or diamond.

Future Work

Add recurrent memory for partial observability.

Improve world-model action sensitivity.

Use longer warmup before imagined actor-critic updates.

Evaluate longer training budgets and more seeds.

Compare against stronger recurrent Craftax baselines.

Acknowledgements

This repository builds on the original Craftax baseline implementation by Michael T. Matthews.