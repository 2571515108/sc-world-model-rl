# SC2 World Model League Agent

An **opponent-conditioned hierarchical world-model reinforcement learning** prototype for partially observable real-time strategy games. It keeps AlphaStar-style structured state, macro actions, and League mechanisms, but does not claim to reproduce full AlphaStar. The world model operates only at macro decision intervals.

The project implements a runnable synthetic-environment research loop for Phases 0–5. Real SC2 is connected through an adapter and is never required for synthetic tests.

## Capabilities by Phase

| Phase | Implementation | Primary entry point |
|---|---|---|
| 0 | `SyntheticMacroEnv`, 106-D normalized state, 20 macro actions, action masks, skill boundary, SC2 adapter, replay | `scripts.collect_synthetic` |
| 1 | Mask-aware PPO, GAE, checkpoints, deterministic evaluation | `scripts.train_ppo` |
| 2 | Opponent-conditioned RSSM, observation/reward/continue/event/opponent-action/value heads, KL free bits and balancing, sequence replay, open-loop evaluation | `scripts.train_world_model` |
| 3 | Five-head dynamics ensemble, disagreement, uncertainty termination, pessimistic imagined reward, lambda-return actor-critic, PPO and imagination warmup | `scripts.train_hybrid` |
| 4 | Known opponent-ID embedding, online `OpponentContextEncoder`, scripted opponent pool | `sc2wmrl.models.opponent_encoder` |
| 5 | Immutable historical snapshots, exploiter records, Elo, payoff matrix, role-weighted sampling | `scripts.train_league` |

## Installation

Python 3.10+ is required. CPU is sufficient for every synthetic test; GPU acceleration is optional.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -e .
python -m pytest
```

On Windows, `requirements.txt` installs the CUDA 12.8 PyTorch build by default. Confirm that CUDA is usable before training:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU fallback')"
```

All training configurations use `device: auto`, which selects CUDA when available and otherwise falls back to CPU. Set `device: cuda` to require GPU execution or `device: cpu` for an explicit CPU run.

## Short End-to-End Workflow

```powershell
# 1. Collect reloadable macro trajectories from the synthetic environment.
python -m scripts.collect_synthetic --config configs/env/synthetic.yaml --episodes 100 --output outputs/synthetic_replay.npz

# 2. Train the model-free baseline.
python -m scripts.train_ppo --config configs/train/model_free.yaml

# 3. Pretrain and evaluate the RSSM on continuous replay sequences.
python -m scripts.train_world_model --config configs/train/world_model.yaml
python -m scripts.evaluate_world_model --checkpoint outputs/checkpoints/world_model.pt --replay outputs/synthetic_replay.npz

# 4. Keep real PPO enabled while gradually adding uncertainty-gated imagination.
python -m scripts.train_hybrid --config configs/train/hybrid.yaml

# 5. Initialize the League. Passing a checkpoint also creates a main snapshot and exploiter.
python -m scripts.train_league --config configs/league/small_league.yaml --checkpoint outputs/checkpoints/hybrid_policy.pt
```

## Core Guarantees

- Environment observations are fixed `float32[106]` vectors. Continuous values use range/log normalization, and unknown enemy positions include an explicit presence bit.
- Action masks are never empty. Environments reject illegal actions; PPO and latent actors assign illegal logits negative infinity.
- `MacroTransition` validates shapes, legal actions, finite values, and episode IDs on insertion and reload. The RSSM sequence sampler never crosses episodes and validates `next_observation -> observation` continuity.
- The world model uses stochastic RSSM latents, KL free bits, KL balancing, sequence masking, gradient clipping, and per-horizon open-loop error reporting.
- Ensemble disagreement contributes to both pessimistic rewards and adaptive rollout termination. NaNs, empty masks, predicted terminals, and excessive uncertainty terminate affected imagined trajectories.
- Real PPO updates remain enabled in hybrid training. Imagination weight only increases during warmup up to its configured cap and cannot dominate training at the start.
- League snapshot IDs cannot be overwritten. Elo and payoff-matrix values change only through explicit recorded results.

## Configuration

- `configs/env/`: synthetic and real-SC2 environment settings.
- `configs/model/ppo.yaml`: PPO network and optimizer hyperparameters.
- `configs/model/rssm.yaml`: world-model latent, ensemble, and KL parameters.
- `configs/train/model_free.yaml`, `world_model.yaml`, and `hybrid.yaml`: phase-specific training settings.
- `configs/league/small_league.yaml`: default small-League sampling proportions.

All rewards, model sizes, rollout horizons, uncertainty thresholds, and training coefficients are YAML-managed. Do not reorder `MacroAction`; its integer IDs are replay and checkpoint compatibility contracts.

## Evaluation and Experiments

`sc2wmrl.evaluation.world_model_metrics` provides a persistence baseline and event F1. `calibration` provides uncertainty/error correlation, while `payoff_matrix` outputs League results. Recommended comparisons include random policy, rule-based macro policy, PPO, deterministic/stochastic RSSM, no opponent context, no ensemble, no pessimistic reward, fixed rollout horizon, and no real PPO loss.

The implementation guarantees executable training and metrics code but does not fabricate research claims. Run controlled experiments with the same seed, opponent pool, and real-interaction budget before claiming sample-efficiency or win-rate improvements.

## Real SC2 Integration

Install the optional dependency:

```powershell
python -m pip install .[sc2]
```

Implement `RealSC2Backend` (`reset_game`, `execute_macro`, `get_action_mask`, and `close`) and provide it to `RealSC2MacroEnv`. The backend or existing bot owns the SC2 client lifecycle, building placement, unit selection, pathing, focus fire, and micro control. This project owns the macro-decision, state, reward, and trajectory contracts. Real execution also requires an authorized SC2 client, maps, and bot runtime; this repository never represents a synthetic match as a real SC2 result.

## Tests

```powershell
python -m pytest
```

Tests cover features, masks, replay persistence and sequences, the real-adapter contract, PPO update/checkpoints, RSSM loss/checkpoints/open-loop evaluation, uncertainty gating, imagined actor-critic, and League snapshot/Elo/payoff behavior. All tests run on CPU; GPU support is optional through PyTorch.

## Reporting, Visualization, and Reproducibility

Every training command accepts an optional `--run-dir`. Without it, the command creates an independent directory under `outputs/runs/` containing configuration, runtime metadata, CSV/JSONL metrics, TensorBoard events when available, checkpoints, evaluation artifacts, figures, replays, and reports.

```powershell
python -m scripts.train_ppo --config configs/train/model_free.yaml --run-dir outputs/runs/ppo_seed0
python -m scripts.postprocess_run --run-dir outputs/runs/ppo_seed0
python -m scripts.compare_experiments --runs outputs/runs/ppo_seed0 outputs/runs/ppo_seed1 outputs/runs/ppo_seed2 --group-by algorithm --output outputs/comparisons/ppo
python -m scripts.generate_report --runs outputs/runs/ppo_seed0 --output outputs/reports/ppo_seed0
```

`ExperimentLogger` writes the same scalar record to CSV, JSONL, and TensorBoard. Aggregation interpolates each run on true environment steps before computing mean, standard deviation, and confidence intervals. Plotters write PNG, SVG, and the corresponding CSV data. Missing metrics are listed as warnings; no synthetic chart data is generated.

## Real-Time Deployment

`RealtimeRLAgent` is a safe macro-level deployment layer. It strictly checks checkpoint version, observation dimension, action definitions, and optional configuration hash; it schedules inference at macro intervals or urgent events; records every decision; and uses an explicit fallback policy after an isolated error or timeout.

The project cannot invent a real SC2 client or micro-controller. Supply a configured BotAI/Ares backend factory that implements the documented `RealtimeBackend` protocol and `execute_macro` command sink:

```powershell
python -m scripts.run_trained_agent --checkpoint outputs/checkpoints/ppo_synthetic.pt --backend-factory your_backend_module:create_backend --deterministic --realtime --overlay --record-decisions
```

The runtime writes `decisions.jsonl`, `macro_timeline.csv`, `trajectory.npz`, `game_summary.json`, and an explicit replay-save failure record when an SC2 backend cannot save `replay.SC2Replay`. It never replaces a real match with a synthetic one.
