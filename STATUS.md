# Implementation status

Last updated: 2026-07-31

## Completed phases

- **Phase 0:** fixed normalized macro features, synthetic environment, SC2 adapter contract, legal action mask, reward accounting, replay persistence and sequence sampling.
- **Phase 1:** mask-aware PPO/GAE, resumable checkpoint, deterministic synthetic evaluation.
- **Phase 2:** opponent-conditioned RSSM, observation/reward/continue/event/opponent-action/value heads, KL free bits and balancing, sequence training, multi-horizon open-loop evaluation.
- **Phase 3:** five-head dynamics ensemble, disagreement uncertainty, pessimistic reward, adaptive stop, lambda-return imagined actor-critic, real PPO + warmed-up imagination distillation.
- **Phase 4:** identity embedding, online opponent-context/strategy encoder, all six scripted opponents, action/strategy labels recorded in replay.
- **Phase 5:** immutable snapshot records, exploiter registration, role-weighted pool sampling, Elo updates and payoff matrix.

## Latest verification

Executed in `work/phase01-venv` after installing `requirements.txt`:

- `python -m pytest`: **18 passed**.
- `python -m scripts.collect_synthetic --episodes 12`: **1,152** transitions saved and reloaded.
- `python -m scripts.train_world_model --config configs/train/world_model.yaml`: completed 50 RSSM updates, wrote `outputs/checkpoints/world_model.pt`, and emitted finite open-loop errors through horizon 16.
- `python -m scripts.train_hybrid --config configs/train/hybrid.yaml`: completed 10 updates, including warmup, uncertainty-gated imagination and checkpoint save.
- `python -m scripts.train_league --config configs/league/small_league.yaml`: created a six-opponent scripted League output.

## External limitation

No SC2 client/runtime or existing micro controller is in this workspace. Real SC2 execution requires a concrete `RealSC2Backend`, licensed client, maps, and bot runtime. The adapter is covered by contract test; no synthetic result is presented as a real SC2 result.

## Research result policy

The implementation produces metrics but does not fabricate empirical claims. Sample-efficiency, win-rate, uncertainty-calibration, and ablation conclusions require the documented controlled experiment runs.

## Supplement: reporting and real-time deployment

- Added persistent experiment logging (CSV, JSONL, TensorBoard where installed), run directories, aligned multi-seed aggregation, smoothing, statistical confidence bands, PNG/SVG/CSV plotting, and HTML/Markdown/JSON reporting.
- Added strict deployment checkpoint loading, macro inference scheduling, lifecycle-aware skill dispatch, action-mask enforcement, safe fallback, optional overlay, JSONL/CSV/NPZ decision recording, and best-effort replay management.
- Added mock real-time lifecycle tests; a real SC2 smoke test remains dependent on an externally configured BotAI/Ares backend, licensed SC2 client, maps, and micro controller.
- Desktop-project verification on 2026-07-31: 25 unit tests passed and source compilation passed. Plot/report execution additionally requires the reporting dependencies listed in `requirements.txt`; real SC2 smoke testing remains explicitly external-state dependent.
