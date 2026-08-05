# Expert Replay Pretraining

## Scope and safety boundaries

This pipeline converts one human player's fog-of-war viewpoint from a
`.SC2Replay` archive into the project's macro-transition format. It does not
read an omniscient observer state, infer hidden enemy information, or treat raw
replay commands as PPO rollout data.

The same converted dataset has two different roles:

1. Every transition trains the visible-state world model.
2. Only high-confidence macro labels train the PPO actor with behavior cloning.
3. The PPO value head is not supervised by replay outcomes. It is recalibrated
   during later online PPO training.

The observed opponent's macro action is unknown in a one-player replay
viewpoint. Each such transition has `opponent_action_valid=false`, so the
world-model opponent-action loss excludes it.

## Dataset layout

```text
data/replays/                         Local input archives, ignored by Git
outputs/replays/expert_terran.npz     Numeric transition arrays
outputs/replays/expert_terran.npz.json
                                      Transition metadata and label evidence
outputs/replays/expert_terran.npz.summary.json
                                      Conversion summary per replay viewpoint
```

Each metadata entry retains the source replay, selected player, result, APM,
raw ability IDs, macro-label confidence, label evidence, and action-mask
corrections. This makes label quality auditable before any expensive training.

## Conversion requirements

Metadata inspection only needs `mpyq`. Conversion starts StarCraft II through
PySC2, and requires the exact SC2 base build recorded by the replay. Do not
substitute a nearby build: replay playback compatibility is not guaranteed
between minor SC2 releases.

Before conversion, `prepare_sc2_replay_version` can send
`RequestReplayInfo(download_data=true)` through the current SC2 client. This
requests replay data from Blizzard and avoids PySC2's outdated static version
table by launching a fully specified header-derived version. The command
reports the exact result; it does not assume that a retired executable remains
available to download.

If replay startup reports a missing `.s2ma` archive, the map cache is missing.
Obtain the exact `.SC2Map` archive for the replay map, place it under the SC2
`Maps/` directory (or use an absolute path), and set `map_file` in
`configs/replays/expert_terran.yaml`. The converter supplies it directly in
`RequestStartReplay.map_data` and does not need a Battle.net cache entry.

```powershell
python -m scripts.inspect_sc2_replay data/replays/example.SC2Replay --race Terran --min-apm 150
python -m scripts.prepare_sc2_replay_version --replay data/replays/example.SC2Replay --output outputs/replay_version.json
python -m scripts.convert_sc2_replays --config configs/replays/expert_terran.yaml --replays data/replays/example.SC2Replay --output outputs/replays/expert_terran.npz
python -m scripts.validate_expert_replay_dataset --replay outputs/replays/expert_terran.npz --output outputs/reports/expert_dataset_audit.json
```

Use `--player-id` only when the automatic race, result, and APM filters should
not choose the viewpoint. Expert train/validation splits must remain disjoint
by whole replay episode.

## Training order

```powershell
python -m scripts.train_world_model --config configs/train/world_model_expert.yaml
python -m scripts.train_behavior_cloning --config configs/train/behavior_cloning_expert.yaml
python -m scripts.train_ppo --config configs/train/ppo_real_after_expert.yaml
python -m scripts.merge_replays --inputs outputs/replays/expert_terran.npz outputs/replays/ppo_real_after_expert.npz --output outputs/replays/real_expert_plus_online.npz
python -m scripts.train_world_model --config configs/train/world_model_expert_plus_online.yaml
```

Behavior cloning uses masked cross entropy, confidence weighting, and action
class balancing. It updates the shared actor trunk and policy head but leaves
the value head untouched. Online PPO remains the only source of PPO advantages,
returns, and critic regression targets.

## Quality gates

Do not use a converted dataset for pretraining until the audit reports
non-passive macro coverage, nonzero behavior-cloning labels, sensible action
mask correction counts, terminal transitions, and an unknown-opponent-action
fraction of one for single-viewpoint expert replays. One replay is only an
end-to-end integration test; meaningful pretraining needs many whole matches
covering Terran matchups, maps, openings, and game lengths.
