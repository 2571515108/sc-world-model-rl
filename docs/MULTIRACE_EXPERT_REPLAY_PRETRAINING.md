# Multi-Race Expert Replay Pretraining

## Design

Each P/T/Z replay is extracted twice: once through player 1's local
fog-of-war observation and once through player 2's. They become distinct
episodes. The data loader never concatenates a pair into a policy observation.
`paired_view_indices` is metadata for a future opponent-belief loss only.

The shared RSSM is conditioned on `player_races` and `opponent_races`. It
predicts transitions from a 14-class `UniversalIntent` vocabulary and receives
an additional masked universal-intent loss. Feature reconstruction is masked by
`next_feature_valid_masks`, so unavailable strategy and map-control fields are
not silently learned as zeros.

## Labels and actor routing

The dataset persists these independent targets:

- `universal_intents`: 14 cross-race macro intents for the shared world model.
- `race_actions`: broad, race-conditioned labels for future per-race actors.
- `terran_macro_actions`: the existing 20-action Terran vocabulary.

Only rows satisfying all of the following can update the existing Terran PPO
actor: `player_races == Terran`, `terran_macro_action_valid == true`, and a
label confidence at or above the configured threshold. Protoss and Zerg rows
remain available to the shared world model but cannot update the Terran policy
head.

## Dataset quality gates

Run `scripts.validate_multirace_replay_dataset` before training. It verifies
that all train-critical metadata resides in the NPZ, feature/mask dimensions
match, universal intents are bounded, paired views are reciprocal, at least two
episodes exist, and non-Terran rows are not routed to the Terran actor.

The converter records per-viewpoint successes and failures in
`expert_multirace.npz.conversion.json`. A replay that cannot launch because its
exact Base build or map is missing is reported there and is not silently
replaced with synthetic data.
