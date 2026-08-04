"""Conservative macro policy for collecting useful real-game trajectories."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from sc2wmrl.envs.base_macro_env import MacroAction


class RuleBasedAgent:
    """Terran opening policy that preserves minerals for critical milestones."""

    def act(self, observation: np.ndarray, action_mask: np.ndarray, raw_state: Mapping[str, object] | None = None) -> int:
        """Choose an available priority action; observations are kept for API parity."""
        del observation
        state = raw_state or {}
        buildings = state.get("buildings", {}) if isinstance(state.get("buildings", {}), Mapping) else {}
        supply = float(state.get("current_supply", 0.0)); cap = float(state.get("maximum_supply", 0.0))
        army = float(state.get("army_value", 0.0))
        mask = np.asarray(action_mask, dtype=np.bool_)
        barracks = float(buildings.get("barracks", 0))
        supply_depots = float(buildings.get("supply_depot", 0))

        # Terran Barracks requires a completed Supply Depot.  Preserve minerals
        # for this two-step opening instead of spending them on workers or gas.
        if barracks < 1:
            if supply_depots < 1 and mask[MacroAction.BUILD_SUPPLY]:
                return int(MacroAction.BUILD_SUPPLY)
            if supply_depots < 1:
                return int(MacroAction.NO_OP)
            if mask[MacroAction.BUILD_BARRACKS]:
                return int(MacroAction.BUILD_BARRACKS)
            return int(MacroAction.NO_OP)

        priorities: list[MacroAction] = []
        if cap - supply <= 3:
            priorities.append(MacroAction.BUILD_SUPPLY)
        # Once a credible force exists, issue the attack before queuing more
        # Marines.  This guarantees collection covers combat transitions
        # instead of indefinitely cycling through production commands.
        if army >= 250:
            priorities.append(MacroAction.ATTACK_ENEMY_MAIN)
        priorities.append(MacroAction.TRAIN_BASIC_ARMY)
        if float(buildings.get("refinery", 0)) < 1:
            priorities.append(MacroAction.BUILD_REFINERY)
        if float(buildings.get("factory", 0)) < 1:
            priorities.append(MacroAction.BUILD_FACTORY)
        priorities.append(MacroAction.TRAIN_WORKERS)
        if army > 0:
            priorities.extend((MacroAction.SCOUT_ENEMY_MAIN, MacroAction.DEFEND_MAIN))
        priorities.append(MacroAction.NO_OP)
        for action in priorities:
            if mask[action]: return int(action)
        return int(np.flatnonzero(mask)[0])
