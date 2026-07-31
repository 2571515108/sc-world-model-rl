"""League snapshot preservation, Elo, payoff and role sampling tests."""

import unittest

from sc2wmrl.league.league import League


class LeagueTests(unittest.TestCase):
    """Exercise the smallest useful League lifecycle."""
    def test_snapshot_payoff_and_elo(self) -> None:
        league = League(seed=3); league.add_scripted_opponents(); snapshot = league.add_snapshot("main.pt", 1100, {"update": 4})
        before = league.pool.get(snapshot.snapshot_id).rating; new_a, new_b = league.record_result(snapshot.snapshot_id, "economy_bot", 1.0)
        labels, matrix = league.payoff_matrix()
        self.assertIn(snapshot.snapshot_id, labels); self.assertGreater(new_a, before); self.assertEqual(matrix[labels.index(snapshot.snapshot_id)][labels.index("economy_bot")], 1.0)
    def test_duplicate_snapshot_id_is_not_overwritten(self) -> None:
        league = League(); league.add_snapshot("a.pt", 1000, {})
        with self.assertRaises(ValueError): league.pool.add(league.pool.get("snapshot-00000"))
