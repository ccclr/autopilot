from __future__ import annotations

import sys
import unittest
from pathlib import Path


RL_ROOT = Path(__file__).resolve().parents[1]
if str(RL_ROOT) not in sys.path:
    sys.path.insert(0, str(RL_ROOT))

from cmab.factor_freeze import FactorFreezer


def _reward(latency_ms: float) -> float:
    return 1000.0 / (latency_ms + 1.0)


def _arm(batch=100000, header=32, cut=4, timeout=0, k=1) -> str:
    return (
        f"batch_size={batch},header_size={header},"
        f"cut_condition_type={cut},fast_path_timeout={timeout},k={k}"
    )


def _fill(freezer: FactorFreezer, arm: str, latency_ms: float, count: int) -> None:
    for _ in range(count):
        freezer.observe(arm, _reward(latency_ms))


class FactorFreezeTests(unittest.TestCase):
    def test_equivalent_k_collapses_once_each_level_has_enough_samples(self):
        low = _arm(k=1)
        high = _arm(k=4)
        freezer = FactorFreezer(min_samples=20, margin_ms=30)
        _fill(freezer, low, 400, 20)
        _fill(freezer, high, 410, 20)

        self.assertEqual(freezer.filter_arms([low, high]), [low])

    def test_waits_until_every_level_reaches_the_sample_floor(self):
        low = _arm(k=1)
        high = _arm(k=4)
        freezer = FactorFreezer(min_samples=20, margin_ms=30)
        _fill(freezer, low, 400, 20)
        _fill(freezer, high, 410, 19)

        self.assertEqual(freezer.filter_arms([low, high]), [low, high])

    def test_keeps_a_level_that_is_worse_than_the_margin(self):
        small = _arm(batch=100000)
        large = _arm(batch=500000)
        freezer = FactorFreezer(min_samples=20, margin_ms=30)
        _fill(freezer, small, 400, 20)
        _fill(freezer, large, 800, 20)

        self.assertEqual(freezer.filter_arms([small, large]), [small, large])

    def test_timeout_stays_when_cuts_disagree(self):
        arms = [
            _arm(cut=2, timeout=0),
            _arm(cut=2, timeout=100),
            _arm(cut=4, timeout=0),
            _arm(cut=4, timeout=100),
        ]
        freezer = FactorFreezer(min_samples=20, margin_ms=30)
        _fill(freezer, arms[0], 1000, 20)
        _fill(freezer, arms[1], 400, 20)
        _fill(freezer, arms[2], 400, 20)
        _fill(freezer, arms[3], 1000, 20)

        self.assertEqual(freezer.filter_arms(arms), arms)

    def test_timeout_collapses_when_every_cut_agrees(self):
        arms = [
            _arm(cut=2, timeout=0),
            _arm(cut=2, timeout=100),
            _arm(cut=4, timeout=0),
            _arm(cut=4, timeout=100),
        ]
        freezer = FactorFreezer(min_samples=20, margin_ms=30)
        _fill(freezer, arms[0], 420, 20)
        _fill(freezer, arms[1], 400, 20)
        _fill(freezer, arms[2], 415, 20)
        _fill(freezer, arms[3], 405, 20)

        self.assertEqual(
            freezer.filter_arms(arms),
            [_arm(cut=2, timeout=100), _arm(cut=4, timeout=100)],
        )

    def test_later_samples_can_reopen_a_collapsed_level(self):
        low = _arm(k=1)
        high = _arm(k=4)
        freezer = FactorFreezer(min_samples=20, margin_ms=30)
        _fill(freezer, low, 400, 20)
        _fill(freezer, high, 410, 20)
        self.assertEqual(freezer.filter_arms([low, high]), [low])

        _fill(freezer, low, 800, 21)
        self.assertEqual(freezer.filter_arms([low, high]), [low, high])

    def test_cover_fills_the_level_that_is_still_short(self):
        low = _arm(k=1)
        high = _arm(k=4)
        freezer = FactorFreezer(min_samples=20, margin_ms=30)

        self.assertEqual(freezer.cover_arm([low, high]), low)
        _fill(freezer, low, 400, 20)
        self.assertEqual(freezer.cover_arm([low, high]), high)
        _fill(freezer, high, 410, 20)
        self.assertIsNone(freezer.cover_arm([low, high]))


if __name__ == "__main__":
    unittest.main()
