"""Wall-clock and peak-RSS accounting for experiment stages (ablations / budgets)."""

from __future__ import annotations

import resource
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


def peak_rss_kb() -> int:
    """Max RSS of this process (Linux: KB; macOS often bytes — stored as returned)."""
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


@dataclass
class StageTimer:
    """Accumulate named stage timings and peak RSS snapshots."""

    stages: dict[str, float] = field(default_factory=dict)
    rss_after_kb: dict[str, int] = field(default_factory=dict)
    t0: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start)
            self.rss_after_kb[name] = peak_rss_kb()

    def wall_seconds(self) -> float:
        return time.perf_counter() - self.t0

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_seconds": self.wall_seconds(),
            "max_rss_kb": peak_rss_kb(),
            "stage_seconds": dict(self.stages),
            "rss_after_kb": dict(self.rss_after_kb),
        }
