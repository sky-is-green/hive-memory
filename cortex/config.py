"""HiveConfig — all tunable parameters in one place.

Lets A/B testing and automated rollback swap whole configurations cleanly, and
loads/saves via the Gatekeeper merge contract (cortex.interop).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Optional

from cortex.interop import GatekeeperSeam


@dataclass
class HiveConfig:
    # --- retention ---
    decay_multiplier_init: float = 1.8
    remembrance_threshold: float = 0.65
    stale_threshold: int = 20
    # --- membrane ---
    dedup_threshold: float = 0.92
    drift_threshold: float = 0.6
    # --- routing ---
    routing_threshold: int = 2
    # --- sieve ---
    ultra_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    medium_model: str = "microsoft/graphcodebert-base"
    enable_medium: bool = False  # medium drone is heavy + VRAM-contending; opt-in
    vocab_boost: float = 0.15
    confidence_mode: str = "mcdropout"  # mcdropout | single | off
    # --- security ---
    sanitize_context: bool = True
    # --- focal ---
    generation_headroom: int = 2048
    max_context: int = 8192
    max_chunks: int = 1000
    max_tokens: Optional[int] = None  # reply cap for iteration/stability runs
    # When the backend returns a refusal/hedge ("no information regarding X"),
    # do NOT store it as a chunk: such replies pollute the store and later get
    # retrieved as "context", poisoning retrieval. Filtering them keeps the
    # store fact-bearing (live finding: ~50% of replies were hedges).
    filter_hedge_replies: bool = True
    budget_ranges: dict = field(default_factory=lambda: {
        "ultra_small": (1000, 3000),
        "medium": (3000, 5000),
        "escalation": (4000, 6000),
    })

    @classmethod
    def defaults(cls) -> "HiveConfig":
        return cls()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HiveConfig":
        known = {f.name for f in fields(cls)}
        cleaned = {k: v for k, v in (data or {}).items() if k in known}
        # JSON round-trips tuples as lists; restore the budget-range tuples.
        if "budget_ranges" in cleaned and cleaned["budget_ranges"]:
            cleaned["budget_ranges"] = {
                k: tuple(v) for k, v in cleaned["budget_ranges"].items()
            }
        return cls(**cleaned)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "HiveConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def apply_gatekeeper_overrides(
        self, overrides: dict, seam: Optional[GatekeeperSeam] = None
    ) -> "HiveConfig":
        """Return a new config with Gatekeeper-provided overrides applied safely."""
        seam = seam or GatekeeperSeam()
        merged = seam.merge_config(self.to_dict(), overrides)
        return self.from_dict(merged)
