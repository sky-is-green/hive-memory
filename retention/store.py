"""Context store (retention layer) — the central chunk repository.

Stores conversation context as discrete chunks with metadata: decay state,
times saved, last-referenced turn, and a content fingerprint for dedup. An
optional ``embed_fn`` provides embeddings lazily (used by dedup/assembly).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from cortex.baselines.metrics import now_iso


@dataclass
class ContextChunk:
    id: str
    content: str
    turn: int
    fingerprint: str
    timestamp: str = ""
    decay_multiplier: float = 1.0
    times_saved: int = 0
    last_referenced_turn: Optional[int] = None
    relevance_history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "turn": self.turn,
            "fingerprint": self.fingerprint,
            "timestamp": self.timestamp,
            "decay_multiplier": self.decay_multiplier,
            "times_saved": self.times_saved,
            "last_referenced_turn": self.last_referenced_turn,
            "relevance_history": [list(x) for x in self.relevance_history],
        }


class ContextStore:
    """Chunk-level conversation store with turn indexing and decay metadata."""

    def __init__(
        self,
        embed_fn: Optional[Callable[[str], object]] = None,
        max_chunks: Optional[int] = None,
    ) -> None:
        self.chunks: dict[str, ContextChunk] = {}
        self.turn_index: dict[int, list[str]] = {}
        self.embed_fn = embed_fn
        self._embeddings: dict = {}
        self.max_chunks = max_chunks

    def add_chunk(self, turn: int, content: str, chunk_id: Optional[str] = None) -> str:
        fingerprint = hashlib.md5(content.encode("utf-8")).hexdigest()[:12]
        cid = chunk_id or hashlib.md5(f"{turn}:{content}".encode("utf-8")).hexdigest()[:12]
        self.chunks[cid] = ContextChunk(
            id=cid,
            content=content,
            turn=turn,
            fingerprint=fingerprint,
            timestamp=now_iso(),
            last_referenced_turn=turn,
        )
        self.turn_index.setdefault(turn, []).append(cid)
        return cid

    def apply_refresh(self, refresh_map: dict[str, int]) -> None:
        """Membrane-before-Retention fix: refresh kept chunks' decay state so a
        chunk merged with a fresher duplicate isn't penalized for stale age."""
        for cid, freshest_turn in refresh_map.items():
            chunk = self.chunks.get(cid)
            if chunk:
                chunk.last_referenced_turn = max(
                    chunk.last_referenced_turn or 0, freshest_turn
                )

    def all_chunks(self) -> list[ContextChunk]:
        return list(self.chunks.values())

    def all_contents(self) -> list[str]:
        return [c.content for c in self.all_chunks()]

    def all_embeddings(self) -> dict:
        for c in self.all_chunks():
            if c.id not in self._embeddings and self.embed_fn is not None:
                self._embeddings[c.id] = self.embed_fn(c.content)
        return self._embeddings

    def get_deletion_candidates(self) -> list[ContextChunk]:
        """Chunks that would be evicted on overflow (least-recently-referenced)."""
        if not self.max_chunks or len(self.chunks) <= self.max_chunks:
            return []
        ordered = sorted(self.all_chunks(), key=lambda c: c.last_referenced_turn or 0)
        return ordered[: len(self.chunks) - self.max_chunks]

    def get_recent_chunks(self, n: int) -> list[ContextChunk]:
        """Chunks from the last ``n`` turns, in chronological order."""
        ids: list[str] = []
        for t in sorted(self.turn_index)[-n:]:
            ids.extend(self.turn_index[t])
        return [self.chunks[i] for i in ids]

    def get_turns(self) -> list[int]:
        return sorted(self.turn_index)

    def count(self) -> int:
        return len(self.chunks)

    # ------------------------------------------------------------------
    # Checkpoint serialization (resume support)
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "max_chunks": self.max_chunks,
            "chunks": [c.to_dict() for c in self.all_chunks()],
            "turn_index": {str(t): ids for t, ids in self.turn_index.items()},
        }

    @classmethod
    def from_dict(cls, data: dict, embed_fn: Optional[Callable[[str], object]] = None) -> "ContextStore":
        store = cls(embed_fn=embed_fn, max_chunks=data.get("max_chunks"))
        for raw in data.get("chunks", []):
            chunk = ContextChunk(
                id=raw["id"],
                content=raw["content"],
                turn=raw["turn"],
                fingerprint=raw.get("fingerprint", ""),
                timestamp=raw.get("timestamp", ""),
                decay_multiplier=raw.get("decay_multiplier", 1.0),
                times_saved=raw.get("times_saved", 0),
                last_referenced_turn=raw.get("last_referenced_turn"),
                relevance_history=[tuple(x) for x in raw.get("relevance_history", [])],
            )
            store.chunks[chunk.id] = chunk
        store.turn_index = {
            int(t): list(ids) for t, ids in data.get("turn_index", {}).items()
        }
        return store
