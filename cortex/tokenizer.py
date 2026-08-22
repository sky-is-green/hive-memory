"""Token counting.

``estimate_tokens`` is the cheap heuristic used everywhere. ``Tokenizer`` offers
an optional real tokenizer (tiktoken) when accuracy matters, falling back to the
heuristic if tiktoken is unavailable.
"""

from __future__ import annotations

from typing import Optional


def estimate_tokens(text: str) -> int:
    """Approximate token count (~1 token per 4 chars). No dependencies."""
    return max(1, len(text or "") // 4)


class Tokenizer:
    def __init__(self, use_real: bool = False) -> None:
        self.use_real = use_real
        self._enc = None
        if use_real:
            try:
                import tiktoken

                self._enc = tiktoken.get_encoding("cl100k_base")
            except Exception:  # noqa: BLE001
                self._enc = None

    @property
    def is_real(self) -> bool:
        return self._enc is not None

    def count(self, text: str) -> int:
        if self._enc is not None:
            return len(self._enc.encode(text or ""))
        return estimate_tokens(text)
