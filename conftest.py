"""Ensure the repo root is importable so `logs`, `cortex`, and `tests` packages
resolve regardless of how pytest is invoked."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
