"""Ensure the system (`hive/`), evaluation-suite (`hivebench/`) and harness
(`harness/`) package roots are importable so `cortex`, `sieve`, `tests`,
`experiments`, `harness`, ... resolve regardless of how pytest is invoked
(editable install makes this unnecessary, but from-source runs stay supported)."""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
for _pkg_root in (os.path.join(ROOT, "hive"), os.path.join(ROOT, "hivebench"),
                  os.path.join(ROOT, "harness")):
    sys.path.insert(0, _pkg_root)