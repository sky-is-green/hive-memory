"""Integration: P1-P10 protocol driver runs (mock mode)."""

import json

from backend.lmstudio import LMStudioBackend
from cortex.baselines.runner import load_conversations
from cortex.e2e import FakeUltraSmall, MockTransport
from experiments.run_p1_p10 import PredictionSuite, _load_labels
from oracle.async_oracle import AsyncOracle
from sieve.medium import MediumDrone


def _suite():
    convs = load_conversations("tests/fixtures/generated")
    labels = _load_labels(convs)
    backend = LMStudioBackend(base_url="localhost", model="m", transport=MockTransport())
    oracle = AsyncOracle(
        generate_fn=lambda p: json.dumps({"sufficient": True, "used_pieces": [], "missing": [], "score": 4})
    )
    return PredictionSuite(
        backend, FakeUltraSmall(), MediumDrone(score_pair_fn=lambda q, c: 0.5),
        convs, labels, oracle, live=False,
    )


def test_protocol_runs_all_ten_predictions():
    results = _suite().run()
    assert len(results) == 10
    assert [r.id for r in results] == [f"P{i}" for i in range(1, 11)]
    for r in results:
        assert r.title
        assert r.status in ("PASS", "FAIL", "SKIP", "REPORT")


def test_protocol_predictions_have_evidence_or_note():
    for r in _suite().run():
        assert r.evidence or r.note


def test_protocol_stable_ids():
    ids = {r.id for r in _suite().run()}
    assert ids == {f"P{i}" for i in range(1, 11)}