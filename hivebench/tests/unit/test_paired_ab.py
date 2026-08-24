"""paired_ab — paired hive-vs-FIFO answer-quality A/B (offline)."""

import json

import numpy as np

from experiments.paired_ab import run_paired
from experiments.retrieval_diagnostic import _answer_fact_terms, _content_terms


class _DistinctEmbedUltra:
    """FakeUltraSmall with distinct embeddings: the stock fake returns a
    constant vector, which makes the dedup pass collapse query and reply
    chunks as identical. One-hot term embeddings keep them distinct."""

    def score(self, query, chunks):
        from cortex.e2e import FakeUltraSmall

        return FakeUltraSmall().score(query, chunks)

    def embed(self, text):
        v = np.zeros(16)
        for i, w in enumerate(sorted(_content_terms(text))[:16]):
            v[i] = 1.0
        return v


class _ContextAwareBackend:
    """Stub backend: answers with the fixture facts only when the context
    already contains them — the behavior a real model should show when the
    needed facts are present. Deterministic and network-free."""

    def __init__(self, conversations):
        from experiments.retrieval_diagnostic import _fixture_answer_map

        per_conv = _fixture_answer_map(conversations)
        self.answers = {
            q: a
            for conv_answers in per_conv.values()
            for q, a in conv_answers.items()
        }

    def generate(self, context, query, sampling=None):
        answer = self.answers.get(query, "")
        facts = _answer_fact_terms(query, answer) if answer else set()
        if facts and facts <= _content_terms(context or ""):
            return "The answer is: " + " ".join(sorted(facts))
        return "I am not sure."


class _DumbBackend:
    """Stub backend that never states the facts."""

    def generate(self, context, query, sampling=None):
        return "I am not sure."


def _conv():
    """Two user turns; the second repeats the first, so its facts are
    retrievable from history (>=2 shared query terms; 'handle' is a diagnostic
    stopword, so the query needs distinct content terms)."""
    return [{
        "conversation_id": "c1",
        "profile": "code",
        "turns": [
            {"role": "user", "content": "Which tokens do I use for auth expiry?"},
            {"role": "assistant",
             "content": "Use JWT tokens with rotation and short expiry."},
            {"role": "user", "content": "Which tokens do I use for auth expiry?"},
        ],
    }]


def _ultra():
    from sieve.medium import MediumDrone

    return _DistinctEmbedUltra(), MediumDrone(score_pair_fn=lambda q, c: 0.5)


def test_strict_hive_win_when_fifo_drops_the_fact():
    convs = _conv()
    ultra, medium = _ultra()
    report = run_paired(convs, _ContextAwareBackend(convs), ultra, medium,
                        fifo_budget=1)  # FIFO window fits only the current turn
    m = report["metrics"]
    assert m["turns_compared"] == 1
    assert m["hive_only"] == 1
    assert m["fifo_only"] == 0
    assert m["hive_answer_recall"] == 100.0
    assert m["fifo_answer_recall"] == 0.0
    assert m["hive_ge_fifo_ratio"] == 100.0
    assert m["strict_hive_only_ratio"] == 100.0
    assert m["hive_avg_fact_hit_ratio"] == 1.0
    assert m["fifo_avg_fact_hit_ratio"] == 0.0


def test_fifo_catches_up_with_full_window():
    convs = _conv()
    ultra, medium = _ultra()
    # A full FIFO window includes the earlier assistant answer, so both arms
    # carry the facts -> both bucket, hive >= FIFO ratio still 100%.
    report = run_paired(convs, _ContextAwareBackend(convs), ultra, medium,
                        fifo_budget=100000)
    m = report["metrics"]
    assert m["both_sufficient"] == 1
    assert m["hive_only"] == 0
    assert m["hive_answer_recall"] == 100.0
    assert m["fifo_answer_recall"] == 100.0


def test_zero_recall_when_backend_never_states_facts():
    convs = _conv()
    ultra, medium = _ultra()
    report = run_paired(convs, _DumbBackend(), ultra, medium, fifo_budget=1)
    m = report["metrics"]
    assert m["turns_compared"] == 1
    assert m["neither_sufficient"] == 1
    assert m["hive_answer_recall"] == 0.0
    assert m["fifo_answer_recall"] == 0.0
    assert m["hive_ge_fifo_ratio"] == 0.0
    assert m["strict_hive_only_ratio"] == 0.0


def test_first_mention_excluded():
    convs = [{
        "conversation_id": "c2",
        "profile": "code",
        "turns": [
            {"role": "user", "content": "Which tokens do I use for auth expiry?"},
            {"role": "assistant",
             "content": "Use JWT tokens with rotation and short expiry."},
        ],
    }]
    ultra, medium = _ultra()
    report = run_paired(convs, _DumbBackend(), ultra, medium)
    assert report["turns_compared"] == 0
    assert report["first_mention_excluded"] == 1
    assert report["turns"] == []


def test_main_mock_writes_report(tmp_path, capsys):
    convs = _conv()
    data_dir = tmp_path / "convs"
    data_dir.mkdir()
    (data_dir / "c1.json").write_text(json.dumps(convs[0]), encoding="utf-8")
    out = tmp_path / "report.json"
    from experiments.paired_ab import main

    code = main(["--mock", "--conversations", str(data_dir),
                 "--output", str(out)])
    printed = capsys.readouterr().out
    assert code == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["metrics"]["turns_compared"] == 1
    assert doc["metrics"]["hive_answer_recall"] == 0.0  # mock replies carry no facts
    assert "answer recall" in printed


def test_main_missing_corpus(tmp_path):
    from experiments.paired_ab import main

    assert main(["--mock", "--conversations", str(tmp_path / "none")]) == 2