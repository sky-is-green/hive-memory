"""Deterministic P2 retrieval diagnostic (no LLM oracle).

The oracle-based "retrieval_precision" in a run is really a per-turn context-
sufficiency rate: ``predicted_relevant`` is hardcoded True and
``actually_relevant`` is the oracle's sufficiency verdict, so recall always
reports 100% and false-eviction always 0%. This module measures P2 the way the
white paper defines it — from labeled query-chunk pairs — using the synthetic
corpus's own ground truth: each user query in a fixture has a known assistant
answer, and a turn's relevant context is prior fixture content on the same
topic.

Per sampled turn it computes:
  - ``answer_facts``: the distinctive fact terms the fixture's ground-truth
    answer adds beyond the query (e.g. "Redis", "TTL" for a session-store ask).
  - ``retrievable``: whether the answer facts *could* exist in history — i.e.
    the topic was already covered by a prior fixture turn. First-mention turns
    are structurally unretrievable and reported separately.
  - ``hit``: the share of answer facts present in the assembled context.

Aggregates:
  - ``retrieval_recall``: hits over all sampled turns.
  - ``retrieval_recall_retrievable``: hits over only retrievable turns (the
    hive's real retrieval quality).
  - ``retrieval_precision``: sentence-level approximation — of the content in
    the assembled context, the share that shares a topic term with the query.
"""

from __future__ import annotations

import re
import statistics
from typing import Optional

STOPWORDS = {
    "about", "above", "after", "again", "against", "all", "also", "any",
    "are", "because", "been", "before", "being", "below", "between", "both",
    "but", "can", "cannot", "could", "did", "does", "doing", "down", "during",
    "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "however", "into", "its", "itself", "let", "may", "me", "more", "most",
    "must", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "out", "over", "own", "same",
    "say", "says", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "very",
    "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "would", "you", "your", "yours",
    # corpus scaffolding / question boilerplate
    "address", "answer", "asked", "assistant", "based", "change", "code",
    "context", "conversation", "decision", "does", "fit", "fits", "given",
    "handle", "history", "how", "key", "let", "need", "please", "provide",
    "query", "recommend", "respond", "response", "review", "schema", "service",
    "should", "show", "summary", "system", "turn", "user", "walk", "work",
    "would", "through", "consolidate", "earlier", "currently", "implement",
    "information", "regarding", "specific", "right", "approach", "recall",
    "use", "using", "keep", "keeps", "set", "should", "pipeline",
    # code boilerplate inside the fixture's ```python answers (not facts)
    "accept", "async", "await", "cache", "class", "consistent", "decided",
    "def", "default", "defaulted", "every", "init", "monthly", "process",
    "python", "request", "return", "rotated", "self", "true",
}

WORD_RE = re.compile(r"[a-z][a-z\-]{3,}")
CODE_BLOCK_RE = re.compile(r"```[a-z]*\n.*?```", re.DOTALL | re.IGNORECASE)


def _strip_code(text: str) -> str:
    return CODE_BLOCK_RE.sub(" ", text or "")


def _content_terms(text: str) -> set[str]:
    words = set(WORD_RE.findall(_strip_code(text or "").lower()))
    return {w for w in words if w not in STOPWORDS}


def _fixture_answer_map(conversations: list[dict]) -> dict[str, dict[str, str]]:
    """conversation_id -> {user_query: immediately-following assistant answer}.

    For repeated queries the *first* occurrence is kept (the run's turns map to
    fixture turns in order; the first ask is the one whose answer is novel)."""
    mapping: dict[str, dict[str, str]] = {}
    for conv in conversations:
        cid = conv.get("conversation_id", "unknown")
        turns = conv.get("turns", [])
        per_conv: dict[str, str] = {}
        for i, t in enumerate(turns):
            if t.get("role") != "user":
                continue
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            per_conv.setdefault(
                t.get("content", ""),
                nxt.get("content", "") if nxt and nxt.get("role") == "assistant" else "",
            )
        mapping[cid] = per_conv
    return mapping


def _is_retrievable(query: str, prior_fixture_text: str) -> bool:
    """A turn is retrievable when its topic was already covered before it was
    asked — otherwise the answer could not exist in history (first mention).

    Requires at least two distinct query content-terms to appear in prior
    history (a single shared word such as "auth" or "pipeline" is too loose and
    mislabels first mentions as retrievable).
    """
    q_terms = _content_terms(query)
    if not q_terms:
        return True
    prior = _content_terms(prior_fixture_text)
    return len(q_terms & prior) >= 2


def _answer_fact_terms(query: str, answer: str) -> set[str]:
    """The distinctive facts the ground-truth answer adds beyond the query."""
    return _content_terms(answer) - _content_terms(query)


def _sentence_split(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if s.strip()]


def _turn_precision(query: str, assembled: str) -> Optional[float]:
    """Sentence-level precision proxy: share of assembled-content sentences that
    share a topic term with the query. None when there is no assembled content."""
    assembled = assembled or ""
    if not assembled.strip():
        return None
    q_terms = _content_terms(query)
    if not q_terms:
        return None
    sentences = _sentence_split(assembled)
    if not sentences:
        return None
    relevant = 0
    for s in sentences:
        if _content_terms(s) & q_terms:
            relevant += 1
    return relevant / len(sentences)


def compute_retrieval_vs_fixture(records: list[dict], conversations: list[dict]) -> dict:
    """Compute deterministic P2 metrics from run records + fixture conversations.

    ``records`` is the run report's ``conversations`` list (each with
    ``conversation_id`` and per-turn ``query`` / ``assembled_content``).
    ``conversations`` is the fixture list (as loaded by
    ``cortex.baselines.runner.load_conversations``).
    """
    answer_map = _fixture_answer_map(conversations)
    # fixture text before each user query, per conversation
    prior_map: dict[str, dict[str, str]] = {}
    for conv in conversations:
        cid = conv.get("conversation_id", "unknown")
        acc: dict[str, str] = {}
        buf: list[str] = []
        for t in conv.get("turns", []):
            if t.get("role") == "user":
                acc.setdefault(t.get("content", ""), " ".join(buf))
                buf.append(t.get("content", "") or "")
            else:
                buf.append(t.get("content", "") or "")
        prior_map[cid] = acc

    sampled: list[dict] = []
    for conv in records:
        cid = conv.get("conversation_id", "unknown")
        conv_answers = answer_map.get(cid, {})
        conv_prior = prior_map.get(cid, {})
        for t in conv.get("turns", []):
            q = t.get("query", "")
            answer = conv_answers.get(q, "")
            if not answer:
                continue
            prior_text = conv_prior.get(q, "")
            facts = _answer_fact_terms(q, answer)
            assembled = t.get("assembled_content") or ""
            acc_terms = _content_terms(assembled)
            hits = facts & acc_terms
            hit_ratio = len(hits) / len(facts) if facts else None
            sampled.append({
                "turn": t.get("turn"),
                "conversation_id": cid,
                "query": q[:80],
                "answer_facts": sorted(facts),
                "facts_found": sorted(hits),
                "hit_ratio": hit_ratio,
                "retrievable": _is_retrievable(q, prior_text),
                "first_mention": not _is_retrievable(q, prior_text),
                "precision": _turn_precision(q, assembled),
            })

    if not sampled:
        return {"sampled_turns": 0, "retrieval_recall": None,
                "retrieval_recall_retrievable": None, "retrieval_precision": None,
                "turns": [], "note": "no sampled turns with fixture answers"}

    measurable = [s for s in sampled if s["hit_ratio"] is not None]
    recall = (statistics.mean(1.0 if s["hit_ratio"] >= 0.5 else 0.0 for s in measurable)
              if measurable else None)
    retr = [s for s in measurable if s["retrievable"]]
    recall_retr = (statistics.mean(1.0 if s["hit_ratio"] >= 0.5 else 0.0 for s in retr)
                   if retr else None)
    precs = [s["precision"] for s in sampled if s["precision"] is not None]
    precision = statistics.mean(precs) if precs else None

    return {
        "sampled_turns": len(sampled),
        "measurable_turns": len(measurable),
        "retrievable_turns": len(retr),
        "first_mention_turns": len(sampled) - len(retr),
        "retrieval_recall": round(recall * 100.0, 1) if recall is not None else None,
        "retrieval_recall_retrievable": (
            round(recall_retr * 100.0, 1) if recall_retr is not None else None),
        "retrieval_precision": round(precision * 100.0, 1) if precision is not None else None,
        "turns": sampled,
    }


def main() -> None:
    import argparse
    import json
    from pathlib import Path

    from cortex.baselines.runner import load_conversations

    parser = argparse.ArgumentParser(description="Deterministic P2 retrieval diagnostic")
    parser.add_argument("run_dir", help="a run directory containing run_report.json")
    parser.add_argument("--conversations", default="tests/fixtures/generated")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    conversations = load_conversations(args.conversations)
    result = compute_retrieval_vs_fixture(report.get("conversations", []), conversations)

    print(f"run {run_dir.name}: deterministic P2 (fixture ground truth, no oracle)")
    print(f"  sampled turns         : {result['sampled_turns']}")
    print(f"  measurable (has facts): {result['measurable_turns']}")
    print(f"  retrievable           : {result['retrievable_turns']} "
          f"(first-mention excluded: {result['first_mention_turns']})")
    print(f"  retrieval_recall      : {result['retrieval_recall']}% "
          f"(all measurable turns)")
    print(f"  recall (retrievable)  : {result['retrieval_recall_retrievable']}% "
          f"(only turns where the answer could exist)")
    print(f"  retrieval_precision   : {result['retrieval_precision']}% "
          f"(sentence-level proxy)")
    print()
    for s in result.get("turns", []):
        if s["hit_ratio"] is None:
            label = "n/a "
        elif s["hit_ratio"] >= 0.5:
            label = "HIT "
        else:
            label = "miss"
        fm = " [first-mention]" if s["first_mention"] else ""
        print(f"  {label} turn {s['turn']:3d} {s['conversation_id']:9s} "
              f"hit={s['hit_ratio']} prec={s['precision']} {fm} :: {s['query']}")
        if s["facts_found"]:
            print(f"        facts found: {s['facts_found']}")
        elif s["answer_facts"]:
            print(f"        expected   : {s['answer_facts']}")


if __name__ == "__main__":
    main()