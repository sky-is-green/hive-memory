"""Deterministic synthetic conversation generator.

Produces the fixed test corpus (Appendix D.2):

  - 10 short   (10-20 turns, single topic)
  - 20 medium  (30-50 turns, 2-3 topic shifts)
  - 15 long    (80-100 turns, multiple shifts, cross-cutting dependencies)
  -  5 edge    (rapid topic switching, very long single turns, contradictions)

Conversations contain domain vocabulary and code blocks (for later router
tests) and long-range references to earlier decisions (for context-loss tests).
Output is written to ``tests/fixtures/generated/`` (gitignored).
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "generated"

# ---------------------------------------------------------------------------
# Domain content: per-topic feature + decisions (facts) that turn templates use.
# ---------------------------------------------------------------------------
TOPICS = {
    "authentication": {
        "feature": "auth service",
        "aspects": ["JWT expiry", "refresh-token rotation", "session store", "OAuth2 scopes", "bcrypt hashing"],
        "decisions": {
            "JWT expiry": "15 minutes",
            "refresh-token rotation": "rotate on every refresh",
            "session store": "Redis with TTL",
            "OAuth2 scopes": "read/write split",
            "bcrypt hashing": "cost factor 12",
        },
    },
    "database_schema": {
        "feature": "order schema",
        "aspects": ["normalization", "indexes", "soft deletes", "migrations", "foreign keys"],
        "decisions": {
            "normalization": "3NF with a denormalized read model",
            "indexes": "composite on (customer_id, created_at)",
            "soft deletes": "deleted_at column",
            "migrations": "Alembic, forward-only",
            "foreign keys": "ON DELETE CASCADE",
        },
    },
    "logging": {
        "feature": "log pipeline",
        "aspects": ["structured logs", "log levels", "sampling", "correlation ids", "retention"],
        "decisions": {
            "structured logs": "JSON via python logging",
            "log levels": "INFO by default, DEBUG in dev",
            "sampling": "10% trace sampling",
            "correlation ids": "X-Request-Id header",
            "retention": "30 days hot, 12 months cold",
        },
    },
    "deployment": {
        "feature": "deploy pipeline",
        "aspects": ["blue-green", "health checks", "rollbacks", "canary", "secrets"],
        "decisions": {
            "blue-green": "two live slots",
            "health checks": "/healthz with DB ping",
            "rollbacks": "auto on 5% error rate",
            "canary": "5% for 10 minutes",
            "secrets": "Vault, rotated monthly",
        },
    },
    "api_design": {
        "feature": "REST API",
        "aspects": ["pagination", "versioning", "rate limits", "error envelope", "idempotency"],
        "decisions": {
            "pagination": "cursor-based",
            "versioning": "URL /v1 prefix",
            "rate limits": "100 req/min per key",
            "error envelope": "problem+json",
            "idempotency": "Idempotency-Key header",
        },
    },
}

CODE_SNIPPETS = [
    "def handle_{feat}():\n    return {{\"ok\": True}}\n",
    "async def process_{feat}(request):\n    await request.accept()\n    return request\n",
    "class {Feat}Service:\n    def __init__(self):\n        self._cache = {{}}\n",
]

USER_TPL = [
    "Let's work on the {feature}. How should we handle {aspect}?",
    "What would you recommend for {aspect} in our {feature}?",
    "Can we change how {aspect} works in the {feature}?",
    "We need to address {aspect} for the {feature}. Walk me through it.",
    "How does {aspect} fit with {ref}?",
    "Show me the code for {aspect} in the {feature}.",
]

ASST_TPL = [
    "For the {feature}, {aspect} should use {decision}. Key decision: {aspect} = {decision}.\n```python\n{code}\n```",
    "The right approach for {aspect} is {decision}. Recall that earlier we decided {ref}.\n```python\n{code}\n```",
    "We'll set {aspect} to {decision} for the {feature}. This keeps {ref} consistent.\n```python\n{code}\n```",
]


def _fact_list(topic: str) -> list[str]:
    return [f"{k}={v}" for k, v in TOPICS[topic]["decisions"].items()]


def _turns_for_topic(rng: random.Random, topic: str, count: int,
                     established_facts: list[str]) -> list[dict]:
    data = TOPICS[topic]
    turns: list[dict] = []
    facts = _fact_list(topic)
    for i in range(count):
        aspect = rng.choice(data["aspects"])
        decision = data["decisions"][aspect]
        ref = rng.choice(established_facts) if established_facts and rng.random() < 0.5 else aspect
        code = rng.choice(CODE_SNIPPETS).format(
            feat=data["feature"].replace(" ", "_"),
            Feat="".join(w.capitalize() for w in data["feature"].split()),
        )
        user = rng.choice(USER_TPL).format(
            feature=data["feature"], aspect=aspect, ref=ref
        )
        asst = rng.choice(ASST_TPL).format(
            feature=data["feature"], aspect=aspect, decision=decision,
            ref=ref, code=code,
        )
        turns.append({"role": "user", "content": user})
        turns.append({"role": "assistant", "content": asst})
    # Record this topic's facts as established for cross-references.
    established_facts.append(f"{data['feature']}: {facts[0]}")
    return turns


def _make_conversation(rng: random.Random, profile: str, idx: int,
                       topic_names: list[str]) -> dict:
    conv_id = f"{profile}_{idx:03d}"
    established_facts: list[str] = []
    turns: list[dict] = []

    if profile == "short":
        topic = rng.choice(topic_names)
        n = rng.randint(10, 20)
        turns.extend(_turns_for_topic(rng, topic, n // 2, established_facts))

    elif profile == "medium":
        n_topics = rng.randint(2, 3)
        topics = rng.sample(topic_names, min(n_topics, len(topic_names)))
        per = rng.randint(15, 25)
        for t in topics:
            turns.extend(_turns_for_topic(rng, t, per // 2, established_facts))

    elif profile == "long":
        topics = rng.sample(topic_names, min(4, len(topic_names)))
        per = rng.randint(20, 25)
        for t in topics:
            turns.extend(_turns_for_topic(rng, t, per // 2, established_facts))

    elif profile == "edge":
        edge_type = idx % 3
        if edge_type == 0:
            # Rapid topic switching: many short blocks.
            for t in rng.sample(topic_names, len(topic_names)):
                turns.extend(_turns_for_topic(rng, t, 3, established_facts))
        elif edge_type == 1:
            # Very long single turns.
            topic = rng.choice(topic_names)
            data = TOPICS[topic]
            long_user = (
                "Given the long history of this project, consolidate everything "
                f"about the {data['feature']}. " + " ".join(
                    f"({k}) {v};" for k, v in data["decisions"].items()
                )
            )
            turns.append({"role": "user", "content": long_user})
            turns.append({"role": "assistant",
                          "content": f"Summary of {data['feature']}: " + ", ".join(_fact_list(topic))})
        else:
            # Contradictory instructions: same aspect, opposite decisions.
            topic = rng.choice(topic_names)
            data = TOPICS[topic]
            aspect = data["aspects"][0]
            turns.append({"role": "user",
                          "content": f"Set {aspect} to the default for the {data['feature']}."})
            turns.append({"role": "assistant",
                          "content": f"OK, {aspect} defaulted for the {data['feature']}."})
            turns.append({"role": "user",
                          "content": f"Actually, never mind — override {aspect} to a non-standard value now."})
            turns.append({"role": "assistant",
                          "content": f"Understood, overriding {aspect} for the {data['feature']}."})

    return {
        "conversation_id": conv_id,
        "profile": profile,
        "topic": topic_names[0],
        "turns": turns,
    }


def generate(output_dir: str | Path = OUTPUT_DIR, seed: int = 2026) -> list[dict]:
    rng = random.Random(seed)
    topic_names = list(TOPICS)
    profiles = (
        [("short", 10)] + [("medium", 20)] + [("long", 15)] + [("edge", 5)]
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[dict] = []
    idx = 0
    for profile, count in profiles:
        for i in range(count):
            conv = _make_conversation(rng, profile, i + 1, topic_names)
            path = out / f"{profile}_{i + 1:03d}.json"
            path.write_text(json.dumps(conv, indent=2), encoding="utf-8")
            written.append(conv)
            idx += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    convs = generate(args.output, args.seed)
    total_turns = sum(len(c["turns"]) for c in convs)
    print(f"Generated {len(convs)} conversations -> {args.output}")
    print(f"Total turns: {total_turns}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())