"""Unit tests for the deterministic P2 retrieval diagnostic."""

from experiments.retrieval_diagnostic import (
    _answer_fact_terms,
    _is_retrievable,
    _turn_precision,
    compute_retrieval_vs_fixture,
)

# A minimal fixture with a known query -> answer mapping and prior context.
FIXTURE = [
    {
        "conversation_id": "test_conv",
        "turns": [
            {"role": "user", "content": "How should we handle session store?"},
            {"role": "assistant", "content": "Use Redis with TTL for session store."},
            {"role": "user", "content": "How should we handle pagination?"},
            {"role": "assistant", "content": "Use cursor-based for pagination."},
            # Repeated ask: the answer now exists in prior history.
            {"role": "user", "content": "Can we change how pagination works?"},
            {"role": "assistant", "content": "Use cursor-based for pagination."},
        ],
    }
]

# First-mention turn: session-store answer cannot exist in history.
FIRST_MENTION_RECORD = {
    "conversation_id": "test_conv",
    "turns": [
        {"turn": 1, "query": "How should we handle session store?",
         "assembled_content": "Use Redis with TTL for session store."},
    ],
}

# Retrievable turn: pagination was answered before it is asked again.
RETRIEVABLE_RECORD = {
    "conversation_id": "test_conv",
    "turns": [
        {"turn": 2, "query": "Can we change how pagination works?",
         "assembled_content": "Use cursor-based for pagination. Use Redis with TTL."},
    ],
}

MISS_RECORD = {
    "conversation_id": "test_conv",
    "turns": [
        {"turn": 3, "query": "Can we change how pagination works?",
         "assembled_content": "Only unrelated content about the order schema."},
    ],
}


def test_answer_fact_terms_excludes_query_words():
    facts = _answer_fact_terms(
        "How should we handle session store?",
        "Use Redis with TTL for session store.",
    )
    # "session" and "store" appear in the query, so only the answer's additions
    # are facts; "ttl" is below the >=4-char word filter.
    assert "redis" in facts
    assert "session" not in facts
    assert "ttl" not in facts


def test_is_retrievable_distinguishes_first_mention():
    assert not _is_retrievable(
        "How should we handle session store?", ""  # no prior context
    )
    assert _is_retrievable(
        "Can we change how pagination works?",
        "Use Redis with TTL for session store. Use cursor-based for pagination.",
    )


def test_turn_precision_proxy():
    # All sentences share the topic term "pagination".
    assert _turn_precision(
        "How should we handle pagination?",
        "Use cursor-based for pagination. Pagination is important.",
    ) == 1.0
    # None share a topic term.
    assert _turn_precision(
        "How should we handle pagination?",
        "Only unrelated content about the order schema.",
    ) == 0.0
    # Empty assembled content -> None (not measurable).
    assert _turn_precision("How should we handle pagination?", "") is None


def test_compute_retrieval_recall_full_hit():
    result = compute_retrieval_vs_fixture(
        [RETRIEVABLE_RECORD, FIRST_MENTION_RECORD], FIXTURE
    )
    assert result["sampled_turns"] == 2
    assert result["retrievable_turns"] == 1
    assert result["first_mention_turns"] == 1
    assert result["retrieval_recall_retrievable"] == 100.0


def test_compute_retrieval_recall_miss():
    result = compute_retrieval_vs_fixture([MISS_RECORD], FIXTURE)
    assert result["retrieval_recall"] == 0.0
    assert result["retrieval_recall_retrievable"] == 0.0


def test_compute_retrieval_no_answer_skipped():
    # A turn whose query has no fixture answer is not sampled at all.
    result = compute_retrieval_vs_fixture(
        [{"conversation_id": "test_conv", "turns": [
            {"turn": 9, "query": "Something with no fixture answer.",
             "assembled_content": "whatever"}]}],
        FIXTURE,
    )
    assert result["sampled_turns"] == 0