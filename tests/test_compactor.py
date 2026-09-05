"""Tests for the context compactor."""

from forgeharness.context.compactor import Compactor, Turn


def _make_turn(i: int, words: int = 10, role: str = "user") -> Turn:
    return Turn(id=f"t{i}", role=role, content="word " * words)


def test_tracks_token_budget_and_triggers_at_threshold():
    compactor = Compactor(token_budget=100, trigger_ratio=0.5, keep_last=1)
    for i in range(4):
        compactor.add_turn(_make_turn(i, words=20))  # 80 tokens total
    assert compactor.total_tokens() == 80
    assert compactor.used_ratio() == 0.8
    assert compactor.should_compact() is True


def test_does_not_trigger_below_threshold():
    compactor = Compactor(token_budget=1000, trigger_ratio=0.8, keep_last=1)
    compactor.add_turn(_make_turn(0, words=10))
    assert compactor.should_compact() is False


def test_compaction_preserves_most_recent_k_verbatim():
    compactor = Compactor(token_budget=100, trigger_ratio=0.8, keep_last=2)
    for i in range(5):
        compactor.add_turn(_make_turn(i))
    result = compactor.compact()

    # the last two turns (t3, t4) survive verbatim (not summaries)
    surviving_ids = {t.id for t in result.turns if not t.summary}
    assert {"t3", "t4"} <= surviving_ids
    assert len(result.removed_turn_ids) == 3


def test_compaction_preserves_in_flight_turn():
    compactor = Compactor(token_budget=100, trigger_ratio=0.8, keep_last=1)
    for i in range(5):
        compactor.add_turn(_make_turn(i))
    compactor.mark_in_flight("t0")  # oldest turn is in-flight
    result = compactor.compact()

    assert "t0" in {t.id for t in result.turns}
    t0 = next(t for t in result.turns if t.id == "t0")
    assert t0.in_flight is True
    assert t0.summary is False
    # t0 must survive even though it's not in the recent window
    assert "t0" not in result.removed_turn_ids


def test_compaction_event_is_logged():
    compactor = Compactor(token_budget=100, trigger_ratio=0.8, keep_last=1)
    for i in range(5):
        compactor.add_turn(_make_turn(i, words=50))  # long turns so summary compresses
    compactor.compact()
    assert len(compactor.compaction_log) == 1
    evt = compactor.compaction_log[0]
    assert evt.tokens_before > evt.tokens_after
    assert evt.turns_compacted == 4
    assert evt.timestamp is not None


def test_compaction_produces_summary_turn():
    compactor = Compactor(token_budget=100, trigger_ratio=0.8, keep_last=1)
    for i in range(5):
        compactor.add_turn(_make_turn(i))
    result = compactor.compact()
    summaries = [t for t in result.turns if t.summary]
    assert len(summaries) == 1
    assert summaries[0].role == "system"
    assert result.summary_turn is not None


def test_compaction_noop_when_nothing_compactable():
    compactor = Compactor(token_budget=1000, trigger_ratio=0.8, keep_last=3)
    compactor.add_turn(_make_turn(0))
    result = compactor.compact()
    assert len(result.removed_turn_ids) == 0
    assert compactor.compaction_log == []


def test_maybe_compact_only_at_threshold():
    compactor = Compactor(token_budget=100, trigger_ratio=0.5, keep_last=1)
    compactor.add_turn(_make_turn(0, words=10))
    assert compactor.maybe_compact() is None
    compactor.add_turn(_make_turn(1, words=10))
    compactor.add_turn(_make_turn(2, words=10))
    compactor.add_turn(_make_turn(3, words=10))
    compactor.add_turn(_make_turn(4, words=10))
    assert compactor.maybe_compact() is not None
    assert len(compactor.compaction_log) == 1
