from datetime import datetime, timedelta, timezone

from npkmath.engine import NPKCoreEngine
from npkmath.pipeline import verify_equality_and_emit, verify_matrix_and_emit
from npkmath.signals import AIFeedbackInterface, SignalRanker


def _old(fb, status, claim_type="numeric", days=30):
    sig = fb.emit_signal("old-claim", claim_type, status, 0.9, 1.0)
    sig.timestamp = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    return sig


def test_old_signal_is_decayed_in_summary():
    # Regression: stale signals used to keep freshness 1.0 and count as high confidence.
    fb = AIFeedbackInterface()
    _old(fb, "VERIFIED", days=30)
    s = fb.get_summary()
    assert s["high_confidence_signals"] == 0
    assert s["fresh_signals_24h"] == 0
    assert s["average_effective_weight"] < 0.1


def test_proofs_do_not_decay():
    fb = AIFeedbackInterface()
    _old(fb, "PROVED", claim_type="symbolic", days=365)
    assert fb.get_summary()["high_confidence_signals"] == 1
    assert len(fb.get_fresh_signals()) == 1
    assert fb.get_fresh_signals(include_timeless=False) == []


def test_reading_summary_does_not_mutate_stored_signals():
    fb = AIFeedbackInterface()
    sig = _old(fb, "VERIFIED", days=3)
    before = sig.freshness_score
    fb.get_summary()
    fb.get_fresh_signals(max_age_hours=1000)
    assert sig.freshness_score == before


def test_refuted_goes_to_its_own_prompt_section():
    fb = AIFeedbackInterface()
    fb.emit_signal("true-1", "symbolic", "PROVED", 1.0, 1.0, rationale="r1")
    fb.emit_signal("false-1", "symbolic", "REFUTED", 1.0, 1.0, rationale="r2")
    text = fb.export_for_prompt()
    verified, refuted = text.split("## Refuted Claims")
    assert "true-1" in verified and "false-1" not in verified
    assert "false-1" in refuted and "do not assert" in text


def test_claim_holds_separates_verdict_from_confidence():
    fb = AIFeedbackInterface()
    assert fb.emit_signal("a", "symbolic", "PROVED", 1, 1).claim_holds is True
    assert fb.emit_signal("b", "symbolic", "REFUTED", 1, 1).claim_holds is False
    assert fb.emit_signal("c", "symbolic", "CONDITIONAL", 1, 1).claim_holds is None


def test_pipeline_emits_measured_values():
    e, fb = NPKCoreEngine(), AIFeedbackInterface()
    s1 = verify_equality_and_emit(e, fb, "t1", "sin(x)**2+cos(x)**2", "1")
    s2 = verify_matrix_and_emit(e, fb, "m1", "orthogonal", [["1", "2"], ["3", "4"]])
    assert s1.status == "PROVED" and s1.execution_time_ms > 0 and s1.precision_bits == 166
    assert s2.status == "REFUTED" and s2.claim_holds is False
    assert s2.metadata["method"] == "numeric-counterexample"


def test_refuted_with_counterexample_is_fully_confident():
    fb = AIFeedbackInterface()
    with_ce = fb.emit_signal("a", "symbolic", "REFUTED", 1, 1, metadata={"counterexample": {"point": {"x": 1.0}}})
    without = fb.emit_signal("b", "symbolic", "REFUTED", 1, 1)
    assert with_ce.confidence == 1.0 and without.confidence == 0.95
