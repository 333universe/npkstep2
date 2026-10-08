from npkmath.dataset import build_dataset, majority_baseline
from npkmath.engine import NPKCoreEngine
from npkmath.llm_eval import (
    ContextBuilder, MockModel, accuracy, answer_with_verification, collect_answers, compare,
    export_batches, log_feedback, make_prompt, parse_answer, parse_batch, score, summarize,
    compact_context,
)
from npkmath.signals import AIFeedbackInterface

ENGINE = NPKCoreEngine()
ITEMS = build_dataset(seed=0, max_items=40, engine=ENGINE)


def test_dataset_is_deterministic_and_capped():
    again = build_dataset(seed=0, max_items=40, engine=ENGINE)
    assert [i.id for i in ITEMS] == [i.id for i in again]
    assert len(ITEMS) <= 40
    assert len({i.id for i in ITEMS}) == len(ITEMS)


def test_dataset_labels_are_valid_and_diverse():
    assert {i.truth for i in ITEMS} <= {"TRUE", "FALSE", "CONDITIONAL"}
    assert {"TRUE", "FALSE"} <= {i.truth for i in ITEMS}


def test_mutants_are_engine_labelled_and_different_from_base():
    full = build_dataset(seed=1, max_items=500, engine=ENGINE)
    mutants = [i for i in full if i.source == "mutant"]
    assert len(mutants) >= 10
    for m in mutants:
        base = next(i for i in full if i.id == m.family)
        assert m.payload["rhs"] != base.payload["rhs"]
        res = ENGINE.evaluate_symbolic_equality(m.payload["lhs"], m.payload["rhs"])
        assert {"PROVED": "TRUE", "REFUTED": "FALSE", "CONDITIONAL": "CONDITIONAL"}[res["status"]] == m.truth


def test_parse_answer_and_batch():
    assert parse_answer("**False**, because...") == "FALSE"
    assert parse_answer("no idea") is None
    qmap = {"Q01": "a", "Q02": "b", "Q03": "c"}
    text = "Q01: TRUE\nQ02 - false\n**Q03:** conditional\nQ09: TRUE"
    assert parse_batch(text, qmap) == {"a": "TRUE", "b": "FALSE", "c": "CONDITIONAL"}


def test_grounded_context_never_leaks_target_or_family():
    cb = ContextBuilder(ITEMS, ENGINE, k=4)
    for it in ITEMS:
        rel = cb.related(it)
        assert len(rel) <= 4
        assert all(r.id != it.id and r.family != it.family for r in rel)
    ctx = cb.context_for(ITEMS[0])
    assert ctx and all(l.startswith("- ") for l in ctx.splitlines())
    assert ITEMS[0].text not in ctx
    assert "Verified Knowledge" in cb.context_for(ITEMS[0], style="full") or "Refuted" in cb.context_for(ITEMS[0], style="full")


def test_context_only_offers_decisive_verdicts():
    cb = ContextBuilder(ITEMS, ENGINE, k=6)
    for it in ITEMS:
        assert all(cb.signals[r.id].status in ("PROVED", "REFUTED", "CONDITIONAL") for r in cb.related(it))


def test_scoring_and_perfect_vs_worst_model():
    perfect = collect_answers(ITEMS, MockModel(ITEMS, accuracy=1.0))
    assert accuracy(score(ITEMS, perfect, "baseline")) == 1.0
    worst = collect_answers(ITEMS, MockModel(ITEMS, accuracy=0.0))
    assert accuracy(score(ITEMS, worst, "baseline")) == 0.0


def test_unparsed_answers_count_as_incorrect():
    rows = score(ITEMS, {}, "baseline")
    assert accuracy(rows) == 0.0 and all(r["unparsed"] for r in rows)


def test_compare_counts_and_sign_test():
    base = [{"id": str(i), "correct": False, "false_assurance": False} for i in range(8)]
    other = [{"id": str(i), "correct": True, "false_assurance": False} for i in range(8)]
    c = compare(base, other)
    assert (c["gained"], c["lost"], c["net"]) == (8, 0, 8)
    assert abs(c["sign_test_p"] - 2 / 256) < 1e-4
    same = compare(base, base)
    assert same["net"] == 0 and same["sign_test_p"] == 1.0


def test_false_assurance_flag():
    items = [i for i in ITEMS if i.truth == "FALSE"][:3]
    rows = score(items, {i.id: "TRUE" for i in items}, "baseline")
    assert all(r["false_assurance"] for r in rows)


def test_harness_detects_a_helpful_context():
    # Mock gets +0.3 accuracy when grounded: harness should see a positive net gain.
    mock = MockModel(ITEMS, accuracy=0.6, seed=3, grounded_boost=0.35)
    cb = ContextBuilder(ITEMS, ENGINE)
    base = score(ITEMS, collect_answers(ITEMS, mock), "baseline")
    grounded = score(ITEMS, collect_answers(ITEMS, mock, cb.contexts()), "grounded")
    c = compare(base, grounded)
    assert c["net"] > 0 and c["other_accuracy"] > c["baseline_accuracy"]


def test_feedback_log_records_measured_outcomes():
    fb = AIFeedbackInterface()
    rows = score(ITEMS, collect_answers(ITEMS, MockModel(ITEMS, accuracy=0.5, seed=1)), "baseline")
    n = log_feedback(fb, rows, "mock")
    assert n == len(ITEMS) == len(fb.feedback_log)
    assert {e["outcome"] for e in fb.feedback_log} <= {"correct", "incorrect", "unparsed"}
    assert sum(e["outcome"] == "correct" for e in fb.feedback_log) == sum(r["correct"] for r in rows)


def test_batch_export_round_trip():
    batches = export_batches(ITEMS, batch_size=15)
    assert len(batches) == -(-len(ITEMS) // 15)
    prompt, qmap = batches[0]
    assert "Q01:" in prompt and len(qmap) == min(15, len(ITEMS))
    reply = "\n".join(f"{q}: TRUE" for q in qmap)
    got = parse_batch(reply, qmap)
    assert set(got) == set(qmap.values())


def test_prompt_contains_claim_marker_and_context():
    p = make_prompt(ITEMS[0], context="CTX-TEXT")
    assert f"CLAIM [{ITEMS[0].id}]" in p and "CTX-TEXT" in p


def test_answer_with_verification_overrides_wrong_model():
    fb = AIFeedbackInterface()
    liar = lambda prompt: "TRUE"
    r = answer_with_verification(liar, ENGINE, fb, "live-1", "(x+y)**2", "x**2+y**2")
    assert r["final"] == "FALSE" and r["outcome"] == "overridden_by_verifier"
    ok = answer_with_verification(liar, ENGINE, fb, "live-2", "sin(x)**2+cos(x)**2", "1")
    assert ok["final"] == "TRUE" and ok["outcome"] == "agreed"
    assert [e["outcome"] for e in fb.feedback_log] == ["overridden_by_verifier", "agreed"]


def test_dataset_is_balanced_so_constant_answers_score_poorly():
    items = build_dataset(seed=0, max_items=60, engine=ENGINE)
    assert majority_baseline(items)["accuracy"] <= 0.55
    assert {i.source for i in items} >= {"benchmark", "rewrite"}


def test_rewrites_keep_meaning_and_differ_in_form():
    full = build_dataset(seed=2, max_items=500, engine=ENGINE)
    rw = [i for i in full if i.source == "rewrite"]
    assert len(rw) >= 15
    base = {i.id: i for i in full}
    for r in rw:
        b = base[r.family]
        assert r.text != b.text
        assert r.truth == b.truth or b.truth == "FALSE" or b.truth == "CONDITIONAL"


def test_compact_context_lines_are_short_and_readable():
    cb = ContextBuilder(ITEMS, ENGINE, k=4)
    ctx = cb.context_for(ITEMS[1])
    assert max(len(l) for l in ctx.splitlines()) < 160
    assert all(("TRUE" in l or "FALSE" in l or "CONDITIONAL" in l) for l in ctx.splitlines())
    assert "mx-" not in ctx  # matrix facts are shown as text, not internal ids


def test_summarize_reports_floor_and_comparison():
    mock = MockModel(ITEMS, accuracy=0.6, seed=5, grounded_boost=0.3)
    cb = ContextBuilder(ITEMS, ENGINE)
    rows = {"baseline": score(ITEMS, collect_answers(ITEMS, mock), "baseline"),
            "grounded": score(ITEMS, collect_answers(ITEMS, mock, cb.contexts()), "grounded")}
    rep = summarize(ITEMS, rows)
    assert rep["majority_baseline"]["accuracy"] <= 0.55
    assert set(rep["conditions"]) == {"baseline", "grounded"}
    assert "baseline->grounded" in rep["comparisons"]
