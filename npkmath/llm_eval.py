"""
LLM-in-the-loop evaluation.

Question: does giving a model verified knowledge (from the signals layer) make it
more accurate on math claims? This module measures that with real outcomes
instead of hand-written feedback.

Two ways to get answers from a model (any model, any vendor):
  1. API:    pass ask(prompt) -> str   to collect_answers(...)
  2. Manual: export_batches(...) writes prompts you paste into any chat app;
             paste its reply back and parse with parse_batch(...)

Conditions:
  baseline  the claim alone
  grounded  the claim plus verified signals about OTHER, related claims

By default the grounded context never contains the target claim or anything from
its own family, so it cannot leak the answer. Use exclude_family=False only to
simulate "look it up in a verified knowledge base" (that is leaky by design).
"""
import math
import random
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .dataset import Item, LABEL
from .engine import NPKCoreEngine
from .pipeline import verify_equality_and_emit, verify_matrix_and_emit
from .signals import AIFeedbackInterface, VerificationSignal

ANSWERS = ("TRUE", "FALSE", "CONDITIONAL")

RULES = (
    "Treat all variables and symbols as real numbers. Ignore isolated exceptional points "
    "(such as division by zero). If a side is not a real number over a whole range of "
    "values (for example log or sqrt of a negative number) the claim fails there.\n"
    "Answer TRUE if the statement holds for every real value. "
    "Answer CONDITIONAL if it holds for positive values but fails for some negative values. "
    "Answer FALSE otherwise."
)

_FUNCS = ("sinh", "cosh", "sin", "cos", "tan", "atan", "exp", "log", "sqrt", "Abs")


# ------------------------------------------------------------------------ prompts
def make_prompt(item: Item, context: Optional[str] = None) -> str:
    parts = [RULES]
    if context:
        parts.append("Machine-verified facts you may use:\n" + context)
    parts.append(f"CLAIM [{item.id}]: {item.text}")
    parts.append("Reply with exactly one word: TRUE, FALSE or CONDITIONAL.")
    return "\n\n".join(parts)


def make_batch_prompt(items: Sequence[Item], contexts: Optional[Dict[str, str]] = None,
                      start: int = 1) -> Tuple[str, Dict[str, str]]:
    """One prompt for many claims. Returns (prompt, {Qnn: item_id})."""
    qmap, lines = {}, []
    for i, it in enumerate(items, start):
        q = f"Q{i:02d}"
        qmap[q] = it.id
        lines.append(f"{q}: {it.text}")
        if contexts and contexts.get(it.id):
            ctx = " | ".join(l[2:] if l.startswith("- ") else l
                             for l in contexts[it.id].splitlines() if l.strip())
            lines.append(f"   verified facts: {ctx}")
    prompt = (
        RULES + "\n\nFor each numbered claim answer with one word. "
        "Reply only with lines in the form 'Q01: TRUE'.\n\n" + "\n".join(lines)
    )
    return prompt, qmap


def parse_answer(text: str) -> Optional[str]:
    m = re.search(r"\b(TRUE|FALSE|CONDITIONAL)\b", text or "", re.IGNORECASE)
    return m.group(1).upper() if m else None


def parse_batch(text: str, qmap: Dict[str, str]) -> Dict[str, str]:
    """Parse 'Q01: TRUE' style lines into {item_id: answer}. Unparseable lines are skipped."""
    out: Dict[str, str] = {}
    for m in re.finditer(r"\b(Q\d{2,3})\b\W{0,6}\**\s*(TRUE|FALSE|CONDITIONAL)\b", text or "", re.IGNORECASE):
        q = m.group(1).upper()
        if q in qmap:
            out[qmap[q]] = m.group(2).upper()
    return out


# ----------------------------------------------------------------------- context
def _tokens(item: Item) -> set:
    if item.kind == "matrix":
        return {"matrix:" + item.payload["op"]}
    return {f for f in _FUNCS if re.search(rf"\b{f}\b", item.text)}


class ContextBuilder:
    """Verifies every item once with the engine, then builds per-target context."""

    def __init__(self, items: Sequence[Item], engine: Optional[NPKCoreEngine] = None,
                 k: int = 4, seed: int = 0, exclude_family: bool = True,
                 source_reliability: float = 1.0):
        self.items, self.k, self.seed, self.exclude_family = list(items), k, seed, exclude_family
        self.engine = engine or NPKCoreEngine()
        self._fb = AIFeedbackInterface()
        self.signals: Dict[str, VerificationSignal] = {}
        for it in self.items:
            if it.kind == "equality":
                sig = verify_equality_and_emit(self.engine, self._fb, it.id, it.payload["lhs"],
                                               it.payload["rhs"], source_reliability)
            else:
                sig = verify_matrix_and_emit(self.engine, self._fb, it.id, it.payload["op"],
                                             it.payload["a"], it.payload["b"], source_reliability,
                                             rationale=it.text, claim_text=it.text)
            self.signals[it.id] = sig

    def related(self, target: Item) -> List[Item]:
        rng = random.Random(f"{self.seed}:{target.id}")
        mine = _tokens(target)
        cands = []
        for it in self.items:
            if it.id == target.id or (self.exclude_family and it.family == target.family):
                continue
            if self.signals[it.id].status not in ("PROVED", "REFUTED", "CONDITIONAL"):
                continue  # only decisive verdicts are offered as knowledge
            cands.append((len(mine & _tokens(it)), rng.random(), it))
        cands.sort(key=lambda t: (-t[0], t[1]))
        return [c[2] for c in cands[: self.k]]

    def context_for(self, target: Item, style: str = "compact") -> str:
        related = self.related(target)
        if style == "full":  # the signals module's own prompt export (long)
            fb = AIFeedbackInterface()
            fb.signals.extend(self.signals[it.id] for it in related)
            return fb.export_for_prompt(max_signals=self.k)
        return compact_context([self.signals[it.id] for it in related])

    def contexts(self, items: Optional[Sequence[Item]] = None) -> Dict[str, str]:
        return {it.id: self.context_for(it) for it in (items or self.items)}


def compact_context(signals: Sequence[VerificationSignal]) -> str:
    """One short line per verified fact (built from the signals, ~80 chars each)."""
    lines = []
    for sig in signals:
        claim = sig.metadata.get("claim") or sig.claim_id
        if sig.status == "PROVED":
            lines.append(f"- TRUE (proved): {claim}")
        elif sig.status == "REFUTED":
            ce = (sig.metadata.get("counterexample") or {}).get("point", {})
            at = ", ".join(f"{k}={v:.3g}" for k, v in ce.items())
            lines.append(f"- FALSE (fails at {at}): {claim}" if at else f"- FALSE: {claim}")
        elif sig.status == "CONDITIONAL":
            lines.append(f"- CONDITIONAL (positive values only): {claim}")
    return "\n".join(lines)


# ------------------------------------------------------------------- collecting
def collect_answers(items: Sequence[Item], ask: Callable[[str], str],
                    contexts: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Ask a model about each claim (API mode). Unparseable replies are omitted."""
    out = {}
    for it in items:
        ans = parse_answer(ask(make_prompt(it, (contexts or {}).get(it.id))))
        if ans:
            out[it.id] = ans
    return out


def export_batches(items: Sequence[Item], contexts: Optional[Dict[str, str]] = None,
                   batch_size: int = 20) -> List[Tuple[str, Dict[str, str]]]:
    """Split into copy-paste sized batches: [(prompt, {Qnn: item_id}), ...]."""
    out = []
    for i in range(0, len(items), batch_size):
        out.append(make_batch_prompt(items[i:i + batch_size], contexts))
    return out


# ------------------------------------------------------------------------ scoring
def score(items: Sequence[Item], answers: Dict[str, str], condition: str) -> List[Dict]:
    rows = []
    for it in items:
        ans = answers.get(it.id)
        rows.append({
            "id": it.id, "condition": condition, "truth": it.truth, "answer": ans,
            "correct": ans == it.truth if ans else False,
            "unparsed": ans is None,
            # the dangerous direction: model says TRUE about a claim that is not TRUE
            "false_assurance": ans == "TRUE" and it.truth != "TRUE",
        })
    return rows


def accuracy(rows: Sequence[Dict]) -> float:
    return round(sum(r["correct"] for r in rows) / len(rows), 3) if rows else 0.0


def _sign_test_p(gained: int, lost: int) -> float:
    n = gained + lost
    if n == 0:
        return 1.0
    k = min(gained, lost)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return round(min(1.0, 2 * tail), 4)


def compare(base_rows: Sequence[Dict], other_rows: Sequence[Dict]) -> Dict:
    """Paired comparison on the same items: who got right what the other got wrong."""
    b = {r["id"]: r for r in base_rows}
    o = {r["id"]: r for r in other_rows}
    ids = sorted(set(b) & set(o))
    gained = sum((not b[i]["correct"]) and o[i]["correct"] for i in ids)
    lost = sum(b[i]["correct"] and (not o[i]["correct"]) for i in ids)
    return {
        "n": len(ids),
        "baseline_accuracy": accuracy([b[i] for i in ids]),
        "other_accuracy": accuracy([o[i] for i in ids]),
        "gained": gained, "lost": lost, "net": gained - lost,
        "sign_test_p": _sign_test_p(gained, lost),
        "baseline_false_assurance": sum(b[i]["false_assurance"] for i in ids),
        "other_false_assurance": sum(o[i]["false_assurance"] for i in ids),
        "note": "p is a rough guide only; small samples and one model prove little.",
    }


def summarize(items: Sequence[Item], rows_by_condition: Dict[str, Sequence[Dict]]) -> Dict:
    """
    One-stop report. Always shows the 'always give the commonest answer' floor, because
    accuracy means little without it. Compares every condition to 'baseline' if present.
    """
    from .dataset import majority_baseline
    out = {"n_items": len(items), "majority_baseline": majority_baseline(list(items)),
           "conditions": {}, "comparisons": {}}
    for cond, rows in rows_by_condition.items():
        out["conditions"][cond] = {
            "accuracy": accuracy(rows),
            "false_assurance": sum(r["false_assurance"] for r in rows),
            "unparsed": sum(r["unparsed"] for r in rows),
        }
    if "baseline" in rows_by_condition:
        for cond, rows in rows_by_condition.items():
            if cond != "baseline":
                out["comparisons"][f"baseline->{cond}"] = compare(rows_by_condition["baseline"], rows)
    return out


def log_feedback(fb: AIFeedbackInterface, rows: Sequence[Dict], model_name: str) -> int:
    """Write MEASURED outcomes to the feedback log (one entry per answered claim)."""
    for r in rows:
        outcome = "unparsed" if r["unparsed"] else ("correct" if r["correct"] else "incorrect")
        fb.log_ai_feedback(
            signal_id=r["id"],
            ai_decision=f"[{model_name}/{r['condition']}] answered {r['answer']}",
            outcome=outcome,
            metadata={"condition": r["condition"], "truth": r["truth"],
                      "answer": r["answer"], "false_assurance": r["false_assurance"]},
        )
    return len(rows)


# ------------------------------------------------- the live loop (not a benchmark)
def answer_with_verification(ask: Callable[[str], str], engine: NPKCoreEngine,
                             fb: AIFeedbackInterface, claim_id: str, lhs: str, rhs: str,
                             model_name: str = "model") -> Dict:
    """
    Ask a model, then check it with the engine. Use this on NEW claims: the verifier
    overrides the model whenever it has a decisive verdict, and the feedback log
    records what really happened.
    """
    item = Item(claim_id, "equality", claim_id, f"{lhs} = {rhs}", "?", {"lhs": lhs, "rhs": rhs})
    model_ans = parse_answer(ask(make_prompt(item)))
    sig = verify_equality_and_emit(engine, fb, claim_id, lhs, rhs)
    engine_ans = LABEL.get(sig.status)  # None when the engine can't decide
    if engine_ans is None:
        final, outcome = model_ans, "unverifiable"
    elif model_ans == engine_ans:
        final, outcome = engine_ans, "agreed"
    else:
        final, outcome = engine_ans, "overridden_by_verifier"
    fb.log_ai_feedback(
        signal_id=claim_id,
        ai_decision=f"[{model_name}] answered {model_ans}; verifier says {sig.status}",
        outcome=outcome,
        metadata={"model_answer": model_ans, "engine_status": sig.status, "final": final},
    )
    return {"claim_id": claim_id, "model_answer": model_ans, "engine_status": sig.status,
            "final": final, "outcome": outcome}


# --------------------------------------------------- offline stand-in (testing only)
class MockModel:
    """
    A fake model for testing the harness offline. NOT evidence about any real AI.
    Answers correctly with probability `accuracy`; errors skew toward saying TRUE.
    """

    def __init__(self, items: Sequence[Item], accuracy: float = 0.8, seed: int = 0,
                 grounded_boost: float = 0.0):
        self.truth = {it.id: it.truth for it in items}
        self.acc, self.boost = accuracy, grounded_boost
        self.rng = random.Random(seed)

    def __call__(self, prompt: str) -> str:
        m = re.search(r"CLAIM \[(.+?)\]", prompt)
        truth = self.truth[m.group(1)]
        acc = min(1.0, self.acc + (self.boost if "Machine-verified facts" in prompt else 0.0))
        if self.rng.random() < acc:
            return truth
        wrong = [a for a in ANSWERS if a != truth]
        if truth != "TRUE" and self.rng.random() < 0.7:
            return "TRUE"
        return self.rng.choice(wrong)
