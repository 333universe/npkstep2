"""
Evaluation dataset for testing AI models against the verifier.

Base items come from the validated benchmark (human labels). Mutants are made by
corrupting a base identity (flip a sign, bump a number, swap sin/cos, ...) and are
labelled by the engine, keeping only decisive verdicts (PROVED / REFUTED /
CONDITIONAL). Plausible-looking-but-wrong claims are exactly where models slip.

Deterministic for a given seed.
"""
import random
import re

import sympy as sp
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .benchmark import EQUALITY, MATRIX
from .engine import NPKCoreEngine

LABEL = {"PROVED": "TRUE", "REFUTED": "FALSE", "CONDITIONAL": "CONDITIONAL"}


@dataclass
class Item:
    id: str
    kind: str            # "equality" | "matrix"
    family: str          # id of the base claim this came from
    text: str            # claim as shown to a model
    truth: str           # TRUE | FALSE | CONDITIONAL
    payload: Dict[str, Any] = field(default_factory=dict)
    source: str = "benchmark"   # "benchmark" | "mutant"


# --------------------------------------------------------------------- mutations
def _mut_sign(s: str, rng: random.Random) -> Optional[str]:
    idx = [i for i, c in enumerate(s) if c in "+-" and i > 0 and s[i - 1] not in "*(^e "]
    if not idx:
        return None
    i = rng.choice(idx)
    return s[:i] + ("-" if s[i] == "+" else "+") + s[i + 1:]


def _mut_number(s: str, rng: random.Random) -> Optional[str]:
    ms = list(re.finditer(r"(?<![\w.])\d+(?![\w.])", s))
    if not ms:
        return None
    m = rng.choice(ms)
    return s[:m.start()] + str(int(m.group()) + 1) + s[m.end():]


def _mut_trig(s: str, rng: random.Random) -> Optional[str]:
    ms = list(re.finditer(r"\b(sin|cos)\b", s))
    if not ms:
        return None
    m = rng.choice(ms)
    return s[:m.start()] + ("cos" if m.group() == "sin" else "sin") + s[m.end():]


def _mut_power(s: str, rng: random.Random) -> Optional[str]:
    ms = list(re.finditer(r"\*\*(\d)", s))
    if not ms:
        return None
    m = rng.choice(ms)
    return s[:m.start(1)] + str(int(m.group(1)) + 1) + s[m.end(1):]


def _mut_var(s: str, rng: random.Random) -> Optional[str]:
    if not re.search(r"\bx\b", s) or not re.search(r"\by\b", s):
        return None
    ms = list(re.finditer(r"\bx\b", s))
    m = rng.choice(ms)
    return s[:m.start()] + "y" + s[m.end():]


_MUTATORS = [_mut_sign, _mut_number, _mut_trig, _mut_power, _mut_var]


def _rewrites(lhs: str, rhs: str, engine: NPKCoreEngine) -> List[tuple]:
    """Meaning-preserving rewrites: same claim, different surface form (mostly TRUE items)."""
    out = [(rhs, lhs)]  # swap sides
    try:
        R = engine.parse(rhs)
        for fn in (sp.expand, sp.factor, sp.trigsimp):
            out.append((lhs, str(fn(R)).replace(" ", "")))
    except Exception:
        pass
    return out


def _mutate(rhs: str, rng: random.Random) -> Optional[str]:
    order = _MUTATORS[:]
    rng.shuffle(order)
    for fn in order:
        out = fn(rhs, rng)
        if out and out != rhs:
            return out
    return None


# ---------------------------------------------------------------------- builders
def _matrix_text(op: str, a, b) -> str:
    fmt = lambda m: "[" + ", ".join("[" + ", ".join(r) + "]" for r in m) + "]"
    stmt = {
        "orthogonal": "A^T * A = I (the identity)",
        "symmetric": "A^T = A",
        "inverse": "A * B = I (the identity)",
        "equal": "A = B",
        "commute": "A * B = B * A",
    }[op]
    s = f"A = {fmt(a)}" + (f", B = {fmt(b)}" if b else "")
    return f"{s}. Statement: {stmt}"


def build_dataset(seed: int = 0, mutants_per_base: int = 1, rewrites_per_base: int = 1,
                  max_items: int = 60, engine: Optional[NPKCoreEngine] = None) -> List[Item]:
    engine = engine or NPKCoreEngine()
    rng = random.Random(seed)
    items: List[Item] = []
    seen = set()

    for cid, lhs, rhs, truth in EQUALITY:
        if truth not in ("TRUE", "FALSE", "CONDITIONAL"):
            continue
        key = (lhs, rhs)
        seen.add(key)
        items.append(Item(cid, "equality", cid, f"{lhs} = {rhs}", truth,
                          {"lhs": lhs, "rhs": rhs}, "benchmark"))
        made = 0
        for _ in range(mutants_per_base * 4):
            if made >= mutants_per_base:
                break
            new_rhs = _mutate(rhs, rng)
            if not new_rhs or (lhs, new_rhs) in seen:
                continue
            res = engine.evaluate_symbolic_equality(lhs, new_rhs)
            if res["status"] not in LABEL:
                continue
            seen.add((lhs, new_rhs))
            made += 1
            items.append(Item(f"{cid}-m{made}", "equality", cid, f"{lhs} = {new_rhs}",
                              LABEL[res["status"]], {"lhs": lhs, "rhs": new_rhs}, "mutant"))

        rw = _rewrites(lhs, rhs, engine)
        rng.shuffle(rw)
        made_rw = 0
        for new_lhs, new_rhs in rw:
            if made_rw >= rewrites_per_base:
                break
            if (new_lhs, new_rhs) in seen or new_rhs.replace(" ", "") == rhs.replace(" ", "") and new_lhs == lhs:
                continue
            res = engine.evaluate_symbolic_equality(new_lhs, new_rhs)
            if res["status"] not in LABEL:
                continue
            seen.add((new_lhs, new_rhs))
            made_rw += 1
            items.append(Item(f"{cid}-r{made_rw}", "equality", cid, f"{new_lhs} = {new_rhs}",
                              LABEL[res["status"]], {"lhs": new_lhs, "rhs": new_rhs}, "rewrite"))

    for cid, op, a, b, truth in MATRIX:
        if truth not in ("TRUE", "FALSE"):
            continue
        items.append(Item(cid, "matrix", cid, _matrix_text(op, a, b), truth,
                          {"op": op, "a": a, "b": b}, "benchmark"))

    return _balanced_sample(items, max_items, rng)


def _balanced_sample(items: List[Item], max_items: int, rng: random.Random) -> List[Item]:
    """
    Keep the label mix near 42% TRUE / 46% FALSE / rest CONDITIONAL (as far as supply allows),
    so a model cannot score well by always giving the same answer.
    """
    if len(items) <= max_items:
        rng.shuffle(items)
        return items
    pools: Dict[str, List[Item]] = {"TRUE": [], "FALSE": [], "CONDITIONAL": []}
    for it in items:
        pools[it.truth].append(it)
    for g in pools.values():
        rng.shuffle(g)
    quota = {"TRUE": round(0.42 * max_items), "FALSE": round(0.46 * max_items)}
    quota["CONDITIONAL"] = max_items - quota["TRUE"] - quota["FALSE"]
    keep: List[Item] = []
    for t in ("TRUE", "FALSE", "CONDITIONAL"):
        keep.extend(pools[t][: quota[t]])
    # fill any shortfall (e.g. few CONDITIONAL items exist) from the larger pools
    leftovers = [it for t in ("TRUE", "FALSE", "CONDITIONAL") for it in pools[t][quota[t]:]]
    rng.shuffle(leftovers)
    keep.extend(leftovers[: max_items - len(keep)])
    rng.shuffle(keep)
    return keep


def majority_baseline(items: List[Item]) -> Dict[str, Any]:
    """Accuracy of always answering the most common label (the floor a model must beat)."""
    counts: Dict[str, int] = {}
    for it in items:
        counts[it.truth] = counts.get(it.truth, 0) + 1
    label = max(counts, key=counts.get)
    return {"label": label, "accuracy": round(counts[label] / len(items), 3), "counts": counts}
