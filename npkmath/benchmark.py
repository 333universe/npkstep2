"""
Ground-truth benchmark for the verifier.

Each claim has a human-known label. We score the engine against it and report
the errors that matter most for AI use:

  critical        PROVED on a non-true claim, or REFUTED on a true claim
  false_assurance SUPPORTED on a false / domain-dependent claim

Run:  python -m npkmath.benchmark
"""
import json
import sys
from collections import Counter
from typing import Any, Dict, List

from .engine import NPKCoreEngine

R = lambda t: [["1", "0", "0"], ["0", f"cos({t})", f"-sin({t})"], ["0", f"sin({t})", f"cos({t})"]]

# (id, lhs, rhs, truth)  truth in TRUE / FALSE / CONDITIONAL / ILL_FORMED
EQUALITY: List[tuple] = [
    ("eq-pythagoras", "sin(x)**2+cos(x)**2", "1", "TRUE"),
    ("eq-square-expand", "(x+1)**2", "x**2+2*x+1", "TRUE"),
    ("eq-diff-squares", "(x-y)*(x+y)", "x**2-y**2", "TRUE"),
    ("eq-exp-add", "exp(x)*exp(y)", "exp(x+y)", "TRUE"),
    ("eq-sin-double", "sin(2*x)", "2*sin(x)*cos(x)", "TRUE"),
    ("eq-cos-double", "cos(2*x)", "1-2*sin(x)**2", "TRUE"),
    ("eq-tan-ratio", "tan(x)", "sin(x)/cos(x)", "TRUE"),
    ("eq-cancel", "(x**2-1)/(x-1)", "x+1", "TRUE"),
    ("eq-sqrt-abs", "sqrt(x**2)", "Abs(x)", "TRUE"),
    ("eq-diff-cubes", "x**3-y**3", "(x-y)*(x**2+x*y+y**2)", "TRUE"),
    ("eq-sinh-def", "sinh(x)", "(exp(x)-exp(-x))/2", "TRUE"),
    ("eq-hyp-pyth", "cosh(x)**2-sinh(x)**2", "1", "TRUE"),
    ("eq-log-exp-real", "log(exp(x))", "x", "TRUE"),
    ("eq-sin-sum", "sin(x+y)", "sin(x)*cos(y)+cos(x)*sin(y)", "TRUE"),
    ("eq-cube-expand", "(x+y)**3", "x**3+3*x**2*y+3*x*y**2+y**3", "TRUE"),
    ("eq-euler", "exp(I*x)", "cos(x)+I*sin(x)", "TRUE"),
    ("eq-fourth-powers", "sin(x)**4-cos(x)**4", "sin(x)**2-cos(x)**2", "TRUE"),
    ("eq-log-product", "log(x*y)", "log(x)+log(y)", "CONDITIONAL"),
    ("eq-sqrt-square", "sqrt(x**2)", "x", "CONDITIONAL"),
    ("eq-atan-recip", "atan(x)+atan(1/x)", "pi/2", "CONDITIONAL"),
    ("eq-binomial-wrong", "(x+y)**2", "x**2+y**2", "FALSE"),
    ("eq-sin-sum-wrong", "sin(x+y)", "sin(x)+sin(y)", "FALSE"),
    ("eq-sqrt-sum-wrong", "sqrt(x+y)", "sqrt(x)+sqrt(y)", "FALSE"),
    ("eq-log-sum-wrong", "log(x+y)", "log(x)+log(y)", "FALSE"),
    ("eq-sin-cos-minus", "sin(x)**2-cos(x)**2", "1", "FALSE"),
    ("eq-exp-sum-wrong", "exp(x+y)", "exp(x)+exp(y)", "FALSE"),
    ("eq-near-miss-1e30", "x**2+10**(-30)", "x**2", "FALSE"),
    ("eq-cos-wrong", "cos(x)**2", "1-cos(x)**2", "FALSE"),
    ("eq-cube-diff-wrong", "(x-y)**3", "x**3-y**3", "FALSE"),
    ("eq-recip-sum-wrong", "1/(x+y)", "1/x+1/y", "FALSE"),
    ("eq-tan-sum-wrong", "tan(x)+tan(y)", "tan(x+y)", "FALSE"),
    ("eq-sin-small-angle", "sin(x)", "x", "FALSE"),
    ("eq-square-double", "x**2", "2*x", "FALSE"),
    ("eq-exp-square-wrong", "exp(x)**2", "exp(x**2)", "FALSE"),
    ("eq-hyp-plus-wrong", "sinh(x)**2+cosh(x)**2", "1", "FALSE"),
    ("eq-sin-triple-wrong", "sin(3*x)", "3*sin(x)", "FALSE"),
    ("eq-malformed", "x +* 2", "3", "ILL_FORMED"),
    ("eq-injection", "__import__('os').system('echo hi')", "1", "ILL_FORMED"),
]

# (id, operation, A, B, truth)
MATRIX: List[tuple] = [
    ("mx-rot-x-orthogonal", "orthogonal", R("t"), None, "TRUE"),
    ("mx-identity-orthogonal", "orthogonal", [["1", "0"], ["0", "1"]], None, "TRUE"),
    ("mx-1234-orthogonal", "orthogonal", [["1", "2"], ["3", "4"]], None, "FALSE"),
    ("mx-rot2d-orthogonal", "orthogonal", [["cos(t)", "-sin(t)"], ["sin(t)", "cos(t)"]], None, "TRUE"),
    ("mx-reflect-orthogonal", "orthogonal", [["cos(t)", "sin(t)"], ["sin(t)", "-cos(t)"]], None, "TRUE"),
    ("mx-scaled-rot-orthogonal", "orthogonal", [["2*cos(t)", "-2*sin(t)"], ["2*sin(t)", "2*cos(t)"]], None, "FALSE"),
    ("mx-sym-true", "symmetric", [["a", "b"], ["b", "c"]], None, "TRUE"),
    ("mx-sym-false", "symmetric", [["a", "b"], ["c", "d"]], None, "FALSE"),
    ("mx-sym-nonsquare", "symmetric", [["1", "2", "3"], ["4", "5", "6"]], None, "ILL_FORMED"),
    ("mx-inverse-true", "inverse", [["2", "1"], ["1", "1"]], [["1", "-1"], ["-1", "2"]], "TRUE"),
    ("mx-inverse-false", "inverse", [["1", "2"], ["3", "4"]], [["1", "2"], ["3", "4"]], "FALSE"),
    ("mx-commute-false", "commute", [["1", "2"], ["3", "4"]], [["0", "1"], ["1", "0"]], "FALSE"),
    ("mx-commute-diag", "commute", [["a", "0"], ["0", "b"]], [["c", "0"], ["0", "d"]], "TRUE"),
    ("mx-equal-true", "equal", [["1", "0"], ["0", "1"]], [["1", "0"], ["0", "1"]], "TRUE"),
    ("mx-equal-false", "equal", [["x", "0"], ["0", "x"]], [["x", "0"], ["0", "x**2"]], "FALSE"),
]

# truth -> engine status -> grade
GRADES = {
    "TRUE":        {"PROVED": "full", "SUPPORTED": "partial", "REFUTED": "critical"},
    "FALSE":       {"REFUTED": "full", "PROVED": "critical", "SUPPORTED": "false_assurance"},
    "CONDITIONAL": {"CONDITIONAL": "full", "REFUTED": "partial", "PROVED": "critical", "SUPPORTED": "false_assurance"},
    "ILL_FORMED":  {"ILL_FORMED": "full"},
}


def grade(truth: str, status: str) -> str:
    return GRADES[truth].get(status, "miss")


def run_benchmark(engine: NPKCoreEngine = None) -> Dict[str, Any]:
    engine = engine or NPKCoreEngine()
    rows: List[Dict[str, Any]] = []
    for cid, lhs, rhs, truth in EQUALITY:
        res = engine.evaluate_symbolic_equality(lhs, rhs)
        rows.append({"id": cid, "kind": "equality", "truth": truth, "status": res["status"],
                     "grade": grade(truth, res["status"]), "ms": res["execution_time_ms"]})
    for cid, op, a, b, truth in MATRIX:
        res = engine.matrix_verify(op, a, b)
        rows.append({"id": cid, "kind": "matrix", "truth": truth, "status": res["status"],
                     "grade": grade(truth, res["status"]), "ms": res["execution_time_ms"]})
    g = Counter(r["grade"] for r in rows)
    n = len(rows)
    proved = [r for r in rows if r["status"] == "PROVED"]
    refuted = [r for r in rows if r["status"] == "REFUTED"]
    summary = {
        "claims": n,
        "full": g["full"], "partial": g["partial"], "miss": g["miss"],
        "critical_errors": g["critical"], "false_assurance": g["false_assurance"],
        "exact_rate": round(g["full"] / n, 3),
        # measured reliability of each verdict type (use as source_reliability)
        "precision_PROVED": round(sum(r["truth"] == "TRUE" for r in proved) / len(proved), 3) if proved else None,
        "precision_REFUTED": round(sum(r["truth"] == "FALSE" for r in refuted) / len(refuted), 3) if refuted else None,
        "max_ms": max(r["ms"] for r in rows),
    }
    return {"summary": summary, "rows": rows}


def main() -> int:
    report = run_benchmark()
    s = report["summary"]
    print("NPKmath verifier benchmark")
    print("-" * 44)
    for k, v in s.items():
        print(f"{k:>20}: {v}")
    bad = [r for r in report["rows"] if r["grade"] not in ("full",)]
    if bad:
        print("\nNot a full match:")
        for r in bad:
            print(f"  [{r['grade']:>15}] {r['id']:<28} truth={r['truth']:<11} got={r['status']}")
    with open("benchmark_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("\nSaved benchmark_report.json")
    return 1 if (s["critical_errors"] or s["false_assurance"]) else 0


if __name__ == "__main__":
    sys.exit(main())
