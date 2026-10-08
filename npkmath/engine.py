"""
NPKmath verification engine.

Turns a mathematical claim into an honest verdict:

    PROVED       simplifies to exactly zero symbolically (and agrees numerically)
    REFUTED      a concrete numeric counterexample exists on positive reals
    CONDITIONAL  holds on positive reals but fails elsewhere (domain-dependent),
                 or symbolic and numeric checks disagree, or too few valid points
    SUPPORTED    numerically equal at every sample point, but NOT proved
    ILL_FORMED   could not be parsed / wrong shape

Every REFUTED verdict carries the counterexample that justifies it, so the
verdict can be re-checked by hand. Matrix checks read their inputs (nothing is
hard-coded).

Note: expressions are parsed with sympy's parser (which uses eval internally),
so a character whitelist rejects anything that is not plain math. Still, only
verify claims you trust the origin of.
"""

import random
import re
import signal as _signal
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)
_ALLOWED = re.compile(r"^[A-Za-z0-9_\s\+\-\*/\^\(\)\.,]*$")
_MAX_LEN = 400


class _Timeout(Exception):
    pass


def _run_with_timeout(seconds: int, fn, *args):
    """Run fn with a wall-clock limit (Unix main thread); otherwise run plainly."""
    if not hasattr(_signal, "SIGALRM"):
        return fn(*args)
    try:
        def handler(signum, frame):
            raise _Timeout()
        old = _signal.signal(_signal.SIGALRM, handler)
    except ValueError:  # not in main thread
        return fn(*args)
    _signal.alarm(seconds)
    try:
        return fn(*args)
    finally:
        _signal.alarm(0)
        _signal.signal(_signal.SIGALRM, old)


class NPKCoreEngine:
    def __init__(
        self,
        precision_digits: int = 50,
        samples: int = 12,
        seed: int = 0,
        simplify_timeout_s: int = 5,
    ):
        self.dps = precision_digits
        self.samples = samples
        self.seed = seed
        self.simplify_timeout_s = simplify_timeout_s
        self.min_valid_points = 4

    # ------------------------------------------------------------------ parsing
    def parse(self, text: str) -> sp.Expr:
        if not isinstance(text, str) or len(text) > _MAX_LEN or not _ALLOWED.match(text):
            raise ValueError("expression contains characters outside plain math")
        if "__" in text:
            raise ValueError("expression contains reserved sequence")
        return parse_expr(text, transformations=_TRANSFORMS, evaluate=True)

    # ------------------------------------------------------------ public checks
    def evaluate_symbolic_equality(self, expr1_str: str, expr2_str: str) -> Dict[str, Any]:
        t0 = time.perf_counter()
        try:
            L = self.parse(expr1_str)
            R = self.parse(expr2_str)
        except Exception as e:
            return self._result("ILL_FORMED", "parse", f"could not parse: {e}", t0)
        return self._classify([L - R], f"{expr1_str} == {expr2_str}", t0)

    def matrix_verify(
        self,
        operation: str,
        matrix_a: Sequence[Sequence[str]],
        matrix_b: Optional[Sequence[Sequence[str]]] = None,
    ) -> Dict[str, Any]:
        """
        operation:
          orthogonal  A^T * A == I
          symmetric   A^T == A
          inverse     A * B == I
          equal       A == B
          commute     A*B == B*A
        """
        t0 = time.perf_counter()
        try:
            A = self._to_matrix(matrix_a)
            B = self._to_matrix(matrix_b) if matrix_b else None
        except Exception as e:
            return self._result("ILL_FORMED", "parse", f"could not parse matrix: {e}", t0)

        try:
            diffs, label = self._matrix_residual(operation, A, B)
        except ValueError as e:
            return self._result("ILL_FORMED", "shape", str(e), t0)
        return self._classify(diffs, label, t0)

    # ---------------------------------------------------------------- internals
    def _to_matrix(self, rows: Sequence[Sequence[str]]) -> sp.Matrix:
        if not rows or not rows[0]:
            raise ValueError("empty matrix")
        width = len(rows[0])
        if any(len(r) != width for r in rows):
            raise ValueError("ragged rows")
        return sp.Matrix([[self.parse(str(c)) for c in r] for r in rows])

    @staticmethod
    def _matrix_residual(op: str, A: sp.Matrix, B: Optional[sp.Matrix]) -> Tuple[List[sp.Expr], str]:
        if op == "orthogonal":
            if A.rows != A.cols:
                raise ValueError("orthogonal requires a square matrix")
            D = A.T * A - sp.eye(A.rows)
            return list(D), "A^T*A == I"
        if op == "symmetric":
            if A.rows != A.cols:
                raise ValueError("symmetric requires a square matrix")
            return list(A.T - A), "A^T == A"
        if B is None:
            raise ValueError(f"operation '{op}' needs matrix_b")
        if op == "inverse":
            if A.cols != B.rows or A.rows != B.cols:
                raise ValueError("shape mismatch for A*B == I")
            return list(A * B - sp.eye(A.rows)), "A*B == I"
        if op == "equal":
            if A.shape != B.shape:
                raise ValueError("shape mismatch for A == B")
            return list(A - B), "A == B"
        if op == "commute":
            if A.rows != A.cols or B.shape != A.shape:
                raise ValueError("commute requires equal square matrices")
            return list(A * B - B * A), "A*B == B*A"
        raise ValueError(f"unsupported operation '{op}'")

    def _points(self, symbols: List[sp.Symbol], positive: bool, rng: random.Random):
        pts = []
        for _ in range(self.samples):
            pt = {}
            for s in symbols:
                mag = rng.uniform(0.2, 3.0)
                sign = 1 if positive else rng.choice([-1, 1])
                pt[s] = sp.Float(sign * mag, self.dps)
            pts.append(pt)
        return pts

    def _check_points(self, exprs: List[sp.Expr], points):
        """Return (valid_count, first_counterexample or None)."""
        valid = 0
        tol = sp.Float(10) ** (-(self.dps - 12))
        for pt in points:
            worst = None
            ok = True
            for e in exprs:
                try:
                    v = sp.N(e.subs(pt), self.dps)
                except Exception:
                    ok = False
                    break
                if not v.is_number or v.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
                    ok = False
                    break
                mag = abs(v)
                if worst is None or mag > worst:
                    worst = mag
            if not ok:
                continue
            valid += 1
            if worst is not None and worst > tol:
                return valid, {
                    "point": {str(k): float(v) for k, v in pt.items()},
                    "residual": sp.N(worst, 8).__str__(),
                }
        return valid, None

    def _classify(self, diffs: List[sp.Expr], label: str, t0: float) -> Dict[str, Any]:
        symbols = sorted(set().union(*[d.free_symbols for d in diffs]) if diffs else set(), key=str)

        # 1. symbolic
        symbolic_zero = False
        try:
            def all_zero():
                return all(sp.simplify(d) == 0 for d in diffs)
            symbolic_zero = bool(_run_with_timeout(self.simplify_timeout_s, all_zero))
        except _Timeout:
            symbolic_zero = False
        except Exception:
            symbolic_zero = False

        # 2. numeric (deterministic seed -> reproducible verdicts)
        rng = random.Random(self.seed)
        pos_valid, pos_bad = self._check_points(diffs, self._points(symbols, True, rng))
        gen_valid, gen_bad = self._check_points(diffs, self._points(symbols, False, rng))
        used = pos_valid + gen_valid

        if symbolic_zero:
            if pos_bad or gen_bad:
                bad = pos_bad or gen_bad
                return self._result(
                    "CONDITIONAL", "symbolic+numeric",
                    f"{label}: symbolic zero but numeric mismatch (branch/domain issue) at {bad['point']}",
                    t0, counterexample=bad, samples=used)
            return self._result("PROVED", "symbolic",
                                f"{label}: simplifies to exactly 0; numeric cross-check agrees at {used} points",
                                t0, samples=used)

        if pos_bad:
            return self._result("REFUTED", "numeric-counterexample",
                                f"{label}: fails at {pos_bad['point']} (residual {pos_bad['residual']})",
                                t0, counterexample=pos_bad, samples=used)
        if gen_bad:
            return self._result("CONDITIONAL", "numeric-domain",
                                f"{label}: holds on positive reals but fails at {gen_bad['point']}",
                                t0, counterexample=gen_bad, samples=used)
        if pos_valid < self.min_valid_points or gen_valid < self.min_valid_points:
            return self._result("CONDITIONAL", "numeric-sparse",
                                f"{label}: too few valid sample points ({pos_valid} positive, {gen_valid} general)",
                                t0, samples=used)
        return self._result("SUPPORTED", "numeric",
                            f"{label}: equal at all {used} sample points ({self.dps}-digit), not proved symbolically",
                            t0, samples=used)

    def _result(self, status, method, evidence, t0, counterexample=None, samples=0) -> Dict[str, Any]:
        return {
            "status": status,
            "method": method,
            "evidence_summary": evidence,
            "counterexample": counterexample,
            "samples_used": samples,
            "execution_time_ms": round((time.perf_counter() - t0) * 1000, 3),
            "precision_bits": int(self.dps * 3.3219),
        }
