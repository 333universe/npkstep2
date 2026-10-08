from npkmath.engine import NPKCoreEngine
from npkmath.benchmark import run_benchmark

e = NPKCoreEngine()


def test_benchmark_has_no_critical_errors_or_false_assurance():
    s = run_benchmark(e)["summary"]
    assert s["critical_errors"] == 0
    assert s["false_assurance"] == 0
    assert s["exact_rate"] >= 0.9


def test_matrix_verify_reads_its_input():
    # Regression: the old notebook version ignored matrix_a and returned PROVED for anything.
    bad = e.matrix_verify("orthogonal", [["1", "2"], ["3", "4"]])
    good = e.matrix_verify("orthogonal", [["1", "0"], ["0", "1"]])
    assert bad["status"] == "REFUTED"
    assert bad["counterexample"] is None or isinstance(bad["counterexample"], dict)
    assert good["status"] == "PROVED"


def test_refuted_always_has_justification():
    r = e.evaluate_symbolic_equality("(x+y)**2", "x**2+y**2")
    assert r["status"] == "REFUTED"
    assert r["counterexample"]["point"]


def test_domain_dependent_claim_is_conditional_not_proved():
    r = e.evaluate_symbolic_equality("sqrt(x**2)", "x")
    assert r["status"] == "CONDITIONAL"


def test_numeric_only_agreement_is_supported_not_proved():
    r = e.evaluate_symbolic_equality("log(exp(x))", "x")
    assert r["status"] == "SUPPORTED"


def test_high_precision_catches_tiny_difference():
    r = e.evaluate_symbolic_equality("x**2+10**(-30)", "x**2")
    assert r["status"] == "REFUTED"


def test_rejects_non_math_input():
    assert e.evaluate_symbolic_equality("__import__('os').system('echo hi')", "1")["status"] == "ILL_FORMED"
    assert e.evaluate_symbolic_equality("x +* 2", "3")["status"] == "ILL_FORMED"


def test_matrix_shape_errors_are_ill_formed():
    assert e.matrix_verify("symmetric", [["1", "2", "3"], ["4", "5", "6"]])["status"] == "ILL_FORMED"
    assert e.matrix_verify("inverse", [["1"]])["status"] == "ILL_FORMED"  # missing matrix_b


def test_verdicts_are_reproducible():
    a = e.evaluate_symbolic_equality("sin(x)", "x")
    b = NPKCoreEngine().evaluate_symbolic_equality("sin(x)", "x")
    assert a["status"] == b["status"] and a["counterexample"] == b["counterexample"]


def test_measured_fields_are_real():
    r = e.evaluate_symbolic_equality("sin(x)**2+cos(x)**2", "1")
    assert r["execution_time_ms"] > 0
    assert r["precision_bits"] == int(50 * 3.3219)
