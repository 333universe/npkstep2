"""Glue: run the engine, emit a signal that carries the engine's real measurements."""
from typing import Any, Dict, List, Optional, Sequence

from .engine import NPKCoreEngine
from .signals import AIFeedbackInterface, VerificationSignal


def _emit(fb: AIFeedbackInterface, claim_id: str, claim_type: str, result: Dict[str, Any],
          source_reliability: float, rationale: str, domain_tags: List[str],
          dependencies: Optional[List[str]] = None, claim_text: str = "") -> VerificationSignal:
    return fb.emit_signal(
        claim_id=claim_id,
        claim_type=claim_type,
        status=result["status"],
        confidence=1.0,  # normalized from status inside emit_signal
        source_reliability=source_reliability,
        rationale=rationale,
        evidence_summary=result["evidence_summary"],
        domain_tags=domain_tags,
        execution_time_ms=result["execution_time_ms"],   # measured, not 0.0
        precision_bits=result["precision_bits"],         # actual precision used
        dependencies=dependencies or [],
        metadata={"method": result["method"],
                  "samples_used": result["samples_used"],
                  "counterexample": result["counterexample"],
                  "claim": claim_text},
    )


def verify_equality_and_emit(engine: NPKCoreEngine, fb: AIFeedbackInterface, claim_id: str,
                             lhs: str, rhs: str, source_reliability: float = 1.0,
                             rationale: str = "", domain_tags: Optional[List[str]] = None,
                             dependencies: Optional[List[str]] = None) -> VerificationSignal:
    result = engine.evaluate_symbolic_equality(lhs, rhs)
    return _emit(fb, claim_id, "symbolic", result, source_reliability,
                 rationale or f"Is {lhs} == {rhs}?", domain_tags or ["math", "symbolic"], dependencies,
                 claim_text=f"{lhs} = {rhs}")


def verify_matrix_and_emit(engine: NPKCoreEngine, fb: AIFeedbackInterface, claim_id: str,
                           operation: str, matrix_a: Sequence[Sequence[str]],
                           matrix_b: Optional[Sequence[Sequence[str]]] = None,
                           source_reliability: float = 1.0, rationale: str = "",
                           dependencies: Optional[List[str]] = None,
                           claim_text: str = "") -> VerificationSignal:
    result = engine.matrix_verify(operation, matrix_a, matrix_b)
    return _emit(fb, claim_id, "matrix", result, source_reliability,
                 rationale or f"Matrix check: {operation}", ["math", "linear-algebra", "matrix"], dependencies,
                 claim_text=claim_text)
