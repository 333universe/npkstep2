"""
NPKmath Signals Module
Converts mathematical verification results into structured AI-consumable knowledge signals.

This module enables the NPKmath engine to emit normalized, timestamped verification records
that downstream AI systems (LLMs, retrieval, ranking) can use to improve reasoning.

The signal design prioritizes:
- Freshness: new verifications outweigh stale knowledge
- Confidence: high-confidence claims guide AI decision-making
- Source reliability: Bayesian source tracking influences signal weight
- Dependencies: claim relationships enable cascading verification updates
- Extensibility: signals work across domains (math, science, information systems)
"""

from dataclasses import dataclass, field, asdict, replace
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone
import json
import math


@dataclass
class VerificationSignal:
    """
    Structured output from a single claim verification.

    This is the primary data structure for AI consumption. Every verification
    in NPKmath produces exactly one signal, which is then ranked and exported
    for downstream use.

    Attributes:
        claim_id: Unique identifier for the claim being verified.
        claim_type: Type of claim ('symbolic', 'numeric', 'matrix', 'identity', etc.).
        status: Verification status (PROVED, REFUTED, VERIFIED, SUPPORTED, CONDITIONAL, ILL_FORMED, REJECTED_BY_SIEVE).
        confidence: Normalized confidence score [0.0, 1.0].
            1.0 = proven symbolically
            0.8+ = verified at high precision
            0.5-0.8 = supported but not proven
            <0.5 = uncertain or conditional
        source_reliability: Bayesian reliability score of the source [0.0, 1.0].
        freshness_score: Recency multiplier [0.0, 1.0].
            1.0 = verified moments ago
            0.5 = verified weeks ago
            0.1 = verified months ago
        timestamp: ISO 8601 UTC timestamp of verification.
        dependencies: List of claim_ids this claim depends on.
        rationale: Human-readable explanation of the verification logic.
        evidence_summary: Technical summary (e.g., "simplify(L-R) == 0").
        domain_tags: List of domain markers (e.g., ["math", "symbolic", "algebra"]).
        execution_time_ms: Wall-clock time for verification (milliseconds).
        precision_bits: Numeric precision used (if applicable).
        source_id: Optional identifier of the original knowledge source.
        related_claims: List of related or updated claim_ids (for cascading).
        metadata: Arbitrary additional fields for extensibility.
    """
    claim_id: str
    claim_type: str
    status: str
    confidence: float
    source_reliability: float
    freshness_score: float
    timestamp: str

    dependencies: List[str] = field(default_factory=list)
    rationale: str = ""
    evidence_summary: str = ""
    domain_tags: List[str] = field(default_factory=lambda: ["math"])
    execution_time_ms: float = 0.0
    precision_bits: int = 120
    source_id: Optional[str] = None
    related_claims: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # True = claim holds, False = claim is false, None = no verdict (conditional / error).
    # Keeps "how sure are we of the verdict" (confidence) separate from "what is the verdict".
    claim_holds: Optional[bool] = None
    # Proofs and counterexamples do not go stale, so freshness decay is skipped for them.
    timeless: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize signal to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize signal to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def effective_weight(self) -> float:
        """
        Compute composite weight for ranking and AI consumption.
        Combines confidence, freshness, and source reliability.

        Formula: confidence * freshness * source_reliability
        Result: [0.0, 1.0] where 1.0 is maximally trusted knowledge.
        """
        return self.confidence * self.freshness_score * self.source_reliability

    def is_high_confidence(self, threshold: float = 0.80) -> bool:
        """Check if signal meets confidence threshold."""
        return self.effective_weight() >= threshold

    def is_fresh(self, max_age_hours: float = 24) -> bool:
        """
        Check if signal is recent enough.

        Args:
            max_age_hours: Maximum age in hours to consider "fresh".
        """
        try:
            sig_time = datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))
            now = datetime.now(timezone.utc)
            age_hours = (now - sig_time).total_seconds() / 3600
            return age_hours <= max_age_hours
        except Exception:
            return False


class AIFeedbackInterface:
    """
    Manages signal emission, storage, and export for AI systems.

    This interface sits between the verification engine and downstream AI,
    providing:
    - Signal generation from raw verification results
    - Normalization and scoring
    - Storage of recent signals (prioritizing fresh knowledge)
    - Export formats for retrieval, ranking, and model prompting
    - Feedback tracking: recording which signals influenced AI decisions
    """

    def __init__(self, max_signal_history: int = 1000):
        """
        Initialize the feedback interface.

        Args:
            max_signal_history: Maximum number of signals to retain in memory.
        """
        self.signals: List[VerificationSignal] = []
        self.max_history = max_signal_history
        self.feedback_log: List[Dict[str, Any]] = []

    def emit_signal(
        self,
        claim_id: str,
        claim_type: str,
        status: str,
        confidence: float,
        source_reliability: float,
        rationale: str = "",
        evidence_summary: str = "",
        domain_tags: Optional[List[str]] = None,
        execution_time_ms: float = 0.0,
        precision_bits: int = 120,
        source_id: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        related_claims: Optional[List[str]] = None,
    ) -> VerificationSignal:
        """
        Emit a verification signal from raw verification results.

        This is the primary API for converting engine output into signals.
        It automatically computes freshness_score (always 1.0 for new signals)
        and normalizes confidence.

        Args:
            claim_id: Unique identifier for the claim.
            claim_type: Type of claim (symbolic, numeric, matrix, etc.).
            status: Verification status from engine (PROVED, REFUTED, etc.).
            confidence: Raw confidence [0.0, 1.0].
            source_reliability: Bayesian reliability [0.0, 1.0].
            rationale: Human-readable explanation.
            evidence_summary: Technical summary.
            domain_tags: List of domain markers.
            execution_time_ms: Wall-clock time.
            precision_bits: Numeric precision used.
            source_id: Original source identifier.
            dependencies: List of dependent claim_ids.
            metadata: Additional extensible data.

        Returns:
            VerificationSignal ready for AI consumption.
        """
        if domain_tags is None:
            domain_tags = ["math"]
        if dependencies is None:
            dependencies = []
        if metadata is None:
            metadata = {}

        # Normalize confidence based on status
        normalized_confidence = self._normalize_confidence(status, confidence)
        # A concrete counterexample is conclusive, so a REFUTED backed by one is fully certain.
        if status == "REFUTED" and metadata.get("counterexample"):
            normalized_confidence = 1.0

        # Fresh signals always have freshness_score = 1.0
        freshness_score = 1.0

        # Create signal
        signal = VerificationSignal(
            claim_id=claim_id,
            claim_type=claim_type,
            status=status,
            confidence=normalized_confidence,
            source_reliability=source_reliability,
            freshness_score=freshness_score,
            timestamp=datetime.now(timezone.utc).isoformat(),
            dependencies=dependencies,
            rationale=rationale,
            evidence_summary=evidence_summary,
            domain_tags=domain_tags,
            execution_time_ms=execution_time_ms,
            precision_bits=precision_bits,
            source_id=source_id,
            related_claims=related_claims or [],
            metadata=metadata,
            claim_holds=self._claim_holds(status),
            timeless=status in ("PROVED", "REFUTED"),
        )

        # Store signal
        self.signals.append(signal)
        self._trim_history()

        return signal

    def _normalize_confidence(self, status: str, raw_confidence: float) -> float:
        """
        Map engine status to normalized confidence.

        Args:
            status: Engine status string.
            raw_confidence: Raw confidence value (often 0.0 or 1.0).

        Returns:
            Normalized confidence [0.0, 1.0].
        """
        status_map = {
            "PROVED": 1.0,
            "VERIFIED": 0.9,
            "SUPPORTED": 0.7,
            "CONDITIONAL": 0.5,
            "REFUTED": 0.95,  # High confidence in negation
            "REJECTED_BY_SIEVE": 0.3,
            "ILL_FORMED": 0.1,
            "ERROR_RUNTIME_EXCEPTION": 0.0,
        }

        if status in status_map:
            return status_map[status]

        # Fallback: clamp raw confidence to [0, 1]
        return max(0.0, min(1.0, raw_confidence))

    def decay_freshness(self, age_hours: float) -> float:
        """
        Compute freshness decay factor based on age.

        This enables "new knowledge is most valuable" behavior.
        A claim verified an hour ago has freshness > 0.99.
        A claim verified a month ago has freshness < 0.5.

        Args:
            age_hours: Age of the signal in hours.

        Returns:
            Freshness score [0.0, 1.0].
        """
        # Exponential decay: F(t) = exp(-t / tau) where tau = 168 (1 week)
        tau_hours = 168.0
        return max(0.0, math.exp(-age_hours / tau_hours))

    @staticmethod
    def _claim_holds(status: str) -> Optional[bool]:
        """Map a status to the verdict about the claim itself."""
        if status in ("PROVED", "VERIFIED", "SUPPORTED"):
            return True
        if status == "REFUTED":
            return False
        return None

    @staticmethod
    def _age_hours(signal: VerificationSignal, now: datetime) -> Optional[float]:
        try:
            sig_time = datetime.fromisoformat(signal.timestamp.replace('Z', '+00:00'))
            return (now - sig_time).total_seconds() / 3600
        except Exception:
            return None

    def _current_view(
        self, now: Optional[datetime] = None
    ) -> List[Tuple[VerificationSignal, Optional[float]]]:
        """
        Copies of every stored signal with freshness recomputed for `now`.

        Never mutates stored signals, so reading a report has no side effects,
        and the age of EVERY signal is accounted for (not just recent ones).
        Timeless signals (proofs, counterexamples) keep freshness 1.0.
        """
        now = now or datetime.now(timezone.utc)
        view = []
        for s in self.signals:
            age = self._age_hours(s, now)
            if age is None:
                continue
            fresh = 1.0 if s.timeless else self.decay_freshness(max(age, 0.0))
            view.append((replace(s, freshness_score=fresh), age))
        return view

    def get_fresh_signals(
        self,
        max_age_hours: float = 24,
        min_weight: float = 0.0,
        include_timeless: bool = True,
    ) -> List[VerificationSignal]:
        """
        Retrieve current, high-quality signals (copies, sorted best first).

        Timeless signals (PROVED / REFUTED) are included regardless of age
        unless include_timeless=False, because a proof does not expire.
        """
        out = []
        for sig, age in self._current_view():
            in_window = age <= max_age_hours
            if (in_window or (include_timeless and sig.timeless)) and sig.effective_weight() >= min_weight:
                out.append(sig)
        out.sort(key=lambda s: s.effective_weight(), reverse=True)
        return out

    def get_signals_by_domain(
        self, domain_tag: str, max_age_hours: float = 24
    ) -> List[VerificationSignal]:
        """
        Retrieve signals filtered by domain tag.

        Args:
            domain_tag: Domain to filter (e.g., "math", "physics", "biology").
            max_age_hours: Maximum age.

        Returns:
            Sorted list of domain-specific signals.
        """
        domain_signals = [
            s for s in self.get_fresh_signals(max_age_hours=max_age_hours)
            if domain_tag in s.domain_tags
        ]
        return domain_signals

    def export_for_prompt(
        self, max_signals: int = 10, max_age_hours: float = 24
    ) -> str:
        """
        Export signals in a format suitable for LLM prompting.

        Returns a markdown-formatted list of high-confidence, fresh signals
        that an AI assistant can use to ground its reasoning.

        Args:
            max_signals: Maximum number of signals to include.
            max_age_hours: Only include recent signals.

        Returns:
            Markdown-formatted string ready for prompt injection.
        """
        signals = self.get_fresh_signals(max_age_hours=max_age_hours)
        holds = [x for x in signals if x.claim_holds is True][:max_signals]
        refuted = [x for x in signals if x.claim_holds is False][:max_signals]
        conditional = [x for x in signals if x.status == "CONDITIONAL"][:max_signals]

        if not (holds or refuted or conditional):
            return "## Verified Knowledge\nNo recent verified claims available.\n"

        def render(title: str, items: List[VerificationSignal]) -> List[str]:
            if not items:
                return []
            lines = [title + "\n"]
            for i, sig in enumerate(items, 1):
                lines.append(f"### Claim {i}: {sig.claim_id}")
                lines.append(f"- **Status**: {sig.status}")
                lines.append(f"- **Confidence in verdict**: {sig.effective_weight():.1%}")
                lines.append(f"- **Type**: {sig.claim_type}")
                lines.append(f"- **Rationale**: {sig.rationale}")
                if sig.evidence_summary:
                    lines.append(f"- **Evidence**: {sig.evidence_summary}")
                lines.append("")
            return lines

        out = []
        out += render("## Verified Knowledge (High Confidence)", holds)
        out += render("## Refuted Claims (these are FALSE - do not assert them)", refuted)
        out += render("## Conditional Claims (hold only under extra assumptions)", conditional)
        return "\n".join(out)

    def export_for_retrieval(
        self, max_age_hours: float = 24
    ) -> List[Dict[str, Any]]:
        """
        Export signals as ranked retrieval results.

        Suitable for vector databases, search systems, or ranking models.
        Each signal includes all metadata and an effective_weight ranking.

        Args:
            max_age_hours: Only include recent signals.

        Returns:
            List of signal dicts, ranked by effective_weight.
        """
        signals = self.get_fresh_signals(max_age_hours=max_age_hours)
        return [s.to_dict() for s in signals]

    def log_ai_feedback(
        self,
        signal_id: str,
        ai_decision: str,
        outcome: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Log when an AI system used a signal and what happened.

        This enables closed-loop learning: you can track which signals
        led to good AI predictions and which led to errors.

        Args:
            signal_id: claim_id of the signal used.
            ai_decision: What the AI decided (description).
            outcome: Result (e.g., "correct", "incorrect", "partially_correct").
            metadata: Additional context.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "signal_id": signal_id,
            "ai_decision": ai_decision,
            "outcome": outcome,
            "metadata": metadata or {},
        }
        self.feedback_log.append(entry)

    def _trim_history(self) -> None:
        """Remove oldest signals if history exceeds max_signal_history."""
        if len(self.signals) > self.max_history:
            self.signals = self.signals[-self.max_history:]

    def get_summary(self) -> Dict[str, Any]:
        """
        Return summary statistics of the signal store.

        Useful for monitoring and debugging.
        """
        view = self._current_view()
        total_signals = len(view)
        fresh_signals = len([1 for _, age in view if age <= 24])
        high_confidence = len([1 for sig, _ in view if sig.is_high_confidence()])

        avg_weight = (
            sum(sig.effective_weight() for sig, _ in view) / total_signals
            if total_signals > 0
            else 0.0
        )

        return {
            "total_signals": total_signals,
            "fresh_signals_24h": fresh_signals,
            "high_confidence_signals": high_confidence,
            "average_effective_weight": round(avg_weight, 3),
            "feedback_log_entries": len(self.feedback_log),
        }


class SignalRanker:
    """
    Advanced ranking and filtering for signals.

    Enables complex queries like:
    - "Give me the 5 most trusted math claims from the last 7 days"
    - "Rank claims by (confidence * freshness), filtered by domain"
    - "Find claims that contradict each other"
    """

    @staticmethod
    def rank_by_weight(
        signals: List[VerificationSignal], descending: bool = True
    ) -> List[VerificationSignal]:
        """Rank signals by effective_weight()."""
        return sorted(
            signals, key=lambda s: s.effective_weight(), reverse=descending
        )

    @staticmethod
    def filter_by_confidence(
        signals: List[VerificationSignal], min_confidence: float = 0.8
    ) -> List[VerificationSignal]:
        """Filter signals meeting confidence threshold."""
        return [s for s in signals if s.confidence >= min_confidence]

    @staticmethod
    def filter_by_status(
        signals: List[VerificationSignal], statuses: List[str]
    ) -> List[VerificationSignal]:
        """Filter signals by status."""
        return [s for s in signals if s.status in statuses]

    @staticmethod
    def deduplicate_by_claim(
        signals: List[VerificationSignal],
    ) -> Dict[str, VerificationSignal]:
        """
        Keep only the highest-weight signal per claim_id.

        Useful for deduplication when the same claim is verified multiple times.
        """
        best = {}
        for signal in signals:
            if signal.claim_id not in best:
                best[signal.claim_id] = signal
            elif signal.effective_weight() > best[signal.claim_id].effective_weight():
                best[signal.claim_id] = signal
        return best