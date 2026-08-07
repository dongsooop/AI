from dataclasses import dataclass, field
from typing import Optional


def normalize_intent_override(intent: Optional[str]) -> Optional[str]:
    normalized = (intent or "").strip()
    return normalized or None


@dataclass(frozen=True)
class RoutingMetadata:
    """Internal metadata describing how a chatbot response was selected.

    This model is intentionally separate from the public chatbot response. It is
    used to make routing decisions observable without exposing implementation
    details through the API contract.
    """

    intent: str
    stage: str
    tool: Optional[str] = None
    confidence: Optional[float] = None
    decision_source: str = "rule"
    source_urls: tuple[str, ...] = field(default_factory=tuple)
    fallback_used: bool = False
    fallback_reason: Optional[str] = None
    llm_required: bool = False

    def __post_init__(self) -> None:
        if not self.intent.strip():
            raise ValueError("intent must not be empty")
        if not self.stage.strip():
            raise ValueError("stage must not be empty")
        if not self.decision_source.strip():
            raise ValueError("decision_source must not be empty")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

    def to_log_fields(self) -> dict[str, object]:
        """Return aggregate-safe fields for structured runtime logging."""

        return {
            "intent": self.intent,
            "route_stage": self.stage,
            "tool": self.tool,
            "confidence": self.confidence,
            "decision_source": self.decision_source,
            "source_count": len(self.source_urls),
            "has_source": bool(self.source_urls),
            "fallback": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "llm_required": self.llm_required,
        }
