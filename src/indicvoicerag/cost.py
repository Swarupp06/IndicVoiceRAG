"""Verified ₹0 cost status per LLM provider (Phase 2.5, Step 8).

Every entry records what was checked against the provider's own public
documentation, when it was checked, and whether the project can stay at ₹0.
`free` is only True when the documentation states a no-payment path; anything
that could not be confirmed from an account we control is marked
`account_verified = False` and must be reported as NOT VERIFIED rather than
"free".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

CHECKED_ON = "2026-08-16"


@dataclass(frozen=True, slots=True)
class ProviderCost:
    provider: str
    model: str
    free: bool
    billing_required: bool
    api_key_required: bool
    card_required: bool
    free_limit: str
    eligible_zero_cost: bool
    account_verified: bool
    source: str
    notes: str = ""
    checked_on: str = CHECKED_ON

    @property
    def status(self) -> str:
        if not self.eligible_zero_cost:
            return "NOT ELIGIBLE"
        if not self.account_verified:
            return "FREE (DOCS) / NOT ACCOUNT-VERIFIED"
        return "FREE (VERIFIED ₹0)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "free": self.free,
            "billing_required": self.billing_required,
            "api_key_required": self.api_key_required,
            "card_required": self.card_required,
            "free_limit": self.free_limit,
            "api_cost_during_benchmark": "₹0" if self.eligible_zero_cost else "n/a (not run)",
            "eligible_zero_cost": self.eligible_zero_cost,
            "account_verified": self.account_verified,
            "status": self.status,
            "source": self.source,
            "notes": self.notes,
            "checked_on": self.checked_on,
        }


PROVIDER_COSTS: dict[str, ProviderCost] = {
    "ollama": ProviderCost(
        provider="ollama",
        model="qwen2.5:1.5b-instruct",
        free=True,
        billing_required=False,
        api_key_required=False,
        card_required=False,
        free_limit="unlimited (local CPU/GPU inference, only local compute)",
        eligible_zero_cost=True,
        account_verified=True,
        source="https://ollama.com/ (local runtime, no account)",
        notes="Runs on this machine, no network call leaves the host, so the ₹0 path is fully verified.",
    ),
    "gemini": ProviderCost(
        provider="gemini",
        model="gemini-2.5-flash-lite",
        free=True,
        billing_required=False,
        api_key_required=True,
        card_required=False,
        free_limit="Free tier per-model RPM/RPD/TPM caps published in the rate-limit docs",
        eligible_zero_cost=True,
        account_verified=False,
        source="https://ai.google.dev/gemini-api/docs/rate-limits, https://ai.google.dev/pricing",
        notes=(
            "Google documents a free tier that needs only an AI Studio API key (no billing account). "
            "No key was available in this session, so no request was made and no quota was consumed."
        ),
    ),
    "groq": ProviderCost(
        provider="groq",
        model="llama-3.1-8b-instant",
        free=True,
        billing_required=False,
        api_key_required=True,
        card_required=False,
        free_limit="Free plan: 30 RPM / 14,400 RPD / 6,000 TPM / 500,000 TPD for llama-3.1-8b-instant",
        eligible_zero_cost=True,
        account_verified=False,
        source="https://console.groq.com/docs/rate-limits",
        notes=(
            "Groq's Free plan needs an API key but no card. No key was available in this session, "
            "so the provider is implemented but not benchmarked."
        ),
    ),
    "openrouter": ProviderCost(
        provider="openrouter",
        model="google/gemma-4-31b-it:free",
        free=True,
        billing_required=False,
        api_key_required=True,
        card_required=False,
        free_limit="`:free` model variants only, 20 requests/min and 50 requests/day without purchased credits",
        eligible_zero_cost=True,
        account_verified=False,
        source="https://openrouter.ai/docs/api-reference/limits",
        notes=(
            "Only `:free` model ids stay at ₹0; the provider rejects any non-`:free` model id. "
            "No key was available in this session, so it is implemented but not benchmarked."
        ),
    ),
    "openai_compatible": ProviderCost(
        provider="openai_compatible",
        model="(generic endpoint)",
        free=False,
        billing_required=True,
        api_key_required=True,
        card_required=True,
        free_limit="none - depends entirely on the endpoint operator",
        eligible_zero_cost=False,
        account_verified=False,
        source="https://openai.com/api/pricing/",
        notes="Escape hatch for self-hosted endpoints. NOT ELIGIBLE as the demo provider: cost is unknown/paid.",
    ),
    "mock": ProviderCost(
        provider="mock",
        model="mock-rag-generator",
        free=True,
        billing_required=False,
        api_key_required=False,
        card_required=False,
        free_limit="unlimited (deterministic offline stub)",
        eligible_zero_cost=True,
        account_verified=True,
        source="in-repo implementation",
        notes="Tests only. Never used as a silent production/demo fallback (allow_mock_fallback defaults to false).",
    ),
}


def cost_table() -> list[dict[str, Any]]:
    return [cost.as_dict() for cost in PROVIDER_COSTS.values()]


def zero_cost_providers() -> list[str]:
    """Providers with a documented ₹0 path (still may need an API key)."""
    return [name for name, cost in PROVIDER_COSTS.items() if cost.eligible_zero_cost and name != "mock"]
