"""Provider selection.

One place that turns configuration into a live provider, so adding a vendor
means adding a module and one branch here -- nothing in the agent changes.
"""

from __future__ import annotations

from tripmate.config import Config
from tripmate.llm.base import LLMProvider, ProviderError
from tripmate.logging_setup import get_logger

log = get_logger(__name__)

SUPPORTED = ("groq", "anthropic")


def build_provider(config: Config) -> LLMProvider:
    provider_name = config.provider.strip().lower()

    if provider_name not in SUPPORTED:
        raise ProviderError(
            f"Unknown provider '{config.provider}'. Set TRIPMATE_PROVIDER to one of: "
            f"{', '.join(SUPPORTED)}."
        )

    api_key = config.require_api_key()

    if provider_name == "groq":
        from tripmate.llm.groq_provider import GroqProvider

        try:
            provider: LLMProvider = GroqProvider(
                api_key=api_key, model=config.model, timeout=config.request_timeout_s
            )
        except ImportError as exc:
            raise ProviderError(
                "The 'groq' package is not installed. Run: pip install groq"
            ) from exc
    else:
        from tripmate.llm.anthropic_provider import AnthropicProvider

        try:
            provider = AnthropicProvider(
                api_key=api_key,
                model=config.model,
                timeout=config.request_timeout_s,
                max_tokens=config.max_tokens,
            )
        except ImportError as exc:
            raise ProviderError(
                "The 'anthropic' package is not installed. Run: pip install anthropic"
            ) from exc

    log.info("llm.provider_selected", extra={"provider": provider.name, "model": provider.model})
    return provider
