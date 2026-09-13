"""Central configuration. Everything tunable lives here and reads from the
environment, so no key, path, or model id is ever hardcoded in logic."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class ConfigError(RuntimeError):
    """Raised when the app is started without the configuration it needs."""


# Per-provider defaults: the env var holding the key, and the model to use
# when TRIPMATE_MODEL is not set.
PROVIDER_DEFAULTS: dict[str, tuple[str, str]] = {
    # Groq retires models fairly often -- `llama-3.3-70b-versatile` was gone by
    # the time this was built. If you get a 404 model_not_found, list what your
    # account can actually see with:
    #     python -c "import os,groq;print([m.id for m in groq.Groq().models.list().data])"
    # and set TRIPMATE_MODEL accordingly.
    "groq": ("GROQ_API_KEY", "openai/gpt-oss-120b"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-opus-5"),
}

DEFAULT_PROVIDER = "groq"


@dataclass(frozen=True)
class Config:
    # --- LLM provider -------------------------------------------------
    provider: str = field(
        default_factory=lambda: os.getenv("TRIPMATE_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    )
    api_key: str | None = None  # resolved in __post_init__ from the provider
    model: str = ""  # resolved in __post_init__ unless TRIPMATE_MODEL is set
    max_tokens: int = field(default_factory=lambda: _env_int("TRIPMATE_MAX_TOKENS", 4096))

    # --- Orchestration guards -----------------------------------------
    # Hard ceiling on agent loop turns. Without this a confused model can
    # ping-pong tool calls forever and burn the budget.
    max_iterations: int = field(default_factory=lambda: _env_int("TRIPMATE_MAX_ITERATIONS", 6))
    request_timeout_s: float = field(
        default_factory=lambda: _env_float("TRIPMATE_TIMEOUT_S", 60.0) or 60.0
    )

    # --- RAG -----------------------------------------------------------
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("TRIPMATE_DATA_DIR", str(PROJECT_ROOT / "data"))))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("TRIPMATE_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    top_k: int = field(default_factory=lambda: _env_int("TRIPMATE_TOP_K", 3))
    # Cosine floor below which retrieval counts as a miss. Left as None by
    # default so each embedder supplies its own -- sparse and dense scores are
    # not on the same scale. Set TRIPMATE_MIN_SIMILARITY to override both.
    min_similarity: float | None = field(
        default_factory=lambda: _env_float("TRIPMATE_MIN_SIMILARITY", None)  # type: ignore[arg-type]
    )

    # --- Logging --------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.getenv("TRIPMATE_LOG_LEVEL", "INFO"))
    log_file: Path | None = field(
        default_factory=lambda: Path(os.environ["TRIPMATE_LOG_FILE"]) if os.getenv("TRIPMATE_LOG_FILE") else None
    )

    def __post_init__(self) -> None:
        # Resolve the key and model from whichever provider is selected, unless
        # they were passed explicitly (as tests do). Frozen dataclass, so we
        # assign through object.__setattr__.
        key_var, default_model = PROVIDER_DEFAULTS.get(
            self.provider, PROVIDER_DEFAULTS[DEFAULT_PROVIDER]
        )
        if self.api_key is None:
            object.__setattr__(self, "api_key", os.getenv(key_var))
        if not self.model:
            object.__setattr__(self, "model", os.getenv("TRIPMATE_MODEL") or default_model)

    @property
    def api_key_var(self) -> str:
        """Name of the environment variable this provider reads its key from."""
        return PROVIDER_DEFAULTS.get(self.provider, PROVIDER_DEFAULTS[DEFAULT_PROVIDER])[0]

    def require_api_key(self) -> str:
        """Fail loudly and early rather than at the first API call."""
        if not self.api_key:
            raise ConfigError(
                f"{self.api_key_var} is not set (provider: {self.provider}). "
                "Copy .env.example to .env and add your key, or export it in your shell "
                "before running TripMate."
            )
        return self.api_key


def load_config() -> Config:
    """Build config, loading a .env file first if python-dotenv is available."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:  # optional convenience dependency
        pass
    return Config()
