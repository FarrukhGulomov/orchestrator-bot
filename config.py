"""
Central configuration for the Senior Business Analyst AI Orchestrator.

All secrets are read from environment variables (see .env.example).
Only Claude (Anthropic) is used — no other AI provider keys required.
"""

import logging
import os
import re
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

_logger = logging.getLogger(__name__)

# Telegram bot tokens look like "<numeric id>:<35+ chars of [A-Za-z0-9_-]>".
_BOT_TOKEN_RE = re.compile(r"^\d+:[\w-]{35,}$")


def _int_env(name: str, default: int) -> int:
    """Read an int env var, falling back to default (with a warning) on junk
    values instead of crashing the whole process at import time."""
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        _logger.warning("Env %s=%r is not a valid integer, using default %d", name, raw, default)
        return default


def _csv_ints(raw: str) -> set[int]:
    out: set[int] = set()
    for piece in (raw or "").split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            out.add(int(piece))
        except ValueError:
            pass
    return out


@dataclass
class Settings:
    # --- Credentials -------------------------------------------------------
    bot_token: str = field(default_factory=lambda: os.getenv("BOT_TOKEN", ""))

    # OpenRouter — free models pool (primary, recommended for cost-free usage).
    # Get key at: openrouter.ai → Keys → Create Key (free, no credit card needed).
    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_referer: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_REFERER", "https://github.com/farrukhgulomov/orchestrator-bot")
    )
    openrouter_title: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_TITLE", "BA Orchestrator Bot")
    )

    # Claude (Anthropic) — optional, used when ANTHROPIC_API_KEY is set and
    # OPENROUTER_API_KEY is NOT set (or you explicitly prefer Claude).
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # --- Access control ----------------------------------------------------
    allowed_chat_ids: set[int] = field(
        default_factory=lambda: _csv_ints(os.getenv("ALLOWED_CHAT_IDS", ""))
    )

    # --- OpenRouter free models --------------------------------------------
    # Main model: deep analysis, agent responses, document generation.
    # meta-llama/llama-4-maverick:free — 128K context, multimodal, great quality, free.
    # google/gemini-2.5-flash:free — fast, smart, free (if available on OpenRouter).
    or_main_model: str = field(
        default_factory=lambda: os.getenv("OR_MAIN_MODEL", "meta-llama/llama-4-maverick:free")
    )
    or_main_model_label: str = field(
        default_factory=lambda: os.getenv("OR_MAIN_MODEL_LABEL", "Llama 4 Maverick (free)")
    )
    # Fast model: routing/classification, memory extraction (cheap & quick).
    or_fast_model: str = field(
        default_factory=lambda: os.getenv("OR_FAST_MODEL", "meta-llama/llama-4-scout:free")
    )
    or_fast_model_label: str = field(
        default_factory=lambda: os.getenv("OR_FAST_MODEL_LABEL", "Llama 4 Scout (free)")
    )
    # Universal fallback — OpenRouter auto-picks best available free model.
    or_auto_model: str = field(
        default_factory=lambda: os.getenv("OR_AUTO_MODEL", "openrouter/auto")
    )

    # --- Claude Models (used only when Claude is the active provider) ------
    claude_model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    )
    claude_model_label: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL_LABEL", "Claude Sonnet 4.6")
    )
    claude_fast_model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-4-5-20251001")
    )
    claude_fast_model_label: str = field(
        default_factory=lambda: os.getenv("CLAUDE_FAST_MODEL_LABEL", "Claude Haiku 4.5")
    )

    @property
    def provider(self) -> str:
        """Active AI provider: 'openrouter' or 'claude'."""
        if self.openrouter_api_key:
            return "openrouter"
        if self.anthropic_api_key:
            return "claude"
        return "none"

    @property
    def main_model_label(self) -> str:
        if self.provider == "openrouter":
            return self.or_main_model_label
        return self.claude_model_label

    @property
    def fast_model_label(self) -> str:
        if self.provider == "openrouter":
            return self.or_fast_model_label
        return self.claude_fast_model_label

    # Max tokens for agent replies. Bump if long deliverables get cut off.
    max_output_tokens: int = field(
        default_factory=lambda: _int_env("MAX_OUTPUT_TOKENS", 8192)
    )

    # Reject inbound files larger than this (Telegram Bot API limit is ~20 MB).
    max_file_size_mb: int = field(
        default_factory=lambda: _int_env("MAX_FILE_SIZE_MB", 20)
    )

    # --- Behaviour ---------------------------------------------------------
    history_turns: int = field(
        default_factory=lambda: _int_env("HISTORY_TURNS", 10)
    )
    request_timeout: int = field(
        default_factory=lambda: _int_env("REQUEST_TIMEOUT", 90)
    )
    # In groups: only respond when explicitly @mentioned or replied to.
    require_mention_in_groups: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_MENTION_IN_GROUPS", "true").lower() == "true"
    )

    # Proactive group mode: bot analyses every group message and joins in when relevant,
    # even without @mention. Cooldown prevents spam (seconds between proactive replies).
    proactive_in_groups: bool = field(
        default_factory=lambda: os.getenv("PROACTIVE_IN_GROUPS", "false").lower() == "true"
    )
    proactive_cooldown_seconds: int = field(
        default_factory=lambda: _int_env("PROACTIVE_COOLDOWN_SECONDS", 45)
    )

    # --- Output metadata header --------------------------------------------
    show_metadata_header: bool = field(
        default_factory=lambda: os.getenv("SHOW_METADATA_HEADER", "false").lower() == "true"
    )

    # --- Optional: GitHub integration ---------------------------------------
    github_token: str = field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    github_repo: str = field(default_factory=lambda: os.getenv("GITHUB_REPO", ""))
    github_default_branch: str = field(
        default_factory=lambda: os.getenv("GITHUB_DEFAULT_BRANCH", "main")
    )
    github_auto_pr: bool = field(
        default_factory=lambda: os.getenv("GITHUB_AUTO_PR", "false").lower() == "true"
    )

    # --- Persistent storage (Redis) ----------------------------------------
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", ""))
    redis_history_ttl_seconds: int = field(
        default_factory=lambda: _int_env("REDIS_HISTORY_TTL_SECONDS", 60 * 60 * 24 * 30)
    )

    # --- Railway integration (optional, read-only) -------------------------
    railway_api_token: str = field(default_factory=lambda: os.getenv("RAILWAY_API_TOKEN", ""))
    railway_project_id: str = field(default_factory=lambda: os.getenv("RAILWAY_PROJECT_ID", ""))
    railway_environment_id: str = field(
        default_factory=lambda: os.getenv("RAILWAY_ENVIRONMENT_ID", "")
    )
    railway_service_id: str = field(default_factory=lambda: os.getenv("RAILWAY_SERVICE_ID", ""))

    @property
    def redis_enabled(self) -> bool:
        return bool(self.redis_url)

    @property
    def railway_enabled(self) -> bool:
        return bool(
            self.railway_api_token
            and self.railway_project_id
            and self.railway_environment_id
            and self.railway_service_id
        )

    @property
    def github_enabled(self) -> bool:
        return bool(self.github_token and self.github_repo)

    def validate(self) -> list[str]:
        problems = []
        if not self.bot_token:
            problems.append("BOT_TOKEN is not set.")
        elif not _BOT_TOKEN_RE.match(self.bot_token):
            problems.append(
                "BOT_TOKEN does not look like a valid Telegram bot token "
                "(expected '<digits>:<35+ chars>' from @BotFather)."
            )
        if self.provider == "none":
            problems.append(
                "No AI provider configured. Set OPENROUTER_API_KEY (free) "
                "or ANTHROPIC_API_KEY (paid)."
            )
        return problems


settings = Settings()
