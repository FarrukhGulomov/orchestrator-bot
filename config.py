"""
Central configuration for the Senior Business Analyst AI Orchestrator.

All secrets are read from environment variables (see .env.example).
Only Claude (Anthropic) is used — no other AI provider keys required.
"""

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


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
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))

    # --- Access control ----------------------------------------------------
    allowed_chat_ids: set[int] = field(
        default_factory=lambda: _csv_ints(os.getenv("ALLOWED_CHAT_IDS", ""))
    )

    # --- Claude Models -----------------------------------------------------
    # Main model for all agents — deep analysis, document generation, reasoning.
    claude_model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
    )
    claude_model_label: str = field(
        default_factory=lambda: os.getenv("CLAUDE_MODEL_LABEL", "Claude Sonnet 4.6")
    )
    # Fast model for routing/classification and quick responses.
    claude_fast_model: str = field(
        default_factory=lambda: os.getenv("CLAUDE_FAST_MODEL", "claude-haiku-4-5-20251001")
    )
    claude_fast_model_label: str = field(
        default_factory=lambda: os.getenv("CLAUDE_FAST_MODEL_LABEL", "Claude Haiku 4.5")
    )

    # Max tokens for agent replies. Bump if long deliverables get cut off.
    max_output_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
    )

    # Reject inbound files larger than this (Telegram Bot API limit is ~20 MB).
    max_file_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "20"))
    )

    # --- Behaviour ---------------------------------------------------------
    history_turns: int = field(
        default_factory=lambda: int(os.getenv("HISTORY_TURNS", "10"))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "90"))
    )
    # In groups: only respond when explicitly @mentioned or replied to.
    require_mention_in_groups: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_MENTION_IN_GROUPS", "true").lower() == "true"
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
        default_factory=lambda: int(os.getenv("REDIS_HISTORY_TTL_SECONDS", str(60 * 60 * 24 * 30)))
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
        if not self.anthropic_api_key:
            problems.append("ANTHROPIC_API_KEY is not set — all agents will fail.")
        return problems


settings = Settings()
