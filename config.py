"""
Central configuration for the Senior Business Analyst AI Orchestrator.

Provider modes (auto-detected at runtime):
  hybrid     — BOTH keys set: OpenRouter handles routing/fast calls (free),
               Claude handles agent responses and vision (quality)
  openrouter — Only OPENROUTER_API_KEY: all calls go through free models
  claude     — Only ANTHROPIC_API_KEY: all calls use Claude Sonnet/Haiku
  none       — No keys set (startup validation warns)
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

    # --- Admin approval gate (private chats) --------------------------------
    # A brand-new private-chat user is identified as the admin by USERNAME
    # match first (works immediately, no setup) — but usernames can change,
    # so once known, set ADMIN_USER_ID (numeric, from /id) for a robust,
    # permanent identity check that doesn't depend on the username staying
    # the same. Either one alone is enough; both together is the most robust.
    admin_username: str = field(default_factory=lambda: os.getenv("ADMIN_USERNAME", "Farruh"))
    admin_user_id: int = field(default_factory=lambda: _int_env("ADMIN_USER_ID", 0))

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
    # Fast model: routing/classification, memory extraction, AND (in hybrid
    # mode) actual answers to low-complexity/simple questions — this is the
    # model that keeps Claude usage down, so pick a genuinely good free one.
    or_fast_model: str = field(
        default_factory=lambda: os.getenv("OR_FAST_MODEL", "google/gemini-2.0-flash-exp:free")
    )
    or_fast_model_label: str = field(
        default_factory=lambda: os.getenv("OR_FAST_MODEL_LABEL", "Gemini 2.0 Flash (free)")
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
        """Active AI provider: 'hybrid' | 'openrouter' | 'claude' | 'none'.

        hybrid = both keys set:
          - fast/routing calls → OpenRouter free models
          - main agent calls   → Claude Sonnet
          - vision/PDF         → Claude (native document understanding)
        """
        if self.openrouter_api_key and self.anthropic_api_key:
            return "hybrid"
        if self.openrouter_api_key:
            return "openrouter"
        if self.anthropic_api_key:
            return "claude"
        return "none"

    @property
    def main_model_label(self) -> str:
        if self.provider in ("claude", "hybrid"):
            return self.claude_model_label
        return self.or_main_model_label

    @property
    def fast_model_label(self) -> str:
        if self.provider == "claude":
            return self.claude_fast_model_label
        return self.or_fast_model_label

    # Max tokens for agent replies. Bump if long deliverables get cut off.
    max_output_tokens: int = field(
        default_factory=lambda: _int_env("MAX_OUTPUT_TOKENS", 8192)
    )

    # Reject inbound files larger than this (Telegram Bot API limit is ~20 MB).
    max_file_size_mb: int = field(
        default_factory=lambda: _int_env("MAX_FILE_SIZE_MB", 20)
    )

    # --- Daily task reminders ------------------------------------------------
    # IANA tz name used to interpret/display due times ("bugun soat 15:00" etc).
    timezone: str = field(default_factory=lambda: os.getenv("TIMEZONE", "Asia/Tashkent"))
    # How often the reminder background loop polls for due tasks.
    reminder_poll_seconds: int = field(
        default_factory=lambda: _int_env("REMINDER_POLL_SECONDS", 30)
    )

    # --- Work-hours awareness (Business Copilot "ishdamisan?" answers) -------
    work_start_hour: int = field(default_factory=lambda: _int_env("WORK_START_HOUR", 9))
    work_end_hour: int = field(default_factory=lambda: _int_env("WORK_END_HOUR", 18))
    # Weekday numbers that are work days, Mon=0..Sun=6. Default Mon-Fri.
    work_days: set[int] = field(
        default_factory=lambda: _csv_ints(os.getenv("WORK_DAYS", "0,1,2,3,4"))
    )
    # Comma-separated holidays — "MM-DD" repeats every year (fixed-date
    # Uzbek national holidays by default); "YYYY-MM-DD" matches once (for
    # movable ones like Ro'za/Qurbon hayit, which shift yearly — add
    # manually each year via this env var).
    holidays: str = field(
        default_factory=lambda: os.getenv("HOLIDAYS", "01-01,03-08,03-21,05-09,09-01,12-08")
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

    # Group mention copilot: in any group the bot is a plain member of (no
    # admin rights needed), if someone @mentions the admin or replies to
    # the admin's own earlier message, AI-analyze it and privately notify
    # the admin with a suggested reply — independent of
    # PROACTIVE_IN_GROUPS/REQUIRE_MENTION_IN_GROUPS, which are about the
    # bot answering the GROUP directly. REQUIRES the bot's Telegram
    # Privacy Mode to be DISABLED (BotFather -> /setprivacy -> Disable)
    # for the group, otherwise Telegram never forwards messages that don't
    # @mention/reply to the BOT itself, and this sees nothing.
    watch_group_mentions: bool = field(
        default_factory=lambda: os.getenv("WATCH_GROUP_MENTIONS", "true").lower() == "true"
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

    # --- Persistent storage (PostgreSQL) ------------------------------------
    # The durable "source of truth" for relational business data: users/
    # access-control, per-user profile notes, tasks/reminders, decision log,
    # project memory. See db.py. Railway: Project -> + New -> Database ->
    # Add PostgreSQL, then on this service set DATABASE_URL = ${{Postgres.DATABASE_URL}}.
    # Falls back to Redis (if configured) then in-memory when unset — see
    # each module's docstring for its exact fallback chain.
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL", ""))

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
    def db_enabled(self) -> bool:
        return bool(self.database_url)

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
                "or ANTHROPIC_API_KEY (paid), or both for hybrid mode."
            )
        return problems


settings = Settings()
