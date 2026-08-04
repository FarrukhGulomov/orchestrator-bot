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

    # Groq — optional, powers voice-message transcription (Whisper Large v3
    # Turbo). Free tier: 2000 requests/day, no credit card needed. Get a key
    # at console.groq.com -> API Keys. Without it, voice messages get a
    # friendly "not available yet" reply instead of a native transcript.
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # Tavily — optional, powers live web search (see web_search.py) so
    # time-sensitive questions (rates, weather, news, competitor info) are
    # answered from the present instead of the model's training cutoff.
    # Free tier: 1000 searches/month, no credit card. Key: tavily.com.
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

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
    #
    # Default is OpenRouter's OWN "openrouter/free" router (launched Feb
    # 2026), NOT a pinned model id. Individual free models get deprecated
    #/retired/renamed by their providers with little notice — this bot
    # previously pinned "google/gemini-2.0-flash-exp:free", which Google
    # shut down (deprecated Feb 2026, fully retired Jun 2026), and every
    # Business/Group Copilot analysis started failing with no way to tell
    # "model is dead" apart from "token/balance ran out" from the generic
    # error alone. openrouter/free self-selects among whatever free models
    # are CURRENTLY live, so this specific failure mode can't recur — if
    # you want a specific pinned model instead, set OR_FAST_MODEL explicitly.
    or_fast_model: str = field(
        default_factory=lambda: os.getenv("OR_FAST_MODEL", "openrouter/free")
    )
    or_fast_model_label: str = field(
        default_factory=lambda: os.getenv("OR_FAST_MODEL_LABEL", "OpenRouter Free Router")
    )
    # Fallback when a pinned model (OR_MAIN_MODEL/OR_FAST_MODEL, if
    # overridden away from the router default above) turns out to be
    # unavailable — see llm_clients.py's _is_model_unavailable(). Also
    # OpenRouter's free auto-router, so the fallback never silently starts
    # billing paid usage.
    or_auto_model: str = field(
        default_factory=lambda: os.getenv("OR_AUTO_MODEL", "openrouter/free")
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

    # --- Extra providers (multi-provider failover) --------------------------
    # Each is fully optional — llm_clients.py only puts a provider in the
    # failover chain if its key is set. Model ids are env-overridable since
    # these vendors rename/retire models more often than this code changes.
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1"))
    openai_model_label: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL_LABEL", "ChatGPT (GPT-4.1)")
    )

    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
    )
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-pro"))
    gemini_model_label: str = field(
        default_factory=lambda: os.getenv("GEMINI_MODEL_LABEL", "Gemini 2.5 Pro")
    )

    xai_api_key: str = field(default_factory=lambda: os.getenv("XAI_API_KEY", ""))
    grok_model: str = field(default_factory=lambda: os.getenv("GROK_MODEL", "grok-4"))
    grok_model_label: str = field(default_factory=lambda: os.getenv("GROK_MODEL_LABEL", "Grok 4"))

    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_model: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    )
    deepseek_model_label: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_MODEL_LABEL", "DeepSeek Chat")
    )

    kimi_api_key: str = field(
        default_factory=lambda: os.getenv("KIMI_API_KEY", os.getenv("MOONSHOT_API_KEY", ""))
    )
    kimi_model: str = field(
        default_factory=lambda: os.getenv("KIMI_MODEL", "kimi-k2-0711-preview")
    )
    kimi_model_label: str = field(default_factory=lambda: os.getenv("KIMI_MODEL_LABEL", "Kimi K2"))

    # Failover order for the main conversational path (llm_clients.py) — a
    # comma-separated list of provider keys, tried in order, skipping any
    # without an API key. Not the same as the legacy `provider` property
    # below, which only ever knew about Claude/OpenRouter and still governs
    # the cheap background-triage paths (business_copilot.py/group_copilot.py)
    # that deliberately don't need top-tier quality.
    provider_priority: str = field(
        default_factory=lambda: os.getenv(
            "PROVIDER_PRIORITY", "openai,claude,gemini,grok,deepseek,kimi,openrouter"
        )
    )

    # --- LLM telemetry & spend guardrails ----------------------------------
    # Recording is on by default (it is read-only bookkeeping); the spend CAP
    # is off by default, because a cap set carelessly locks the user out of
    # their own assistant. Set DAILY_USER_TOKEN_BUDGET to a positive number
    # to switch enforcement on.
    telemetry_enabled: bool = field(
        default_factory=lambda: os.getenv("TELEMETRY_ENABLED", "true").lower() == "true"
    )
    daily_user_token_budget: int = field(
        default_factory=lambda: _int_env("DAILY_USER_TOKEN_BUDGET", 0)
    )

    @property
    def budget_enforced(self) -> bool:
        return self.telemetry_enabled and self.daily_user_token_budget > 0

    @property
    def any_ai_key_set(self) -> bool:
        return bool(
            self.anthropic_api_key or self.openrouter_api_key or self.openai_api_key
            or self.gemini_api_key or self.xai_api_key or self.deepseek_api_key
            or self.kimi_api_key
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

    # Sign each reply with the answering specialist's name+emoji ("👩‍💼 Nodira
    # · Senior Business Analyst"). Turns the agent roster into a visible team
    # — the only way to get per-agent identity from a single bot token, since
    # Telegram ties one name/avatar to one bot. Set false for bare answers.
    show_agent_signature: bool = field(
        default_factory=lambda: os.getenv("SHOW_AGENT_SIGNATURE", "true").lower() == "true"
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
    def groq_enabled(self) -> bool:
        return bool(self.groq_api_key)

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
        if not self.any_ai_key_set:
            problems.append(
                "No AI provider configured. Set OPENROUTER_API_KEY (free), "
                "ANTHROPIC_API_KEY (paid), or any of OPENAI_API_KEY/GEMINI_API_KEY/"
                "XAI_API_KEY/DEEPSEEK_API_KEY/KIMI_API_KEY."
            )
        return problems


settings = Settings()


# ---------------------------------------------------------------------------
# Model pricing — USD per 1,000,000 tokens, as (input, output).
#
# ESTIMATES, deliberately. Vendors reprice without notice and this table is
# not fetched from anywhere, so every figure derived from it is an estimate
# and must be labelled as one wherever it's shown to a user (see
# telemetry.py). It exists to answer "roughly what is this costing, and
# which provider dominates the bill" — not to reconcile an invoice.
#
# A model missing from this table is priced at 0.0 rather than guessed, so
# an unknown model shows as free instead of silently inventing a number.
# Override the whole table with MODEL_PRICES_JSON, e.g.
#   MODEL_PRICES_JSON={"gpt-4.1": [2.0, 8.0]}
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PRICES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    # OpenAI
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    # Google
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.30, 2.50),
    # xAI
    "grok-4": (3.00, 15.00),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    # Moonshot
    "kimi-k2-0711-preview": (0.60, 2.50),
    # OpenRouter's free router and any ":free" pinned model cost nothing.
    "openrouter/free": (0.0, 0.0),
}


def _load_model_prices() -> dict[str, tuple[float, float]]:
    prices = dict(_DEFAULT_MODEL_PRICES)
    raw = os.getenv("MODEL_PRICES_JSON", "").strip()
    if not raw:
        return prices
    try:
        import json

        for model, pair in json.loads(raw).items():
            prices[str(model)] = (float(pair[0]), float(pair[1]))
    except Exception:  # noqa: BLE001 — bad override must not break startup
        _logger.warning("MODEL_PRICES_JSON is not valid JSON of {model: [in, out]} — ignoring it.")
    return prices


MODEL_PRICES: dict[str, tuple[float, float]] = _load_model_prices()


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimated USD cost of one call. Unknown models cost 0.0 — see the
    MODEL_PRICES comment for why that's deliberate rather than a guess."""
    if not model:
        return 0.0
    in_rate, out_rate = MODEL_PRICES.get(model, (0.0, 0.0))
    if (in_rate, out_rate) == (0.0, 0.0) and ":free" in model:
        return 0.0
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
