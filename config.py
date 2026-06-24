"""
Central configuration for the Master Orchestrator Telegram bot.

All secrets and tunables are read from environment variables (see .env.example).
Model IDs are intentionally kept here as constants so they can be swapped in a
single place when providers deprecate models.

IMPORTANT (as of June 2026):
  * Gemini 1.5 Flash is fully shut down (returns 404). Default analysis model is
    now a current GA Flash model.
  * Groq deprecated llama-3.1-8b-instant and llama-3.3-70b-versatile on
    2026-06-17. They still respond during the deprecation window; migrate to the
    gpt-oss IDs below when they are fully removed.
Just change the values here (or override via .env) to point at whatever your
account supports.
"""

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional; env may be set externally
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
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # --- Access control ----------------------------------------------------
    # Comma-separated chat IDs. Empty => allow every chat (use with care).
    allowed_chat_ids: set[int] = field(
        default_factory=lambda: _csv_ints(os.getenv("ALLOWED_CHAT_IDS", ""))
    )

    # --- Models ------------------------------------------------------------
    # ROUTE A — high-context analysis / documentation (Gemini).
    analysis_model: str = field(
        default_factory=lambda: os.getenv("ANALYSIS_MODEL", "gemini-3.5-flash")
    )
    analysis_model_label: str = field(
        default_factory=lambda: os.getenv("ANALYSIS_MODEL_LABEL", "Gemini 3.5 Flash")
    )

    # ROUTE B — fast, simple code tasks (Groq / Llama 8B — low latency).
    # Used for low-complexity technical requests that don't need a full
    # reasoning pass, quick edits, short scripts.
    code_model_fast: str = field(
        default_factory=lambda: os.getenv("CODE_MODEL_FAST", "llama-3.1-8b-instant")
    )
    code_model_fast_label: str = field(
        default_factory=lambda: os.getenv("CODE_MODEL_FAST_LABEL", "Llama 3.1 8B (Groq)")
    )

    # ROUTE C — complex coding, architecture, refactoring, security audits.
    # Supports three providers (pick ONE, set the corresponding env vars):
    #
    #   1. Z.AI Coding Plan (GLM-5.2, recommended — $18/month)
    #      GLM_API_KEY = your Z.AI key
    #      GLM_MODEL = glm-5.2
    #      GLM_BASE_URL = https://api.z.ai/api/coding/paas/v4/   ← Coding Plan endpoint
    #
    #   2. Together AI (GLM-5.2, pay-per-token, no subscription)
    #      GLM_API_KEY = your Together AI key
    #      GLM_MODEL = zai-org/GLM-5.2
    #      GLM_BASE_URL = https://api.together.xyz/v1/
    #
    #   3. Z.AI general API (GLM-4.7 or GLM-4.7-Flash — free tier, older model)
    #      GLM_API_KEY = your Z.AI key
    #      GLM_MODEL = glm-4.7            (or glm-4.7-flash for the free tier)
    #      GLM_BASE_URL = https://api.z.ai/api/paas/v4/   ← general endpoint
    #
    # If GLM_API_KEY is empty, Route C falls back to Groq (Route B) silently.
    # If the model isn't accessible (404), the bot logs the problem and falls
    # back to Groq instead of crashing on every message.
    glm_api_key: str = field(default_factory=lambda: os.getenv("GLM_API_KEY", ""))
    glm_base_url: str = field(
        default_factory=lambda: os.getenv("GLM_BASE_URL", "https://api.z.ai/api/paas/v4/")
    )
    glm_model: str = field(
        default_factory=lambda: os.getenv("GLM_MODEL", "glm-4.7")
    )
    glm_model_label: str = field(
        default_factory=lambda: os.getenv("GLM_MODEL_LABEL", "GLM-4.7 (Z.AI)")
    )

    @property
    def glm_enabled(self) -> bool:
        return bool(self.glm_api_key)

    # Keep legacy names as aliases so nothing breaks if old env vars are set.
    @property
    def code_model_large(self) -> str:
        return os.getenv("CODE_MODEL_LARGE", self.glm_model)

    @property
    def code_model_large_label(self) -> str:
        return os.getenv("CODE_MODEL_LARGE_LABEL", self.glm_model_label)

    @property
    def code_model_small(self) -> str:
        return os.getenv("CODE_MODEL_SMALL", self.code_model_fast)

    @property
    def code_model_small_label(self) -> str:
        return os.getenv("CODE_MODEL_SMALL_LABEL", self.code_model_fast_label)

    # ROUTE F — OpenRouter: unified free model pool for Fintech/Banking agents.
    # Single key → 300+ models, 26+ free (:free suffix). No credit card needed.
    # Get key at: openrouter.ai (sign up → Keys → Create Key).
    # Free tier limits: 50 req/day, 20 req/min per model.
    openrouter_api_key: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    openrouter_referer: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_REFERER", "https://github.com/fintech-orchestrator")
    )
    openrouter_title: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_TITLE", "Fintech AI Orchestrator")
    )
    # Reasoning model for compliance/risk/regulatory analysis
    or_model_reasoning: str = field(
        default_factory=lambda: os.getenv("OR_MODEL_REASONING", "deepseek/deepseek-r1:free")
    )
    # Coding model for banking/payment systems code
    or_model_coding: str = field(
        default_factory=lambda: os.getenv("OR_MODEL_CODING", "qwen/qwen3-coder:free")
    )
    # Fast model for quick analysis tasks
    or_model_fast: str = field(
        default_factory=lambda: os.getenv("OR_MODEL_FAST", "meta-llama/llama-4-scout:free")
    )
    # Auto-select: always free, always available — use as universal fallback
    or_model_auto: str = field(
        default_factory=lambda: os.getenv("OR_MODEL_AUTO", "openrouter/free")
    )

    @property
    def openrouter_enabled(self) -> bool:
        return bool(self.openrouter_api_key)

    # The lightweight model used by the router to classify the agent.
    router_model: str = field(
        default_factory=lambda: os.getenv("ROUTER_MODEL", "llama-3.1-8b-instant")
    )

    # Audio transcription (voice notes / mp3 / wav / etc.) via Groq Whisper.
    whisper_model: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL", "whisper-large-v3-turbo")
    )

    # Max tokens for a single agent reply. Was hardcoded at 2048 in
    # llm_clients.py, which silently truncated longer TASK/BUG/REFINEMENT
    # deliverables (e.g. a full design spec cut off mid-sentence) — bumped to
    # a much safer default and made configurable.
    max_output_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
    )

    # Reject inbound files larger than this (Telegram Bot API download limit
    # is ~20MB regardless; this just fails fast with a clear message).
    max_file_size_mb: int = field(
        default_factory=lambda: int(os.getenv("MAX_FILE_SIZE_MB", "20"))
    )

    # --- Behaviour ---------------------------------------------------------
    # How many (user, assistant) turns of history to keep per chat (for
    # Route A/Gemini and Route C/GLM which have large context windows).
    history_turns: int = field(
        default_factory=lambda: int(os.getenv("HISTORY_TURNS", "8"))
    )
    # Route B (Groq / Llama 8B) has a much tighter token limit on the free
    # tier (6000 TPM). We cap history to the last N turns to avoid 413 errors.
    # Set higher if you've upgraded to Groq Dev Tier.
    groq_max_history_turns: int = field(
        default_factory=lambda: int(os.getenv("GROQ_MAX_HISTORY_TURNS", "2"))
    )
    # Hard cap on total characters in the payload sent to Groq (system +
    # history + current message), as a safety net AFTER history trimming.
    # ~4 chars/token, 6000 token limit → safe ceiling is ~20000 chars.
    groq_max_chars: int = field(
        default_factory=lambda: int(os.getenv("GROQ_MAX_CHARS", "20000"))
    )
    request_timeout: int = field(
        default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "60"))
    )
    # If True, the bot only answers in groups when explicitly mentioned or replied to.
    require_mention_in_groups: bool = field(
        default_factory=lambda: os.getenv("REQUIRE_MENTION_IN_GROUPS", "true").lower()
        == "true"
    )

    # --- Optional: GitHub integration (idea/task/bug/improvement -> tracked work) ---
    # Fully optional. Leave GITHUB_TOKEN or GITHUB_REPO empty to disable.
    # NEVER put the token anywhere but .env on your own server — never in chat,
    # never committed to the repo.
    github_token: str = field(default_factory=lambda: os.getenv("GITHUB_TOKEN", ""))
    github_repo: str = field(default_factory=lambda: os.getenv("GITHUB_REPO", ""))  # "owner/repo"
    github_default_branch: str = field(
        default_factory=lambda: os.getenv("GITHUB_DEFAULT_BRANCH", "main")
    )
    # If true, actionable code tickets that contain file-marked code blocks also
    # open a draft PR with the AI-generated implementation on a new branch.
    # AI-written PRs must always be reviewed before merging — keep this off
    # until you trust the flow.
    github_auto_pr: bool = field(
        default_factory=lambda: os.getenv("GITHUB_AUTO_PR", "false").lower() == "true"
    )

    # Output header format for every reply:
    #   "x_metadata" (default) — the structured **[X_ORCHESTRATOR_METADATA]** block
    #                            with CLASSIFICATION / ASSIGNED_AGENT / ROUTED_MODEL
    #   "simple"               — compact [ACTIVE_AGENT] / [ROUTED_MODEL] lines
    metadata_format: str = field(
        default_factory=lambda: os.getenv("METADATA_FORMAT", "x_metadata").strip().lower()
    )
    # If False (default now), no metadata header is shown to the user at all —
    # routing info still goes to logs, but chat replies read like they came
    # from a person, not a visibly multi-agent system. Set true to restore
    # the visible header (useful while testing/debugging routing).
    show_metadata_header: bool = field(
        default_factory=lambda: os.getenv("SHOW_METADATA_HEADER", "false").lower() == "true"
    )

    # --- Voice replies (voice note in -> voice note out) --------------------
    # Uses Gemini's native speech-generation models. Coverage is solid for
    # English/Russian and many others; Uzbek is NOT explicitly confirmed in
    # Google's published language list as of this writing — test it for your
    # team's actual usage and disable if quality is poor.
    voice_replies_enabled: bool = field(
        default_factory=lambda: os.getenv("VOICE_REPLIES_ENABLED", "true").lower() == "true"
    )
    tts_model: str = field(
        default_factory=lambda: os.getenv("TTS_MODEL", "gemini-3.1-flash-tts-preview")
    )
    # One of Gemini's prebuilt voice names (e.g. Kore, Puck, Charon, Fenrir...).
    tts_voice: str = field(default_factory=lambda: os.getenv("TTS_VOICE", "Kore"))

    # Languages where Gemini TTS has been confirmed (by real testing, twice)
    # to produce the WRONG language entirely (Uzbek -> Kazakh) even with an
    # explicit language_code — the model most likely was never trained on
    # real audio for these languages, so no parameter can fix it. For any
    # language code in this set, voice replies skip TTS entirely and send
    # text instead, rather than risk sending audio in the wrong language.
    # Comma-separated uz/ru/en codes. Remove 'uz' here only if you've
    # verified pronunciation is actually correct for your use case.
    tts_unsupported_languages: set[str] = field(
        default_factory=lambda: {
            s.strip().lower()
            for s in os.getenv("TTS_UNSUPPORTED_LANGUAGES", "uz").split(",")
            if s.strip()
        }
    )

    # --- Persistent storage (Redis) -----------------------------------------
    # Without this, conversation history and project memory live only in this
    # process's RAM and are wiped on every restart/redeploy (Railway does
    # this on every git push). Add Railway's Redis database to your project
    # and set REDIS_URL=${{Redis.REDIS_URL}} on the bot service to persist
    # both across deploys. Empty = in-memory fallback (current behaviour).
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", ""))
    # How long inactive chat HISTORY is kept before Redis expires it
    # (default 30 days). Project MEMORY facts have no TTL — they're meant to
    # be durable until explicitly /forget'ten.
    redis_history_ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("REDIS_HISTORY_TTL_SECONDS", str(60 * 60 * 24 * 30)))
    )

    @property
    def redis_enabled(self) -> bool:
        return bool(self.redis_url)

    # --- Railway integration (optional, read-only) --------------------------
    # Lets the DevOps/SOC/Tech Lead agents read REAL deployment status and
    # logs from this bot's own Railway service via Railway's public GraphQL
    # API — see railway_integration.py. Deliberately read-only: no
    # deploy-trigger or service-mutation capability is exposed (see that
    # module's docstring for why).
    # Only RAILWAY_API_TOKEN needs to be set manually (Railway -> Account
    # Settings -> Tokens). Project/Environment/Service ID are normally
    # auto-injected by Railway into every deployed service's own environment
    # — since this bot runs ON Railway, it already knows its own IDs for free.
    railway_api_token: str = field(default_factory=lambda: os.getenv("RAILWAY_API_TOKEN", ""))
    railway_project_id: str = field(default_factory=lambda: os.getenv("RAILWAY_PROJECT_ID", ""))
    railway_environment_id: str = field(
        default_factory=lambda: os.getenv("RAILWAY_ENVIRONMENT_ID", "")
    )
    railway_service_id: str = field(default_factory=lambda: os.getenv("RAILWAY_SERVICE_ID", ""))

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
        """Return a list of human-readable problems; empty list == OK."""
        problems = []
        if not self.bot_token:
            problems.append("BOT_TOKEN is not set.")
        if not self.gemini_api_key:
            problems.append("GEMINI_API_KEY is not set (ROUTE A will fail).")
        if not self.groq_api_key:
            problems.append("GROQ_API_KEY is not set (ROUTE B + router will fail).")
        return problems


settings = Settings()
