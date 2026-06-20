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
    # ROUTE A — high-context analysis / documentation (the "Gemini" route).
    analysis_model: str = field(
        default_factory=lambda: os.getenv("ANALYSIS_MODEL", "gemini-3.5-flash")
    )
    # Pretty label that goes into the [ROUTED_MODEL] metadata header.
    analysis_model_label: str = field(
        default_factory=lambda: os.getenv("ANALYSIS_MODEL_LABEL", "Gemini 3.5 Flash")
    )

    # ROUTE B — fast technical reasoning / code (the "Llama via Groq" route).
    code_model_large: str = field(
        default_factory=lambda: os.getenv("CODE_MODEL_LARGE", "llama-3.3-70b-versatile")
    )
    code_model_large_label: str = field(
        default_factory=lambda: os.getenv("CODE_MODEL_LARGE_LABEL", "Llama 3.3 70B")
    )
    code_model_small: str = field(
        default_factory=lambda: os.getenv("CODE_MODEL_SMALL", "llama-3.1-8b-instant")
    )
    code_model_small_label: str = field(
        default_factory=lambda: os.getenv("CODE_MODEL_SMALL_LABEL", "Llama 3.1 8B")
    )

    # The lightweight model used by the router to classify the agent.
    # Kept on Groq for low latency; falls back to small code model.
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
    # How many (user, assistant) turns of history to keep per chat.
    history_turns: int = field(
        default_factory=lambda: int(os.getenv("HISTORY_TURNS", "8"))
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
