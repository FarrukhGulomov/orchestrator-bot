"""
Shared test fixtures.

WHY RELOADING, NOT JUST MONKEYPATCHING ENV VARS: config.Settings() reads
os.environ exactly once, at import time, into a module-level `settings`
singleton (see config.py). Several other modules do `from config import
settings`, binding their own name to that same object — so changing
os.environ after import does nothing without reimporting config AND every
module that captured a reference to the old settings object. reload_env()
does both, in dependency order, and undoes it afterwards so one test's
environment can never leak into the next.

No real Postgres, Redis, or API key is used anywhere in this suite —
every module here has a working in-memory fallback tier by design (see
db.py's module docstring), and that fallback IS what these tests exercise.
"""

import importlib

import pytest

# Every var a test might set to influence provider/budget config — cleared
# before each reload_env() call so a test's result never depends on
# whatever happens to be set in the ambient environment (locally or in CI).
_RESETTABLE_VARS = (
    "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY",
    "GOOGLE_API_KEY", "XAI_API_KEY", "DEEPSEEK_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY",
    "PROVIDER_PRIORITY", "TELEMETRY_ENABLED", "DAILY_USER_TOKEN_BUDGET", "MODEL_PRICES_JSON",
)


@pytest.fixture
def reload_env(monkeypatch):
    """reload_env({"OPENAI_API_KEY": "x", ...}) -> clears every resettable
    provider/budget env var, sets the given ones, reloads config/
    llm_clients/telemetry so they pick the change up, and restores the
    previous module state on teardown."""

    def _apply(env: dict[str, str]):
        for key in _RESETTABLE_VARS:
            monkeypatch.delenv(key, raising=False)
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return _reload_all()

    yield _apply
    # pytest tears fixtures down in REVERSE setup order: monkeypatch was set
    # up before reload_env (it's a parameter), so its own env-revert runs
    # AFTER this teardown, not before — reloading here first would bake in
    # the test's env vars, not the real ones. Force the revert to happen
    # NOW (safe to call twice; pytest's own teardown no-ops afterward) so
    # this reload sees the real environment.
    monkeypatch.undo()
    _reload_all()


def _reload_all():
    import config
    importlib.reload(config)
    import llm_clients
    importlib.reload(llm_clients)
    import telemetry
    importlib.reload(telemetry)
    return config, llm_clients, telemetry


@pytest.fixture(autouse=True)
def _clear_telemetry_buffer():
    """telemetry._mem is a module-level deque — without this, spend/budget
    assertions in one test would see rows left over from another."""
    yield
    import telemetry
    telemetry._mem.clear()
    telemetry.clear_call_context()
