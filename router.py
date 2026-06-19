"""
The orchestrator / router.

Stage 1 (this module): a single fast, cheap classifier call decides three things:
  1) WHICH agent should answer (department)
  2) WHAT KIND of request this is (idea / task / bug / improvement / question)
  3) HOW complex it is (drives model size on the code route)
...and writes a short title (used for GitHub issue/PR titles).

The model is then derived deterministically:

    route A  -> analysis model (Gemini)
    route B  -> code model; "large" for non-trivial work, "small" for quick stuff

Stage 2 (handled in bot.py): the chosen agent's persona, combined with the
request-type addendum, answers with the proper model — and, for actionable
types, the result can be filed as a GitHub issue/PR (see github_integration.py).
"""

import json
import logging
from dataclasses import dataclass

from agents import AGENTS, DEFAULT_AGENT_KEY, Agent, get_agent
from config import settings
from llm_clients import groq_generate
from request_types import (
    DEFAULT_REQUEST_TYPE_KEY,
    REQUEST_TYPES,
    RequestType,
    get_request_type,
)

logger = logging.getLogger(__name__)

_AGENT_KEYS = list(AGENTS.keys())
_TYPE_KEYS = list(REQUEST_TYPES.keys())

_ROUTER_SYSTEM = f"""
You are a routing classifier for an IT product team's assistant. Read the user's
message (it may be in Uzbek, Russian, English, or a mix, with IT slang) and
decide three things.

1) WHICH SPECIALIST AGENT should handle it:
- pm              : product requirements, prioritisation, roadmap, backlog, business value
- ba              : functional requirements, user stories, use cases, business processes
- system_analyst : system architecture, integrations, data flow, big documentation
- qa             : test scenarios, bug report structure, quality metrics
- backend        : server logic, DB schemas, APIs, code optimisation, refactoring
- frontend       : UI/UX implementation, web/mobile components, state management
- devops         : CI/CD, Dockerfiles, Kubernetes, cloud/server infrastructure
- soc            : code vulnerability review, encryption logic, security audits
- tech_lead      : technical strategy, architecture validation, code review, coordination

2) WHAT KIND OF REQUEST this is (the Input Classification Matrix):
- idea        : IDEA — a conceptual suggestion, high-level thought, or new feature proposal
- task        : TASK / FEATURE — a direct requirement for a new component, field, endpoint, or script
- bug         : BUG — a system error, crash, broken behaviour, or security vulnerability
- improvement : REFINEMENT / ДОРАБОТКА — improvement to existing code: performance, refactor, UI/UX
- question    : QUESTION — just asking for information or advice, nothing to implement or track
  (use this ONLY for genuine non-actionable chat; if there is any concrete thing
  to build, fix, or improve, classify it as task/bug/improvement instead)

3) COMPLEXITY:
- "low"  : short, simple, quick question or trivial snippet
- "high" : non-trivial reasoning, architecture, multi-step code, refactor, audit

Also write a short TITLE (max ~70 characters) summarising the request, in the
SAME language as the user's message.

Respond with ONLY a JSON object, no prose:
{{"agent": "<one of: {", ".join(_AGENT_KEYS)}>",
  "request_type": "<one of: {", ".join(_TYPE_KEYS)}>",
  "complexity": "low"|"high",
  "title": "<short title>"}}
"""


@dataclass
class Route:
    agent: Agent
    model: str
    model_label: str
    request_type: RequestType
    title: str


def _model_for(agent: Agent, complexity: str) -> tuple[str, str]:
    if agent.route == "A":
        return settings.analysis_model, settings.analysis_model_label
    # route B
    if complexity == "low":
        return settings.code_model_small, settings.code_model_small_label
    return settings.code_model_large, settings.code_model_large_label


async def classify(user_text: str) -> Route:
    """Pick agent + request type + model for a message. Always returns a valid Route."""
    agent_key = DEFAULT_AGENT_KEY
    type_key = DEFAULT_REQUEST_TYPE_KEY
    complexity = "high"
    title = (user_text.strip()[:70] or "Untitled")
    try:
        raw = await groq_generate(
            model=settings.router_model,
            system=_ROUTER_SYSTEM,
            messages=[{"role": "user", "content": user_text[:4000]}],
            temperature=0.0,
            json_mode=True,
        )
        data = json.loads(raw)

        candidate = str(data.get("agent", "")).strip().lower()
        if candidate in AGENTS:
            agent_key = candidate

        type_candidate = str(data.get("request_type", "")).strip().lower()
        if type_candidate in REQUEST_TYPES:
            type_key = type_candidate

        complexity = "low" if str(data.get("complexity")).lower() == "low" else "high"

        raw_title = str(data.get("title", "")).strip()
        if raw_title:
            title = raw_title[:120]
    except Exception as exc:  # noqa: BLE001 - router must never crash the bot
        logger.warning("Router classification failed, using default. %s", exc)

    agent = get_agent(agent_key)
    model, label = _model_for(agent, complexity)
    return Route(
        agent=agent,
        model=model,
        model_label=label,
        request_type=get_request_type(type_key),
        title=title,
    )
