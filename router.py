"""
The orchestrator / router.

Stage 1 (this module): a single fast, cheap classifier call decides three things:
  1) WHICH agent should answer (department)
  2) WHAT KIND of request this is (idea / task / bug / improvement / question)
  3) HOW complex it is (drives model choice on the code route)
...and writes a short title (used for GitHub issue/PR titles).

The model is then derived deterministically — THREE routes now:

    ROUTE A  → Gemini (analysis_model)
               For analytical/text agents: PM, BA, System Analyst, QA, Designer.

    ROUTE B  → Groq / Llama 8B (code_model_fast)
               Fast, low-latency path for simple/quick technical tasks:
               short scripts, quick edits, trivial Q&A on code topics.

    ROUTE C  → GLM-5.2 via Z.AI (glm_model)
               Complex coding, multi-file refactoring, deep architecture
               reasoning, security audits. 200K context window — can hold
               entire codebases in one pass. Falls back to Groq (Route B)
               gracefully if GLM_API_KEY isn't configured.

Stage 2 (handled in bot.py): the chosen agent's persona, combined with the
request-type addendum, answers with the proper model — and, for actionable
types, the result can be filed as a GitHub issue/PR (see github_integration.py).

STICKY ROUTING: callers may pass the (agent, request_type) of the previous
turn in this chat. Short, content-free follow-ups ("ok", "continue", "ha
zur", "davom et") have nothing for the classifier to actually classify, and
guessing fresh on them was a real source of misrouting (e.g. a continuation
of a Product Manager conversation getting reclassified as SOC/BUG). The
hint nudges the classifier to keep the same agent/type for such messages,
and is also used as the fallback if the classifier call itself fails.
"""

import json
import logging
from dataclasses import dataclass, field

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
You are a routing classifier for a Fintech/Banking AI team's assistant. Read the user's
message (it may be in Uzbek, Russian, English, or a mix with IT/fintech slang) and
decide three things.

1) WHICH SPECIALIST AGENT should handle it:
General IT/Product team:
- pm                : product requirements, prioritisation, roadmap, backlog, business value
- ba                : functional requirements, user stories, use cases, business processes
- system_analyst   : system architecture, integrations, data flow, big documentation
- qa               : test scenarios, bug report structure, quality metrics
- product_designer : UX/UI design — user research, personas, journey maps, wireframes, design system
- backend          : server logic, DB schemas, APIs, code optimisation, refactoring
- frontend         : UI/UX implementation, web/mobile components, state management
- devops           : CI/CD, Dockerfiles, Kubernetes, cloud/server infrastructure
- soc              : code vulnerability review, encryption logic, security audits
- tech_lead        : technical strategy, architecture validation, code review, coordination
Fintech/Banking specialists (use when the request is clearly in the financial domain):
- compliance_officer : AML/CFT, KYC/CDD, PCI-DSS, GDPR, regulatory reporting (STR/SAR), policy
- risk_analyst       : credit risk, market risk, VaR/CVaR, stress testing, Basel III/IV, PD/LGD
- core_banking_dev   : T24/Flexcube/Mambu integrations, GL logic, ISO 20022 mapping, CBS APIs
- payment_engineer   : SWIFT MT/MX, ISO 20022, Humo/UzCard, card processing (EMV/3DS2), RTGS/ACH
- open_banking_dev   : PSD2/PSD3, OAuth 2.0/PKCE, FAPI, consent flows, OpenAPI banking specs

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
- "high" : non-trivial reasoning, architecture, multi-step code, compliance analysis

4) DOES THIS NEED A FORMATTED DOCUMENT instead of a chat reply?
Set wants_document=true for commercial proposals, cost estimates, regulatory reports,
compliance frameworks, AML policies, or anything where they ask for "Word", "PDF",
"hujjat", "tijorat taklifi", "smeta", "compliance report", "policy document".

5) IS THIS A META-QUESTION ABOUT WHAT THE ASSISTANT/TEAM ITSELF CAN DO?
Set is_capability_question=true ONLY when the user is asking about the assistant's own
capabilities — e.g. "nima qila olasan", "jamoang kimlardan iborat", "what can you do".

6) DOES THIS NEED INPUT FROM MULTIPLE ROLES to be properly answered?
Only add collaborators when genuinely multidisciplinary. Common fintech combos:
  - A new payment feature → payment_engineer + compliance_officer + backend
  - AML system design → compliance_officer + risk_analyst + core_banking_dev
  - Open banking launch → open_banking_dev + soc + compliance_officer
Max 3 collaborators, exclude the primary agent.

Also write a short TITLE (max ~70 characters) summarising the request, in the
SAME language as the user's message.

Respond with ONLY a JSON object, no prose:
{{"agent": "<one of: {", ".join(_AGENT_KEYS)}>",
  "request_type": "<one of: {", ".join(_TYPE_KEYS)}>",
  "complexity": "low"|"high",
  "wants_document": true|false,
  "is_capability_question": true|false,
  "collaborators": ["<0-3 agent keys, excluding the primary agent>"],
  "title": "<short title>"}}
"""


@dataclass
class Route:
    agent: Agent
    model: str
    model_label: str
    request_type: RequestType
    title: str
    wants_document: bool = False
    is_capability_question: bool = False
    collaborators: list[str] = field(default_factory=list)


def model_for(agent: Agent, complexity: str) -> tuple[str, str]:
    """Pick (model_id, label) for an agent + complexity.

    ROUTE A — Gemini (Google): analysis/text/product agents.
    ROUTE B — Groq fast: technical agents, low-complexity tasks.
    ROUTE C — GLM: technical agents, high-complexity tasks.
              Falls back to Groq Route B if GLM_API_KEY isn't configured.
    ROUTE F — OpenRouter: fintech/banking specialists.
              Uses deepseek-r1:free for reasoning (compliance/risk),
              qwen3-coder:free for code (payments/CBS).
              Falls back to or_model_auto (openrouter/free) if not available.
    """
    if agent.route == "A":
        return settings.analysis_model, settings.analysis_model_label

    if agent.route == "F":
        if not settings.openrouter_enabled:
            # No OpenRouter key: fall back to Gemini for analysis agents
            return settings.analysis_model, settings.analysis_model_label + " (OR fallback)"
        # Reasoning-heavy: compliance/risk → deepseek-r1
        if agent.key in ("compliance_officer", "risk_analyst"):
            return settings.or_model_reasoning, f"{settings.or_model_reasoning} (OpenRouter)"
        # Code-heavy: payment/banking/open-banking → qwen3-coder
        if agent.key in ("core_banking_dev", "payment_engineer", "open_banking_dev"):
            return settings.or_model_coding, f"{settings.or_model_coding} (OpenRouter)"
        # Default fintech: auto-select
        return settings.or_model_auto, "OpenRouter Free"

    # Technical routes (B + C)
    if complexity == "low" or not settings.glm_enabled:
        # Route B: use OpenRouter fast model when available (no TPM limits),
        # fall back to Groq only if OpenRouter isn't configured.
        if settings.openrouter_enabled:
            return settings.or_model_fast, f"{settings.or_model_fast} (OpenRouter)"
        return settings.code_model_fast, settings.code_model_fast_label
    return settings.glm_model, settings.glm_model_label


async def classify(
    user_text: str,
    last_agent: str | None = None,
    last_type: str | None = None,
) -> Route:
    """Pick agent + request type + model for a message. Always returns a valid Route."""
    agent_key = last_agent if last_agent in AGENTS else DEFAULT_AGENT_KEY
    type_key = last_type if last_type in REQUEST_TYPES else DEFAULT_REQUEST_TYPE_KEY
    complexity = "high"
    wants_document = False
    is_capability_question = False
    collaborators: list[str] = []
    title = (user_text.strip()[:70] or "Untitled")

    context_hint = ""
    if last_agent in AGENTS:
        context_hint = (
            "\nCONVERSATION CONTEXT: previous agent was "
            f"'{last_agent}', type '{type_key}'. "
            "Keep same agent/type for short follow-ups (ok, davom et, etc). "
            "Reclassify fresh if the topic clearly changes.\n"
        )

    # Try OpenRouter first (no tight TPM limits), fall back to Groq
    async def _call_classifier(system: str) -> str:
        if settings.openrouter_enabled:
            from openrouter_client import or_generate
            return await or_generate(
                settings.or_model_auto,
                system,
                [{"role": "user", "content": user_text[:4000]}],
                temperature=0.0,
            )
        return await groq_generate(
            model=settings.router_model,
            system=system,
            messages=[{"role": "user", "content": user_text[:4000]}],
            temperature=0.0,
            json_mode=True,
        )

    try:
        raw = await _call_classifier(_ROUTER_SYSTEM + context_hint)
        data = json.loads(raw)

        candidate = str(data.get("agent", "")).strip().lower()
        if candidate in AGENTS:
            agent_key = candidate

        type_candidate = str(data.get("request_type", "")).strip().lower()
        if type_candidate in REQUEST_TYPES:
            type_key = type_candidate

        complexity = "low" if str(data.get("complexity")).lower() == "low" else "high"
        wants_document = bool(data.get("wants_document", False))
        is_capability_question = bool(data.get("is_capability_question", False))

        raw_collabs = data.get("collaborators", [])
        if isinstance(raw_collabs, list):
            seen: list[str] = []
            for c in raw_collabs:
                ck = str(c).strip().lower()
                if ck in AGENTS and ck != agent_key and ck not in seen:
                    seen.append(ck)
                if len(seen) >= 3:
                    break
            collaborators = seen

        raw_title = str(data.get("title", "")).strip()
        if raw_title:
            title = raw_title[:120]
    except Exception as exc:  # noqa: BLE001 - router must never crash the bot
        # Fall back to the previous turn's agent/type when we have one — far
        # safer than a hardcoded default for a follow-up message.
        logger.warning(
            "Router classification failed, using fallback agent=%s type=%s. %s",
            agent_key, type_key, exc,
        )

    agent = get_agent(agent_key)
    model, label = model_for(agent, complexity)
    return Route(
        agent=agent,
        model=model,
        model_label=label,
        request_type=get_request_type(type_key),
        title=title,
        wants_document=wants_document,
        is_capability_question=is_capability_question,
        collaborators=collaborators,
    )
