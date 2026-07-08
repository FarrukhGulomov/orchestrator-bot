"""
Orchestrator / router — Claude-only implementation.

Stage 1 (this module): a fast Claude Haiku classifier call decides:
  1) WHICH agent should answer (department)
  2) WHAT KIND of request this is (idea/task/bug/improvement/question)
  3) HOW complex it is (drives model choice)
  4) Whether it needs a formatted document output
  5) Whether it needs multiple agents to collaborate

Stage 2 (bot.py): the chosen agent's persona answers with Claude Sonnet,
using the chat's own scoped history.

STICKY ROUTING: short, content-free follow-ups ("ok", "continue", "davom et")
reuse the previous turn's agent/type rather than misrouting.
"""

import json
import logging
from dataclasses import dataclass, field

from agents import AGENTS, DEFAULT_AGENT_KEY, Agent, get_agent
from config import settings
from llm_clients import claude_generate_json
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
You are a routing classifier for a Senior Business Analyst's AI team. Read
the user's message (may be in Uzbek, Russian, English, or a professional mix)
and decide the following.

1) WHICH SPECIALIST AGENT should handle it (primary agent):
BA & Analysis team:
- ba                   : business requirements, user stories, acceptance criteria, GAP analysis, stakeholder needs
- data_analyst         : SQL queries, KPIs, data analysis, metrics, Excel formulas, cohort/funnel analysis
- bi_analyst           : dashboards, Power BI, Tableau, Looker, reports, data visualisation for management
- process_analyst      : BPMN, AS-IS/TO-BE process mapping, workflow optimization, RACI, SOPs
- financial_analyst    : P&L, unit economics, forecasting, budgeting, ROI/NPV/IRR, financial models
- market_analyst       : market sizing, SWOT, PESTLE, competitive analysis, customer segmentation, GTM
- requirements_engineer: BRD, FRS, NFR, traceability matrix, use case specs, data dictionary
- data_governance      : data quality, MDM, data governance framework, GDPR, data classification
Product & Delivery team:
- pm                   : product roadmap, backlog, prioritisation (RICE/MoSCoW), OKRs, product metrics
- system_analyst       : system architecture, integrations, API contracts, ERD, data flows
- qa                   : test strategy, test plans, test cases, UAT, bug reports, automation, QA metrics
- project_manager      : project charter, WBS, Gantt, RAID log, status reports, sprint planning
- tech_consultant      : vendor selection, build-vs-buy, tech stack recommendations, digital transformation
Engineering team:
- backend              : server-side code, APIs (REST/GraphQL), database schemas, DDL, query optimisation, auth, caching
- frontend             : React/Vue/Angular components, hooks, state management, Tailwind, a11y, Web Vitals
- devops               : Dockerfile, Kubernetes, CI/CD (GitHub Actions/GitLab CI), Terraform, AWS/Azure/GCP, observability
- product_designer     : UX research, personas, journey maps, wireframes, design system, accessibility specs, handoff
Specialised team:
- translator           : professional translation EN ↔ RU ↔ UZ, business/technical documents, UI copy
- diagram              : Mermaid BPMN/UML/ER/Sequence/Activity/Architecture diagrams
- technical_analyst    : API/Swagger/OpenAPI review, JSON/XML analysis, log analysis, REST critique
- jira                 : Jira Epic/Feature/Story/Task/Bug formatting with AC, story points, labels
Banking domain:
- credit_conveyor      : loan lifecycle (online/offline), credit pipeline stages, KYC/AML, decision engine
- core_banking         : accounts, transactions, GL/posting, core banking operations, accounting
- integration          : REST/SOAP/Kafka/ESB integration design, OpenAPI spec, event-driven architecture
- scoring              : credit scorecard, decision rules, PD/LGD/EAD, fraud detection, IFRS 9
- insurance            : credit life insurance, premium calculation, policy lifecycle, claims processing

2) WHAT KIND OF REQUEST:
- idea        : conceptual suggestion, high-level proposal, brainstorm
- task        : concrete deliverable to produce (document, analysis, SQL, model, etc.)
- bug         : error, broken process, data quality issue, system problem
- improvement : refining existing work, improving a process or document
- question    : information/advice request, no concrete deliverable

3) COMPLEXITY:
- "low"  : quick answer, simple query, brief explanation
- "high" : non-trivial analysis, multi-part deliverable, complex reasoning

4) DOES THIS NEED A FORMATTED DOCUMENT (Word/PDF)?
Set wants_document=true for commercial proposals, business cases, formal
reports, policies, FRS/BRD documents, or when user asks for "Word", "PDF",
"hujjat", "tijorat taklifi", "hisobot", "tahlil hisoboti".

5) IS THIS A CAPABILITY QUESTION?
Set is_capability_question=true ONLY when asking what the assistant/team can do.

6) COLLABORATORS (0-3 other agents for PARALLEL input on this request)?
Use when multiple specialists must each contribute simultaneously.
Examples:
  - New product feature brainstorm: ba + pm + system_analyst
  - Financial reporting dashboard: financial_analyst + bi_analyst + data_analyst
  - Market entry analysis: market_analyst + financial_analyst + pm
  - Data platform governance: data_governance + data_analyst + system_analyst
Max 3 collaborators, exclude the primary agent. Leave [] if execution_chain is set.

7) SEQUENTIAL CHAIN (for complex multi-domain requests needing step-by-step pipeline)?
Use execution_chain when one specialist must BUILD ON the previous specialist's output
(not just parallel contributions). Keep to 2-4 agents. Leave [] for simple requests.
IMPORTANT: if execution_chain is non-empty, set collaborators to [].
Examples:
  - Full feature spec end-to-end: ["ba", "system_analyst", "technical_analyst", "qa"]
  - Banking loan product design: ["ba", "credit_conveyor", "integration", "scoring"]
  - New API design: ["ba", "system_analyst", "technical_analyst", "backend"]
  - Process redesign with diagrams: ["process_analyst", "diagram", "requirements_engineer"]
  - Translation request / simple question → []

Also write a short TITLE (max 70 chars) in the SAME language as the user's message.

Respond with ONLY a JSON object, no prose:
{{"agent": "<one of: {", ".join(_AGENT_KEYS)}>",
  "request_type": "<one of: {", ".join(_TYPE_KEYS)}>",
  "complexity": "low"|"high",
  "wants_document": true|false,
  "is_capability_question": true|false,
  "collaborators": ["<0-3 agent keys>"],
  "execution_chain": ["<0-4 agent keys in sequence>"],
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
    execution_chain: list[str] = field(default_factory=list)


def model_for(agent: Agent, complexity: str) -> tuple[str, str]:
    """Return (model_id, label) based on active provider and complexity.

    hybrid: low  → OpenRouter free model (simple questions — saves Claude usage)
            high → Claude Sonnet (quality-critical work: documents, code, analysis)

    llm_clients.claude_generate() routes per-call based on the returned
    model id's shape (an OpenRouter id always contains "/"), so an OR id
    from here correctly reaches OpenRouter's API instead of Claude's.
    """
    if settings.provider == "openrouter":
        if complexity == "low":
            return settings.or_fast_model, settings.or_fast_model_label
        return settings.or_main_model, settings.or_main_model_label
    if settings.provider == "hybrid":
        if complexity == "low":
            return settings.or_fast_model, settings.or_fast_model_label
        return settings.claude_model, settings.claude_model_label
    # claude-only
    if complexity == "low":
        return settings.claude_fast_model, settings.claude_fast_model_label
    return settings.claude_model, settings.claude_model_label


async def classify(
    user_text: str,
    last_agent: str | None = None,
    last_type: str | None = None,
) -> Route:
    """Classify a message and return its Route. Never raises."""
    agent_key = last_agent if last_agent in AGENTS else DEFAULT_AGENT_KEY
    type_key = last_type if last_type in REQUEST_TYPES else DEFAULT_REQUEST_TYPE_KEY
    complexity = "high"
    wants_document = False
    is_capability_question = False
    collaborators: list[str] = []
    execution_chain: list[str] = []
    title = (user_text.strip()[:70] or "Untitled")

    context_hint = ""
    if last_agent in AGENTS:
        context_hint = (
            "\nCONVERSATION CONTEXT: previous agent was "
            f"'{last_agent}', type '{type_key}'. "
            "Keep same agent/type for short follow-ups (ok, davom et, etc). "
            "Reclassify fresh if the topic clearly changes.\n"
        )

    try:
        raw = await claude_generate_json(
            _ROUTER_SYSTEM + context_hint,
            [{"role": "user", "content": user_text[:4000]}],
        )
        data = json.loads(raw)

        candidate = str(data.get("agent", "")).strip().lower()
        if candidate in AGENTS:
            agent_key = candidate

        type_candidate = str(data.get("request_type", "")).strip().lower()
        if type_candidate in REQUEST_TYPES:
            type_key = type_candidate

        complexity = "low" if str(data.get("complexity", "")).lower() == "low" else "high"
        wants_document = bool(data.get("wants_document", False))
        is_capability_question = bool(data.get("is_capability_question", False))

        raw_collabs = data.get("collaborators", [])
        raw_chain = data.get("execution_chain", [])
        if isinstance(raw_chain, list) and raw_chain:
            seen_chain: list[str] = []
            for c in raw_chain:
                ck = str(c).strip().lower()
                if ck in AGENTS and ck not in seen_chain:
                    seen_chain.append(ck)
                if len(seen_chain) >= 4:
                    break
            execution_chain = seen_chain

        if not execution_chain and isinstance(raw_collabs, list):
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

    except Exception as exc:
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
        execution_chain=execution_chain,
    )
