"""
Request-type classification: IDEA / BUG / REFINEMENT / TASK (+ QUESTION extension).

This is the "Input Classification Matrix" — a second axis on top of agent
routing (which department answers). It answers "what kind of thing is this,
and should it turn into a tracked deliverable". Each type carries:
  - classification_code : the exact value written into the CLASSIFICATION
                           field of the [X_ORCHESTRATOR_METADATA] header
  - label                : friendly emoji label (used by the compact "simple"
                           header format, see config.METADATA_FORMAT)
  - creates_ticket       : whether it should be filed as a GitHub issue/PR
  - github_label         : GitHub label name to attach to that issue/PR
  - addendum             : system-prompt addition that pushes the agent to
                            actually DELIVER the implementation, not discuss it

NOTE — deliberate extension beyond the 4-bucket spec:
The canonical matrix only defines IDEA / BUG / REFINEMENT / TASK. We keep a
5th, non-actionable "question" type for plain chat (greetings, "how does X
work", general advice). Without it, every casual message in a group would get
forced into one of the 4 buckets and — when GitHub integration is on — filed
as a ticket, which would flood the repo with noise. QUESTION never creates a
ticket. If you want strict 4-bucket-only behaviour, see ROUTER_STRICT_4_BUCKET
in config.py / .env.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class RequestType:
    key: str
    classification_code: str
    label: str
    creates_ticket: bool
    github_label: str | None
    addendum: str


REQUEST_TYPES: dict[str, RequestType] = {
    "idea": RequestType(
        key="idea",
        classification_code="IDEA",
        label="💡 Idea",
        creates_ticket=True,
        github_label="idea",
        addendum="""
REQUEST TYPE: IDEA / PROPOSAL.
Turn this into an actionable proposal, not an abstract discussion:
1. Problem / opportunity (1-2 sentences)
2. Proposed solution (concrete)
3. Scope: explicitly in vs. out
4. Key risks or open questions
5. Recommended next step and which role should own it
Keep it short and decision-ready — this will be filed as a tracked ticket.
""",
    ),
    "task": RequestType(
        key="task",
        classification_code="TASK",
        label="🛠 Task",
        creates_ticket=True,
        github_label="task",
        addendum="""
REQUEST TYPE: TASK / FEATURE (implementation request).
Actually deliver this now — do not just describe what you would do.
- Code roles (backend/frontend/devops/soc/tech_lead): produce complete,
  ready-to-apply code. For every file you deliver, the FIRST line inside its
  fenced code block must be a marker comment with nothing else on it:
      # file: relative/path/to/file.ext   (use correct comment syntax for the language)
  After the code, briefly note setup/migration steps and tests if relevant.
- Non-code roles (PM/BA/System Analyst/QA): produce the finished artifact
  itself (the actual user stories / test plan / spec section), not a
  description of what you would write.
This will be filed as a tracked ticket — make it self-contained.
""",
    ),
    "bug": RequestType(
        key="bug",
        classification_code="BUG",
        label="🐞 Bug",
        creates_ticket=True,
        github_label="bug",
        addendum="""
REQUEST TYPE: BUG.
Fix it now — don't just describe the problem.
1. Likely root cause (state assumptions if info is incomplete).
2. The actual fix. Code roles: complete corrected code using the file-marker
   convention (first line of each fenced block: `# file: path/to/file.ext`,
   correct comment syntax for the language). QA: a structured bug report
   (Steps to Reproduce, Expected vs Actual, Severity, suggested owner).
3. Regression risk / what to verify after applying the fix.
This will be filed as a tracked ticket — be precise and concrete.
""",
    ),
    "improvement": RequestType(
        key="improvement",
        classification_code="REFINEMENT",
        label="⚙️ Refinement",
        creates_ticket=True,
        github_label="enhancement",
        addendum="""
REQUEST TYPE: REFINEMENT / ДОРАБОТКА (improvement to existing code/system).
Deliver the improved version now, not a discussion of options.
- Code roles: complete updated code using the file-marker convention (first
  line of each fenced block: `# file: path/to/file.ext`). Briefly explain the
  rationale and any migration/compatibility notes.
- Non-code roles: the improved artifact itself (updated spec, updated test
  plan, etc.).
This will be filed as a tracked ticket.
""",
    ),
    "question": RequestType(
        key="question",
        classification_code="QUESTION",
        label="❓ Question",
        creates_ticket=False,
        github_label=None,
        addendum="",
    ),
}

DEFAULT_REQUEST_TYPE_KEY = "question"


def get_request_type(key: str) -> RequestType:
    return REQUEST_TYPES.get(key, REQUEST_TYPES[DEFAULT_REQUEST_TYPE_KEY])
