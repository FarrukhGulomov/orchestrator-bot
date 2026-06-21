"""
Optional GitHub integration.

Turns actionable requests (task / bug / improvement) into tracked GitHub
Issues, and — if GITHUB_AUTO_PR is enabled — opens a DRAFT Pull Request on a
new branch with the AI-generated implementation.

Fully optional and safe-by-default:
  * Every public function is a no-op (returns None) unless GITHUB_TOKEN and
    GITHUB_REPO are configured.
  * PR creation additionally requires GITHUB_AUTO_PR=true and at least one
    file-marked code block in the agent's answer.
  * PRs are always opened as drafts with an explicit "review before merging"
    note — this bot never merges anything itself.

Never put the token anywhere but .env on your own server.
"""

import asyncio
import base64
import logging
import re
import time

import requests

from config import settings

logger = logging.getLogger(__name__)

API = "https://api.github.com"
_TIMEOUT = 20


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _slug(text: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return (s or "update")[:max_len]


# --- Issues ------------------------------------------------------------------
def _create_issue_sync(title: str, body: str, labels: list[str]) -> str | None:
    url = f"{API}/repos/{settings.github_repo}/issues"
    payload: dict = {"title": title[:250], "body": body}
    if labels:
        payload["labels"] = labels

    resp = requests.post(url, json=payload, headers=_headers(), timeout=_TIMEOUT)
    if resp.status_code == 422 and labels:
        # Most likely an unknown label name — retry without labels rather than
        # losing the whole ticket over it.
        payload.pop("labels", None)
        resp = requests.post(url, json=payload, headers=_headers(), timeout=_TIMEOUT)

    if resp.status_code >= 300:
        logger.warning("GitHub issue creation failed: %s %s", resp.status_code, resp.text[:300])
        return None
    return resp.json().get("html_url")


async def create_issue(title: str, body: str, labels: list[str]) -> str | None:
    if not settings.github_enabled:
        return None
    try:
        return await asyncio.to_thread(_create_issue_sync, title, body, labels)
    except Exception:
        logger.exception("GitHub issue creation raised")
        return None


# --- Implementation PR ---------------------------------------------------------
# Agents are instructed (see request_types.py addenda) to start every file-ready
# code block with a single marker line, e.g.:
#   ```python
#   # file: app/services/cache.py
#   ...
#   ```
_FILE_BLOCK_RE = re.compile(
    r"```[a-zA-Z0-9_+-]*\n"
    r"(?:#|//|--|<!--)\s*file:\s*(?P<path>[^\n]+?)\s*(?:-->)?\s*\n"
    r"(?P<code>.*?)"
    r"```",
    re.DOTALL | re.IGNORECASE,
)


def extract_files(markdown_text: str) -> dict[str, str]:
    """Pull {path: content} out of fenced code blocks marked with a `file:` line."""
    files: dict[str, str] = {}
    for m in _FILE_BLOCK_RE.finditer(markdown_text):
        path = m.group("path").strip().strip("`")
        code = m.group("code")
        if path:
            files[path] = code
    return files


def _get_file_sha(path: str, branch: str) -> str | None:
    url = f"{API}/repos/{settings.github_repo}/contents/{path}"
    resp = requests.get(url, headers=_headers(), params={"ref": branch}, timeout=_TIMEOUT)
    if resp.status_code == 200:
        return resp.json().get("sha")
    return None


def _create_pr_sync(
    title: str, body: str, files: dict[str, str], labels: list[str]
) -> str | None:
    owner_repo = settings.github_repo
    base = settings.github_default_branch
    headers = _headers()

    base_ref = requests.get(
        f"{API}/repos/{owner_repo}/git/ref/heads/{base}", headers=headers, timeout=_TIMEOUT
    )
    if base_ref.status_code >= 300:
        logger.warning("Cannot read base branch '%s': %s", base, base_ref.text[:300])
        return None
    base_sha = base_ref.json()["object"]["sha"]

    branch = f"bot/{_slug(title)}-{int(time.time())}"
    create_ref = requests.post(
        f"{API}/repos/{owner_repo}/git/refs",
        headers=headers,
        json={"ref": f"refs/heads/{branch}", "sha": base_sha},
        timeout=_TIMEOUT,
    )
    if create_ref.status_code >= 300:
        logger.warning("Cannot create branch '%s': %s", branch, create_ref.text[:300])
        return None

    for path, content in files.items():
        existing_sha = _get_file_sha(path, branch)
        payload = {
            "message": f"{title}\n\nAutomated change via Master Orchestrator bot.",
            "content": base64.b64encode(content.encode("utf-8")).decode("ascii"),
            "branch": branch,
        }
        if existing_sha:
            payload["sha"] = existing_sha
        put_resp = requests.put(
            f"{API}/repos/{owner_repo}/contents/{path}",
            headers=headers,
            json=payload,
            timeout=_TIMEOUT,
        )
        if put_resp.status_code >= 300:
            logger.warning("Cannot write '%s': %s", path, put_resp.text[:300])

    pr_resp = requests.post(
        f"{API}/repos/{owner_repo}/pulls",
        headers=headers,
        json={
            "title": title,
            "body": body
            + "\n\n---\n⚠️ Auto-generated by Master Orchestrator bot. Review carefully before merging.",
            "head": branch,
            "base": base,
            "draft": True,
        },
        timeout=_TIMEOUT,
    )
    if pr_resp.status_code >= 300:
        logger.warning("Cannot open PR from '%s': %s", branch, pr_resp.text[:300])
        return None

    pr = pr_resp.json()
    pr_url = pr.get("html_url")
    number = pr.get("number")
    if labels and number:
        requests.post(
            f"{API}/repos/{owner_repo}/issues/{number}/labels",
            headers=headers,
            json={"labels": labels},
            timeout=_TIMEOUT,
        )
    return pr_url


async def create_implementation_pr(
    title: str, body: str, files: dict[str, str], labels: list[str]
) -> str | None:
    if not settings.github_enabled or not settings.github_auto_pr or not files:
        return None
    try:
        return await asyncio.to_thread(_create_pr_sync, title, body, files, labels)
    except Exception:
        logger.exception("GitHub PR creation raised")
        return None


# --- Reading existing files (the "pull" half of push/pull) -------------------
# Real refinement needs to see what's actually there first — generating a
# "fix" blind, without reading the current file, isn't refinement, it's a
# guess. /readfile in bot.py uses this to pull a file's current content into
# the conversation before asking an agent to change it.
def _get_file_content_sync(path: str, ref: str | None = None) -> str | None:
    url = f"{API}/repos/{settings.github_repo}/contents/{path}"
    params = {"ref": ref} if ref else {}
    resp = requests.get(url, headers=_headers(), params=params, timeout=_TIMEOUT)
    if resp.status_code != 200:
        logger.warning("GitHub get_file_content failed for '%s': %s", path, resp.status_code)
        return None
    data = resp.json()
    if data.get("encoding") == "base64" and "content" in data:
        try:
            return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            logger.exception("Failed to decode content for '%s'", path)
            return None
    return None


async def get_file_content(path: str, ref: str | None = None) -> str | None:
    if not settings.github_enabled:
        return None
    try:
        return await asyncio.to_thread(_get_file_content_sync, path, ref)
    except Exception:  # noqa: BLE001
        logger.exception("GitHub get_file_content raised")
        return None
