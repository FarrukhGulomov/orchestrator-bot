"""
Optional Railway integration — READ-ONLY by design.

Lets the DevOps/SOC/Tech Lead agents answer "bug izlash" (find bugs) and
"log ko'rish" (view logs) requests using REAL data from this bot's own
Railway deployment, instead of guessing. Uses Railway's public GraphQL API.

WHY READ-ONLY: this module deliberately does NOT expose a deploy-trigger or
service-mutation capability, even though Railway's API supports one. The
existing GitHub flow (a request becomes a draft PR, a human reviews and
merges it, Railway then auto-deploys from the GitHub-linked service) is the
correct path for shipping changes — it keeps a review step between "the AI
wrote this" and "this is running in production". A chat command that
triggers a redeploy directly would skip that review step entirely. If you
want deploy-trigger added later, do it deliberately and probably behind an
explicit confirmation step, not as a side effect of a routine chat message.

⚠️ NOT LIVE-TESTED: Railway's API (backboard.railway.com) isn't reachable
from the sandbox this was built in, so these queries are implemented
strictly per Railway's published schema/docs but have not been exercised
against a real Railway project. Test with /logs after deploying. Railway's
GraphQL errors are specific and actionable (per their own docs) — if a field
name has changed, the error will say exactly what's wrong; adjust the query
strings below accordingly.

Auth: needs RAILWAY_API_TOKEN (Railway -> Account Settings -> Tokens —
NEVER share this in chat). Project/Environment/Service IDs are normally
already present in this bot's own environment (Railway auto-injects
RAILWAY_PROJECT_ID / RAILWAY_ENVIRONMENT_ID / RAILWAY_SERVICE_ID into every
deployed service), so usually only the token needs to be added manually.
"""

import asyncio
import logging

import requests

from config import settings

logger = logging.getLogger(__name__)

API_URL = "https://backboard.railway.com/graphql/v2"
_TIMEOUT = 20

_LATEST_DEPLOYMENT_QUERY = """
query LatestDeployment($projectId: String!, $environmentId: String!, $serviceId: String!) {
  deployments(
    first: 1
    input: { projectId: $projectId, environmentId: $environmentId, serviceId: $serviceId }
  ) {
    edges { node { id status createdAt staticUrl } }
  }
}
"""

_DEPLOYMENT_LOGS_QUERY = """
query DeploymentLogs($deploymentId: String!, $limit: Int) {
  deploymentLogs(deploymentId: $deploymentId, limit: $limit) {
    timestamp
    message
    severity
  }
}
"""


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.railway_api_token}",
        "Content-Type": "application/json",
    }


def _gql_sync(query: str, variables: dict) -> dict | None:
    try:
        resp = requests.post(
            API_URL,
            headers=_headers(),
            json={"query": query, "variables": variables},
            timeout=_TIMEOUT,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Railway API request failed")
        return None
    if resp.status_code >= 300:
        logger.warning("Railway API HTTP error %s: %s", resp.status_code, resp.text[:300])
        return None
    data = resp.json()
    if data.get("errors"):
        logger.warning("Railway GraphQL errors: %s", data["errors"])
        return None
    return data.get("data")


async def get_latest_deployment() -> dict | None:
    """Returns {'id', 'status', 'createdAt', 'staticUrl'} for the most recent
    deployment of THIS bot's own Railway service, or None if unavailable."""
    if not settings.railway_enabled:
        return None
    data = await asyncio.to_thread(
        _gql_sync,
        _LATEST_DEPLOYMENT_QUERY,
        {
            "projectId": settings.railway_project_id,
            "environmentId": settings.railway_environment_id,
            "serviceId": settings.railway_service_id,
        },
    )
    if not data:
        return None
    edges = (data.get("deployments") or {}).get("edges") or []
    return edges[0]["node"] if edges else None


async def get_recent_logs(limit: int = 300) -> list[dict] | None:
    """Returns recent runtime log entries for the latest deployment (each a
    dict with 'timestamp', 'message', 'severity'), or None if unavailable."""
    if not settings.railway_enabled:
        return None
    deployment = await get_latest_deployment()
    if not deployment:
        return None
    data = await asyncio.to_thread(
        _gql_sync,
        _DEPLOYMENT_LOGS_QUERY,
        {"deploymentId": deployment["id"], "limit": limit},
    )
    if not data:
        return None
    return data.get("deploymentLogs") or []
