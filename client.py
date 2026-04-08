"""
Client for the Loan Advisor OpenEnv environment.
Use this to interact with a running environment server from Python.
"""
from __future__ import annotations

from models import LoanAdvisorAction, LoanAdvisorObservation

# Try openm-core first (official CLI package), then openenv-core, then fallback
_EnvClient = None
try:
    from openm.core.env_client import EnvClient as _EnvClient  # type: ignore
except ImportError:
    try:
        from openenv.core.env_client import EnvClient as _EnvClient  # type: ignore
    except ImportError:
        pass


if _EnvClient is not None:
    class LoanAdvisorEnv(_EnvClient):  # type: ignore[misc,valid-type]
        """
        Python client for the Loan Advisor environment server.

        Usage:
            env = LoanAdvisorEnv(base_url="https://<your-hf-space>.hf.space")
            obs = await env.reset(task_id="task_easy")
            result = await env.step(LoanAdvisorAction(
                action_type="query_info",
                query_field="tuition_fees",
            ))
        """
        action_type = LoanAdvisorAction
        observation_type = LoanAdvisorObservation

        def __init__(self, base_url: str = "http://localhost:7860") -> None:
            super().__init__(base_url=base_url)

else:
    import requests

    class LoanAdvisorEnv:  # type: ignore[no-redef]
        """
        Fallback sync HTTP client (used when openm-core / openenv-core is not installed).
        """

        def __init__(self, base_url: str = "http://localhost:7860") -> None:
            self.base_url = base_url.rstrip("/")

        def reset(self, task_id: str = None) -> LoanAdvisorObservation:
            body = {"task_id": task_id} if task_id else {}
            resp = requests.post(f"{self.base_url}/reset", json=body, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return LoanAdvisorObservation(**data.get("observation", data))

        def step(self, action: LoanAdvisorAction) -> tuple:
            resp = requests.post(
                f"{self.base_url}/step",
                json=action.model_dump(exclude_none=True),
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            obs = LoanAdvisorObservation(**data.get("observation", data))
            return obs, data.get("reward", 0.0), data.get("done", False), data.get("info", {})

        def state(self) -> dict:
            resp = requests.get(f"{self.base_url}/state", timeout=30)
            resp.raise_for_status()
            return resp.json()
