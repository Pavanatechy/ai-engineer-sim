"""
HTTP client for AI Engineer Simulation environments.
"""

from __future__ import annotations

import requests
from typing import Any

from env.base import Action, Observation, Reward, StepResult

from .http_env_client import HTTPEnvClient


class AIEngineerEnv(HTTPEnvClient[Action, Observation]):
    """
    HTTP client for AI Engineer Simulation tasks.

    Connects to an OpenEnv server running AI coding tasks like
    email triage, code review, and data cleaning.
    """

    def reset(self) -> StepResult[Observation]:
        """Reset the environment and return initial observation."""
        url = f"{self.base_url}/reset"
        payload = {"task_id": self.task_id} if self.task_id else {}

        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        self._session = {
            "episode_id": data["episode_id"],
            "task_id": data["task_id"],
        }

        # For reset, construct a StepResult with default reward and done=False
        observation = Observation(**data["observation"])
        reward = Reward(value=0.0, cumulative=0.0, message="Environment reset")
        return StepResult(
            observation=observation,
            reward=reward,
            done=False,
            info={"episode_id": data["episode_id"]}
        )

    def _step_payload(self, action: Action) -> dict[str, Any]:
        """Convert Action to JSON payload."""
        return action.model_dump()

    def _parse_result(self, payload: dict[str, Any]) -> StepResult[Observation]:
        """Parse server response into StepResult."""
        observation_data = payload["observation"]
        reward_data = payload["reward"]

        observation = Observation(**observation_data)
        reward = Reward(**reward_data)

        return StepResult(
            observation=observation,
            reward=reward,
            done=payload["done"],
            info=payload.get("info", {}),
        )