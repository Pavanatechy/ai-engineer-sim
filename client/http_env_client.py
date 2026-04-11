"""
HTTP client for OpenEnv environments.
Provides base class for connecting to OpenEnv servers over HTTP.
"""

from __future__ import annotations

import requests
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from env.base import Action, EnvState, Observation, Reward, StepResult


# Type variables for generic client
ActionT = TypeVar("ActionT", bound=Action)
ObservationT = TypeVar("ObservationT", bound=Observation)


class HTTPEnvClient(ABC, Generic[ActionT, ObservationT]):
    """
    Abstract base class for HTTP clients that connect to OpenEnv servers.

    Subclasses must implement:
        _step_payload(action) -> dict: Convert action to JSON payload
        _parse_result(payload) -> StepResult: Parse server response
    """

    def __init__(self, base_url: str, task_id: str | None = None):
        """
        Initialize the HTTP client.

        Args:
            base_url: Base URL of the OpenEnv server (e.g., "http://localhost:8000")
            task_id: Optional task ID to use for reset
        """
        self.base_url = base_url.rstrip("/")
        self.task_id = task_id
        self._session: dict[str, Any] = {}

    def reset(self) -> StepResult[ObservationT]:
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

        return self._parse_result(data)

    def step(self, action: ActionT) -> StepResult[ObservationT]:
        """Execute an action and return the result."""
        if not self._session:
            raise RuntimeError("Call reset() before step().")

        url = f"{self.base_url}/step"
        payload = {
            "episode_id": self._session["episode_id"],
            "action": self._step_payload(action),
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        return self._parse_result(data)

    def state(self) -> EnvState:
        """Get the current environment state."""
        if not self._session:
            raise RuntimeError("Call reset() before state().")

        url = f"{self.base_url}/state"
        params = {"episode_id": self._session["episode_id"]}

        response = requests.get(url, params=params)
        response.raise_for_status()

        data = response.json()
        return EnvState(**data)

    @abstractmethod
    def _step_payload(self, action: ActionT) -> dict[str, Any]:
        """Convert typed action to JSON payload for HTTP request."""
        ...

    @abstractmethod
    def _parse_result(self, payload: dict[str, Any]) -> StepResult[ObservationT]:
        """Parse HTTP response JSON into typed StepResult."""
        ...