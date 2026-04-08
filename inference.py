from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, Body, HTTPException
from pydantic import BaseModel

import tasks  # noqa: F401
from env.base import Action, Observation, EnvState, StepResult
from env.registry import TaskRegistry

app = FastAPI(
    title="OpenEnv Inference Server",
    description="Minimal OpenEnv-compatible inference endpoint for task reset, step, and state.",
    version="1.0.0",
)

_sessions: dict[str, Any] = {}


def _serialize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value


class ResetRequest(BaseModel):
    task_id: str | None = None


class StepRequest(BaseModel):
    episode_id: str
    action: Action


@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "message": "OpenEnv inference server is running.",
        "tasks": TaskRegistry.all_tasks(),
        "default_task": TaskRegistry.all_tasks()[0] if TaskRegistry.all_tasks() else None,
    }


@app.post("/openenv/reset")
@app.post("/reset")
async def reset(request: ResetRequest = Body(default_factory=ResetRequest)) -> dict[str, Any]:
    task_id = request.task_id or (TaskRegistry.all_tasks()[0] if TaskRegistry.all_tasks() else None)
    if task_id is None:
        raise HTTPException(status_code=400, detail="No registered tasks available.")

    if task_id not in TaskRegistry.all_tasks():
        raise HTTPException(
            status_code=404,
            detail={"error": "Task not found", "available_tasks": TaskRegistry.all_tasks()},
        )

    env = TaskRegistry.instantiate(task_id)
    observation = env.reset()
    _sessions[env.state().episode_id] = env

    return {
        "episode_id": env.state().episode_id,
        "task_id": task_id,
        "observation": _serialize(observation),
    }


@app.post("/openenv/step")
@app.post("/step")
async def step(request: StepRequest) -> dict[str, Any]:
    episode_id = request.episode_id
    env = _sessions.get(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found.")

    try:
        result = env.step(request.action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return {
        "episode_id": episode_id,
        "observation": _serialize(result.observation),
        "reward": _serialize(result.reward),
        "done": result.done,
        "info": _serialize(result.info),
    }


@app.get("/openenv/state")
@app.get("/state")
async def state(episode_id: str) -> dict[str, Any]:
    env = _sessions.get(episode_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Episode '{episode_id}' not found.")
    return _serialize(env.state())


@app.get("/openenv/tasks")
@app.get("/tasks")
async def tasks() -> dict[str, Any]:
    return {"tasks": TaskRegistry.all_tasks()}
