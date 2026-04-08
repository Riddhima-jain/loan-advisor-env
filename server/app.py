"""
FastAPI server for the Loan Advisor OpenEnv environment.

Standard OpenEnv endpoints:
  POST /reset  — start a new episode (body: {"task_id": "task_easy"} optional)
  POST /step   — agent takes an action
  GET  /state  — return current internal state
  GET  /tasks  — list available tasks
  GET  /health — health check
  GET  /       — environment info
"""
from __future__ import annotations

import sys
import os

# Ensure project root is importable from server/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import LoanAdvisorAction, LoanAdvisorObservation
from server.environment import LoanAdvisorEnvironment, TASKS, TASK_ORDER

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Loan Advisor Environment",
    description=(
        "OpenEnv-compatible RL environment for education loan decision-making. "
        "Agents evaluate Go/No-go on education loans based on ROI, affordability, "
        "and course/university lookup data. "
        "Initial tasks: India (INR). Architecture is globally extensible."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

env = LoanAdvisorEnvironment()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: Optional[str] = None

    class Config:
        extra = "allow"


class StepRequest(BaseModel):
    action_type: str
    query_field: Optional[str] = None
    loan_ids: Optional[list[str]] = None
    calculation_type: Optional[str] = None
    loan_id: Optional[str] = None
    recommended_decision: Optional[str] = None
    recommended_loan_id: Optional[str] = None
    reasoning: Optional[str] = None

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "environment": "loan_advisor_env", "version": "1.0.0"}


@app.get("/")
async def root() -> dict:
    return {
        "name": "loan_advisor_env",
        "version": "1.0.0",
        "description": "OpenEnv environment for education loan decision-making.",
        "tasks": TASK_ORDER,
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health"],
    }


@app.post("/reset")
async def reset_env(request: ResetRequest = None) -> dict:
    """
    Start a new episode.
    Body: {"task_id": "task_easy"} — optional. Omit to cycle easy→medium→hard→easy.
    """
    try:
        task_id = request.task_id if request else None
        obs = env.reset(task_id=task_id)
        return obs.model_dump()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/step")
async def step_env(request: StepRequest) -> dict:
    """
    Execute one agent action and return the step result.

    Returns: {"observation": {...}, "reward": 0.00, "done": false, "info": {...}}
    """
    try:
        action = LoanAdvisorAction(**request.model_dump(exclude_none=False))
        obs, reward, done, info = env.step(action)
        return {
            "observation": obs.model_dump(),
            "reward": reward,
            "done": done,
            "info": info,
        }
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/state")
async def get_state() -> dict:
    """Return current internal environment state (for debugging/monitoring)."""
    return env.state()


@app.get("/tasks")
async def list_tasks() -> dict:
    """List available tasks with metadata."""
    difficulty_map = {0: "easy", 1: "medium", 2: "hard"}
    return {
        "tasks": [
            {
                "id": tid,
                "difficulty": difficulty_map.get(i, "unknown"),
                "max_steps": TASKS[tid]["max_steps"],
                "correct_decision": TASKS[tid]["correct_decision"],
                "description": TASKS[tid]["description"][:120] + "...",
            }
            for i, tid in enumerate(TASK_ORDER)
        ]
    }
