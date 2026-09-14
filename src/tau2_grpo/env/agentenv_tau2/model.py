from typing import Optional

from pydantic import BaseModel


class StepQuery(BaseModel):
    env_idx: int
    action: str


class StepResponse(BaseModel):
    state: str
    reward: float
    done: bool
    info: None


class ResetQuery(BaseModel):
    env_idx: int
    session_id: Optional[int] = None
