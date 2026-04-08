from pydantic import BaseModel
from typing import Optional, Dict

class Observation(BaseModel):
    email_text: str
    step: int

class Action(BaseModel):
    predicted_category: str  # spam | work | personal

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Optional[Dict] = {}