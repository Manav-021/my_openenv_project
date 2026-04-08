import random
from env.models import Observation, Action, StepResult
from env.tasks import TASKS


class EmailEnv:

    def __init__(self, task_name=None):
        if task_name is None:
            self.task_name = random.choice(list(TASKS.keys()))
        else:
            self.task_name = task_name

        self.emails = TASKS[self.task_name]   
        self.index = 0
        self.total_steps = len(self.emails)

    async def reset(self):
        self.index = 0
        email, _ = self.emails[self.index]

        return StepResult(
            observation=Observation(email_text=email, step=0),
            reward=0.0,
            done=False,
        )

    async def step(self, action: Action):
        email, true_label = self.emails[self.index]

        # reward shaping
        if action.predicted_category == true_label:
            reward = 0.9
        elif action.predicted_category in ["spam", "work", "personal"]:
            reward = 0.5  # partial correctness
        else:
            reward = 0.1  # invalid action

        self.index += 1
        done = self.index >= self.total_steps

        if not done:
            next_email, _ = self.emails[self.index]
        else:
            next_email = ""

        return StepResult(
            observation=Observation(email_text=next_email, step=self.index),
            reward=reward,
            done=done,
        )

    async def state(self):
        return {
            "task": self.task_name,
            "current_step": self.index
        }

    async def close(self):
        pass