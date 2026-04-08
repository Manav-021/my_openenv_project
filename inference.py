import asyncio
import os
from typing import List
from openai import OpenAI

from env.email_env import EmailEnv
from env.models import Action
from env.grader import compute_score

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

TASK = os.getenv("TASK", "easy")
MAX_STEPS = 10

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success, steps, score, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action(email_text: str):
    prompt = f"Classify this email into spam, work, or personal:\n{email_text}"
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=20,
        )
        return res.choices[0].message.content.strip().lower()
    except:
        return "personal"


async def main():
    env = EmailEnv(task_name=TASK)

    rewards: List[float] = []
    steps = 0

    log_start(TASK, "email_env", MODEL_NAME)

    result = await env.reset()

    try:
        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            email = result.observation.email_text
            action_str = get_action(email)

            result = await env.step(Action(predicted_category=action_str))

            reward = result.reward
            done = result.done

            rewards.append(reward)
            steps = step

            log_step(step, action_str, reward, done, None)

            if done:
                break

        score = compute_score(sum(rewards), len(rewards))
        success = score > 0.5

    finally:
        await env.close()
        log_end(success, steps, score, rewards)


if __name__ == "__main__":
    asyncio.run(main())