from fastapi import FastAPI
import uvicorn

from env.email_env import EmailEnv
from env.models import Action

app = FastAPI()
env = EmailEnv()

@app.post("/reset")
async def reset():
    return (await env.reset()).dict()

@app.post("/step")
async def step(action: Action):
    return (await env.step(action)).dict()

@app.get("/state")
async def state():
    return await env.state()


# ✅ REQUIRED for OpenEnv
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


# ✅ REQUIRED entrypoint
if __name__ == "__main__":
    main()