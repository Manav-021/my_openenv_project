---
title: Email OpenEnv
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
# Email Triage OpenEnv
---
## Description
This environment simulates real-world email classification.

## Tasks
- Easy: simple emails
- Medium: moderate ambiguity
- Hard: complex/ambiguous emails

## Actions
Agent predicts:
- spam
- work
- personal

## Reward
- Correct: +1
- Partial: +0.2
- Wrong: -0.5

## Run

```bash
pip install -r requirements.txt
python inference.py