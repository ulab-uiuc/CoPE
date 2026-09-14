#!/usr/bin/env python
"""Minimal OpenAI-compatible stub that stands in for the tau2 user-simulator LLM.

Test-only. Lets the tau2 env server be exercised end-to-end on a machine with no GPU
and no API access: it answers /v1/chat/completions with canned text, so the
orchestrator, tool execution and evaluator all run for real while the only LLM in the
loop is fake.

    python scripts/tau2_fake_usersim.py --port 38001

Replies are chosen from the length of the incoming conversation, not from a global
counter -- the env server runs many episodes concurrently against one of these, so any
shared mutable turn count would bleed between them (and across runs).
"""

import argparse
import time

import uvicorn
from fastapi import FastAPI, Request

app = FastAPI()


class Cfg:
    max_turns = 4


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": "user-sim", "object": "model"}]}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages") or []
    # Everything but the system prompt is conversation so far; the agent has spoken
    # roughly this many times.
    turns = sum(1 for m in messages if m.get("role") != "system")

    if turns <= 1:
        content = "Hi, I need help with my order. My name is Yusuf Rossi, zip 19122."
    elif turns < Cfg.max_turns * 2:
        content = "Yes, that's right. Please go ahead."
    else:
        content = "###STOP###"

    return {
        "id": f"chatcmpl-fake-{turns}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "user-sim"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=38001)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--turns", type=int, default=4)
    args = parser.parse_args()
    Cfg.max_turns = args.turns
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
