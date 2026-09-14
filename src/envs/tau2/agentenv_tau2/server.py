"""
FastAPI server for the tau2-bench agent environment. Mirrors the webshop/appworld
server contract so the AgentGym client/controller can talk to it unchanged:
  POST /create                        -> env_idx (int)
  POST /reset  {env_idx, session_id}  -> opening observation (str)
  POST /step   {env_idx, action}      -> {state, reward, done, info}
  GET  /observation?env_idx=          -> str
  GET  /system_prompt                 -> str (domain policy + tool list + format spec)
"""

import logging
import time
from typing import List, Optional

from fastapi import FastAPI, Request

from .environment import tau2_env_server
from .model import ResetQuery, StepQuery, StepResponse
from .utils import debug_flg

app = FastAPI(debug=debug_flg)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")


@app.middleware("http")
async def log_request_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logging.info(
        f"{request.client.host} - {request.method} {request.url.path} - "
        f"{response.status_code} - {process_time:.2f}s"
    )
    return response


@app.get("/", response_model=str)
async def generate_ok():
    return "ok"


@app.get("/list_envs", response_model=List[int])
async def list_envs():
    return list(tau2_env_server.env.keys())


@app.get("/debug_threads")
async def debug_threads():
    """Live thread counts. Each active episode owns one orchestrator thread, so this is
    how a leak of abandoned episodes shows up (see scripts/smoke_tau2_leak.py)."""
    import threading

    names = [t.name for t in threading.enumerate()]
    return {
        "total_threads": len(names),
        "orchestrator_threads": sum(1 for n in names if n.startswith("Thread-")),
        "live_env_slots": len(tau2_env_server.env),
    }


@app.get("/config")
async def config():
    """Effective server configuration. Callers running an A/B should assert on this:
    a server that failed to bind leaves the previous variant's process serving the
    port, and the health check cannot tell the difference."""
    s = tau2_env_server
    return {
        "domain": s.domain,
        "domains": s.get_domains(),
        "split": s.split,
        "prompt_variant": s.prompt_variant,
        "reward_shape": s.reward_shape,
        "reward_basis": s.reward_eval_type.value,
        "dense_weight": s.dense_weight,
        "force_done_after": s.force_done_after,
        "max_steps": s.max_steps,
        "solo_mode": s.solo_mode,
        "n_tasks": len(s._task_ids),
    }


@app.get("/system_prompt", response_model=str)
async def system_prompt(domain: Optional[str] = None):
    """Under mixed-domain training each task carries its own policy and tool list, so
    the client must ask per domain; omitting it yields the first configured domain."""
    return tau2_env_server.get_system_prompt(domain)


@app.get("/domains", response_model=List[str])
async def domains():
    return tau2_env_server.get_domains()


@app.post("/create", response_model=int)
async def create():
    return tau2_env_server.create()


# NOTE: /reset and /step are sync `def`, so FastAPI runs them in its threadpool rather
# than on the event loop. Both block: AgentGymEnv hands the action to a background
# orchestrator thread and waits on a threading.Event for the next observation, which in
# turn waits on a user-simulator LLM call. Running them on the event loop would stall
# every other request on this worker for the duration of that call.
@app.post("/reset", response_model=str)
def reset(reset_query: ResetQuery):
    return tau2_env_server.reset(reset_query.env_idx, reset_query.session_id)


@app.post("/step", response_model=StepResponse)
def step(step_query: StepQuery):
    state, reward, done, info = tau2_env_server.step(
        step_query.env_idx, step_query.action
    )
    return StepResponse(state=state, reward=reward, done=done, info=info)


@app.get("/observation", response_model=str)
def observation(env_idx: int):
    return tau2_env_server.observation(env_idx)


@app.post("/close", response_model=str)
def close(reset_query: ResetQuery):
    tau2_env_server.close(reset_query.env_idx)
    return "ok"
