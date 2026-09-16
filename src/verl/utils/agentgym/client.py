from contextlib import contextmanager
import itertools
import os
import time
from agentenv.envs import (
    AcademiaEnvClient,
    AlfWorldEnvClient,
    AppWorldEnvClient,
    BabyAIEnvClient,
    MazeEnvClient,
    MovieEnvClient,
    SciworldEnvClient,
    SheetEnvClient,
    SqlGymEnvClient,
    TextCraftEnvClient,
    TodoEnvClient,
    WeatherEnvClient,
    WebarenaEnvClient,
    WebshopEnvClient,
    WordleEnvClient,
    SearchQAEnvClient,
)

# tau2 is the one environment this repo carries itself (src/envs/tau2) rather than
# taking from the AgentGym submodule, so import it from there. Fall back to AgentGym
# for anyone running against a checkout that has it upstream.
try:
    from envs.tau2.tau2_client import Tau2EnvClient
except ImportError:  # pragma: no cover
    from agentenv.envs import Tau2EnvClient

import torch.distributed as dist

_ADDR_RR = itertools.count()   # round-robin cursor within this rank's server shard


def _select_env_addr(env_addr_value):
    """Pick an env-server address for ONE env client, sharded by distributed rank.

    A rank opens (batch_per_rank * rollout_n) env clients per rollout, all of which
    step concurrently. The old logic gave every rank a single address, so all of that
    concurrency funnelled into one server process -- and since env.step is pure-Python
    behind a sync FastAPI endpoint, the GIL serialised it and left most cores idle.

    Now each rank gets a CONTIGUOUS SHARD of the address list (len(addrs)//world_size
    servers) and its clients round-robin over that shard, so one GPU's env stepping
    spreads across several server processes / cores.

    Backward compatible: when len(addrs) == world_size the shard is exactly one
    address, reproducing the previous addrs[rank % len(addrs)] behaviour.
    """
    addrs = [a.strip() for a in str(env_addr_value).split(",") if a.strip()]
    if len(addrs) <= 1:
        return addrs[0] if addrs else env_addr_value

    if dist.is_initialized():
        rank, world_size = dist.get_rank(), dist.get_world_size()
    else:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    world_size = max(1, world_size)

    per_rank = len(addrs) // world_size
    if per_rank >= 1:
        shard = addrs[rank * per_rank:(rank + 1) * per_rank]
    else:
        # fewer servers than ranks -> fall back to sharing (old behaviour)
        shard = [addrs[rank % len(addrs)]]

    selected_addr = shard[next(_ADDR_RR) % len(shard)]
    return selected_addr

def init_env_client(args):
    # task_name - task dict
    envclient_classes = {
        "webshop": WebshopEnvClient,
        "alfworld": AlfWorldEnvClient,
        "appworld": AppWorldEnvClient,
        "babyai": BabyAIEnvClient,
        "sciworld": SciworldEnvClient,
        "tau2": Tau2EnvClient,
        "textcraft": TextCraftEnvClient,
        "webarena": WebarenaEnvClient,
        "sqlgym": SqlGymEnvClient,
        "maze": MazeEnvClient,
        "wordle": WordleEnvClient,
        "weather": WeatherEnvClient,
        "todo": TodoEnvClient,
        "movie": MovieEnvClient,
        "sheet": SheetEnvClient,
        "academia": AcademiaEnvClient,
        "searchqa": SearchQAEnvClient,
    }
    # select task according to the name
    envclient_class = envclient_classes.get(args.task_name.lower(), None)
    if envclient_class is None:
        raise ValueError(f"Unsupported task name: {args.task_name}")
    
    # Handle multiple comma-separated environment addresses
    selected_env_addr = _select_env_addr(args.env_addr)
    
    retry = 0
    while True:
        try:
            env_client = envclient_class(env_server_base=selected_env_addr, data_len=1, timeout=2400)
            break
        except Exception as e:
            retry += 1
            print(f"Failed to connect to env server {selected_env_addr}, retrying...({retry}/{args.max_retries})")
            if retry > args.max_retries:
                raise e
            time.sleep(5)
    return env_client