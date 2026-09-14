"""Benchmark ONE rank's env-side load for a full rollout batch.

Mirrors verl/workers/rollout/agent_vllm_rollout/vllm_rollout.py: a rank opens
(batch_per_rank * rollout_n) env clients, then each round steps all unfinished ones
concurrently via ThreadPoolExecutor. Clients are assigned to servers round-robin,
exactly like _select_env_addr's per-rank shard.

usage: bench_webshop_env.py <addr1,addr2,...> <n_clients> <n_rounds> <label>
"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

addrs = [a.strip() for a in sys.argv[1].split(",") if a.strip()]
N_CLIENTS = int(sys.argv[2])
N_ROUNDS = int(sys.argv[3])
LABEL = sys.argv[4] if len(sys.argv) > 4 else "run"

_local = threading.local()


def sess():
    if not hasattr(_local, "s"):
        s = requests.Session()
        s.trust_env = False           # never proxy localhost
        _local.s = s
    return _local.s


QUERIES = ["men running shoes", "blue cotton shirt", "wireless headphones",
           "stainless steel water bottle", "organic green tea", "leather wallet",
           "yoga mat", "usb c cable"]

# --- create env instances (round-robin over this rank's server shard) ---
t0 = time.time()
clients = []
for i in range(N_CLIENTS):
    a = addrs[i % len(addrs)]
    eid = sess().post(f"{a}/create", timeout=900).json()
    clients.append((a, eid))
t_create = time.time() - t0


def do_reset(i):
    a, eid = clients[i]
    sess().post(f"{a}/reset", json={"env_idx": eid, "session_id": i}, timeout=900)


t0 = time.time()
with ThreadPoolExecutor(max_workers=N_CLIENTS) as ex:
    list(ex.map(do_reset, range(N_CLIENTS)))
t_reset = time.time() - t0


def do_step(args):
    i, rnd = args
    a, eid = clients[i]
    # alternate search / back-to-search so the (CPU-heavy) BM25 search keeps firing,
    # which is what actually loads the env server
    act = (f"search[{QUERIES[(i + rnd) % len(QUERIES)]}]" if rnd % 2 == 0
           else "click[Back to Search]")
    r = sess().post(f"{a}/step", json={"env_idx": eid, "action": act}, timeout=900)
    return r.status_code == 200


t0 = time.time()
round_times = []
for rnd in range(N_ROUNDS):
    rt = time.time()
    with ThreadPoolExecutor(max_workers=N_CLIENTS) as ex:
        list(ex.map(do_step, [(i, rnd) for i in range(N_CLIENTS)]))
    round_times.append(time.time() - rt)
t_steps = time.time() - t0

print(f"RESULT[{LABEL}] servers={len(addrs)} clients={N_CLIENTS} rounds={N_ROUNDS}")
print(f"  create : {t_create:7.2f}s")
print(f"  reset  : {t_reset:7.2f}s")
print(f"  steps  : {t_steps:7.2f}s   (per-round avg {t_steps/N_ROUNDS:.2f}s)")
print(f"  TOTAL  : {t_create+t_reset+t_steps:7.2f}s  <-- one rank's env cost for a full batch rollout")
print(f"  round times: {[round(x,2) for x in round_times]}")
