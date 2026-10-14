"""Smoke-check an ALFWorld environment server: can it create and reset an env.

    python scripts/verify_alfworld_env.py [game_id]

Each step is reported separately so a failure points at one thing -- the service not
being up, /create not returning an id, or /reset coming back without the fields the
client expects.
"""
import json
import sys
import traceback

import requests


def verify_alfworld_reset(env_addr="http://127.0.0.1:36001", game_id=0):
    print(f"--- verifying AlfWorld env (addr: {env_addr}, game id: {game_id}) ---")

    try:
        # 1. create an environment instance (/create)
        print("step 1: creating an environment instance...")
        try:
            create_res = requests.post(f"{env_addr}/create", timeout=10)
        except requests.exceptions.ConnectionError:
            print(f"CRITICAL: cannot reach the env service at {env_addr}. Is it running?")
            return

        if create_res.status_code != 200:
            print(f"FAILED: /create returned status {create_res.status_code}")
            print(f"body: {create_res.text}")
            return

        env_info = create_res.json()
        if "id" not in env_info:
            print(f"FAILED: /create response has no 'id'. body: {env_info}")
            return

        env_id = env_info["id"]
        print(f"SUCCESS: environment created, id {env_id}")

        # 2. reset it (/reset)
        print(f"\nstep 2: resetting the environment (env_id={env_id}, game={game_id})...")
        reset_payload = {
            "id": env_id,
            "game": game_id,
            "world_type": "Text"
        }
        reset_res = requests.post(f"{env_addr}/reset", json=reset_payload, timeout=60)

        if reset_res.status_code != 200:
            print(f"FAILED: /reset returned status {reset_res.status_code}")
            print(f"body: {reset_res.text}")
            return

        response_data = reset_res.json()

        # 3. check the fields the client depends on
        print("\nstep 3: checking the response structure...")
        if "error" in response_data:
            print("FAILED: the server reported an error:")
            print(f"  Error: {response_data['error']}")
            return

        required_keys = ["observation", "available_actions"]
        missing_keys = [k for k in required_keys if k not in response_data]

        if missing_keys:
            print(f"FAILED: response is missing required keys: {missing_keys}")
            print(f"full body: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
        else:
            print("SUCCESS: reset works and returned every required field.")
            print("-" * 20)
            print(f"Observation (first 100 chars):\n{response_data['observation'][:100]}...")
            print("-" * 20)
            print(f"Available actions: {len(response_data['available_actions'])}")

    except Exception:
        print("CRITICAL ERROR: the verification script raised an uncaught exception")
        traceback.print_exc()


if __name__ == "__main__":
    target_game_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    verify_alfworld_reset(env_addr="http://127.0.0.1:36001", game_id=target_game_id)
