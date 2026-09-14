import requests
import json
import traceback
import sys

def verify_alfworld_reset(env_addr="http://127.0.0.1:36001", game_id=0):
    print(f"--- 开始验证 AlfWorld 环境 (地址: {env_addr}, 游戏ID: {game_id}) ---")
    
    try:
        # 1. 创建环境实例 (模拟 /create)
        print("步骤 1: 尝试创建环境实例...")
        try:
            create_res = requests.post(f"{env_addr}/create", timeout=10)
        except requests.exceptions.ConnectionError:
            print(f"CRITICAL: 无法连接到环境服务 {env_addr}。请确保服务已启动。")
            return
            
        if create_res.status_code != 200:
            print(f"FAILED: /create 请求失败，状态码: {create_res.status_code}")
            print(f"响应内容: {create_res.text}")
            return
        
        env_info = create_res.json()
        if "id" not in env_info:
            print(f"FAILED: /create 返回结果中缺少 'id'. 响应: {env_info}")
            return
        
        env_id = env_info["id"]
        print(f"SUCCESS: 环境创建成功，分配的 ID 为: {env_id}")

        # 2. 调用重置 (模拟 /reset)
        print(f"\n步骤 2: 尝试重置环境 (env_id={env_id}, game={game_id})...")
        reset_payload = {
            "id": env_id,
            "game": game_id,
            "world_type": "Text"
        }
        reset_res = requests.post(f"{env_addr}/reset", json=reset_payload, timeout=60)
        
        if reset_res.status_code != 200:
            print(f"FAILED: /reset 请求失败，状态码: {reset_res.status_code}")
            print(f"响应内容: {reset_res.text}")
            return

        response_data = reset_res.json()
        
        # 3. 验证关键字段
        print("\n步骤 3: 验证返回值结构...")
        if "error" in response_data:
            print(f"FAILED: 服务端返回了明确的错误信息:")
            print(f"  Error: {response_data['error']}")
            return

        required_keys = ["observation", "available_actions"]
        missing_keys = [k for k in required_keys if k not in response_data]
        
        if missing_keys:
            print(f"FAILED: 响应中缺少关键 key: {missing_keys}")
            print(f"完整响应内容: {json.dumps(response_data, indent=2, ensure_ascii=False)}")
        else:
            print("SUCCESS: reset 功能正常，返回了所有必需字段。")
            print("-" * 20)
            print(f"Observation 预览 (前100字):\n{response_data['observation'][:100]}...")
            print("-" * 20)
            print(f"Available Actions 数量: {len(response_data['available_actions'])}")

    except Exception as e:
        print(f"CRITICAL ERROR: 验证脚本运行过程中发生未捕获异常")
        traceback.print_exc()

if __name__ == "__main__":
    # 可以通过命令行参数指定 game_id，默认使用 1
    target_game_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    verify_alfworld_reset(env_addr="http://127.0.0.1:36001", game_id=target_game_id)
