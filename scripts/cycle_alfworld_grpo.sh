#!/usr/bin/env bash

# 默认重启间隔：3 小时 (3 * 3600 秒)
INTERVAL_SECONDS=$((3 * 3600))

echo "=================================================="
echo "ALFWorld 实验循环重启脚本已启动"
echo "重启周期: 3 小时"
echo "使用设备: ${CUDA_VISIBLE_DEVICES:-未指定}"
echo "=================================================="

while true; do
    echo "--------------------------------------------------"
    echo "[$(date)] 正在启动/重启实验..."
    
    # 调用启动脚本。它会自动 kill 掉旧的 tmux session 并开启新的。
    bash launch_alfworld_grpo_tmux.sh
    
    echo "[$(date)] 实验已启动。下一次重启将在 3 小时后。"
    echo "你可以通过 'tmux a -t alfworld_grpo_train' 查看训练日志。"
    
    # 等待 3 小时
    sleep "${INTERVAL_SECONDS}"
done
