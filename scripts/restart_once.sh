#!/usr/bin/env bash

echo "计时器已启动：3 小时后将重启实验..."
sleep 3h

echo "[$(date)] 正在重启实验..."
# launch_alfworld_grpo_tmux.sh 内部会自动 kill 掉正在运行的旧 session
bash launch_alfworld_grpo_tmux.sh

echo "重启指令已发送。"
