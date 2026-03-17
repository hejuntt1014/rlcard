#!/bin/bash
# 清理所有训练相关的 python3 进程（不清理 VLLM 和 text-embeddings）
for pid in $(nvidia-smi --query-compute-apps=pid,name --format=csv,noheader | grep python3 | awk -F, '{print $1}'); do
  echo "Killing PID $pid"
  kill -9 $pid 2>/dev/null
done
sleep 3
echo "=== Remaining GPU processes ==="
nvidia-smi --query-compute-apps=pid,name --format=csv,noheader
