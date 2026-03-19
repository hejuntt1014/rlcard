#!/bin/bash
# A3 地主 DMC v6 训练 — 向量化 Actor 版
# 每进程管理 200 个 C++ 环境 + 批量 GPU 推理
# 进程数从 120 降至 24，GPU 利用率大幅提升

ulimit -n 200000
cd /root/a3dizhu_v6

XPID="a3dizhu_v6"
SAVE_DIR="/root/a3dizhu_v6/experiments"

LOAD_FLAG=""
if [ -f "${SAVE_DIR}/${XPID}/model.tar" ]; then
  LOAD_FLAG="--load_model"
  echo "Found existing v6 model, will continue training."
else
  echo "Starting fresh v6 training (850D state, declare phase support)."
fi

nohup python3 train_dmc.py \
  --cuda 0,1,2,3,4,5 \
  --training_device 5 \
  --num_actor_devices 5 \
  --num_actors 3 \
  --num_threads 1 \
  --xpid ${XPID} \
  --savedir ${SAVE_DIR} \
  --save_interval 5 \
  --total_frames 5000000000 \
  --batch_size 256 \
  --num_buffers 500 \
  --learning_rate 0.0003 \
  --min_lr 0.000001 \
  --unroll_length 60 \
  --initial_epsilon 0.10 \
  --final_epsilon 0.01 \
  --share_weights 1 \
  --greedy_ratio 0.05 \
  --random_ratio 0.01 \
  --vectorized 1 \
  --envs_per_actor 200 \
  $LOAD_FLAG \
  > /root/a3dizhu_v6/train_v6.log 2>&1 &

echo "Training v6 started with PID $!"
echo "  Actors: 3/device × 5 devices = 15 processes × 200 envs = 3000 concurrent games"
echo "Tail log: tail -f /root/a3dizhu_v6/train_v6.log"
