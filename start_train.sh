#!/bin/bash
# A3 地主 DMC v4 训练 — 中后期比例调整版（降低弱规则对手混入）
# GPU 0,1,4,5（跳过被 VLLM 占用的 2,3）
cd /root/a3dizhu

XPID="a3dizhu_v4"
SAVE_DIR="/root/a3dizhu/experiments"

# v4 是全新训练，不加载旧模型（旧模型规则错误）
LOAD_FLAG=""
if [ -f "${SAVE_DIR}/${XPID}/model.tar" ]; then
  LOAD_FLAG="--load_model"
  echo "Found existing v4 model, will continue training."
else
  echo "Starting fresh v4 training (flush rule fix + ResNet + weight sharing)."
fi

nohup python3 train_dmc.py \
  --cuda 0,1,4,5 \
  --training_device 0 \
  --num_actor_devices 4 \
  --num_actors 16 \
  --xpid ${XPID} \
  --savedir ${SAVE_DIR} \
  --save_interval 15 \
  --total_frames 5000000000 \
  --batch_size 128 \
  --num_buffers 138 \
  --learning_rate 0.0003 \
  --min_lr 0.000001 \
  --unroll_length 60 \
  --initial_epsilon 0.10 \
  --final_epsilon 0.01 \
  --share_weights 1 \
  --greedy_ratio 0.05 \
  --random_ratio 0.01 \
  $LOAD_FLAG \
  > /root/a3dizhu/train_v4.log 2>&1 &

echo "Training v4 started with PID $!"
echo "Tail log: tail -f /root/a3dizhu/train_v4.log"
