#!/bin/bash
# A3 地主 DMC v6 训练
# v6 改进：报牌阶段支持（AI 可学习是否报牌），STATE_DIM 849→850

cd /root/a3dizhu_v6

XPID="a3dizhu_v6"
SAVE_DIR="/root/a3dizhu_v6/experiments"

# v6 是全新训练（状态维度从850D，与v5不兼容，不加载旧模型）
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
  --num_actors 20 \
  --xpid ${XPID} \
  --savedir ${SAVE_DIR} \
  --save_interval 15 \
  --total_frames 5000000000 \
  --batch_size 512 \
  --num_buffers 600 \
  --learning_rate 0.0003 \
  --min_lr 0.000001 \
  --unroll_length 60 \
  --initial_epsilon 0.10 \
  --final_epsilon 0.01 \
  --share_weights 1 \
  --greedy_ratio 0.05 \
  --random_ratio 0.01 \
  $LOAD_FLAG \
  > /root/a3dizhu_v6/train_v6.log 2>&1 &

echo "Training v6 started with PID $!"
echo "Tail log: tail -f /root/a3dizhu_v6/train_v6.log"
