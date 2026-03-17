#!/bin/bash
# 只用 GPU 0,1,4,5（跳过被 VLLM 占用的 2,3）
cd /root/a3dizhu

# 加载已有模型继续训练（如果存在）
LOAD_FLAG=""
if [ -f "/root/a3dizhu/experiments/a3dizhu_v3/model.tar" ]; then
  LOAD_FLAG="--load_model"
  echo "Found existing model, will continue training."
fi

nohup python3 train_dmc.py \
  --cuda 0,1,4,5 \
  --training_device 0 \
  --num_actor_devices 4 \
  --num_actors 16 \
  --xpid a3dizhu_v3 \
  --savedir /root/a3dizhu/experiments \
  --save_interval 15 \
  --total_frames 1000000000 \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --greedy_ratio 0.10 \
  --random_ratio 0.05 \
  $LOAD_FLAG \
  > /root/a3dizhu/train.log 2>&1 &

echo "Training started with PID $!"
echo "Tail log: tail -f /root/a3dizhu/train.log"
