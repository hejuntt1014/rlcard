"""
A3 地主 DMC 训练脚本

使用方法：
  # 基本训练（CPU）：
  python train_dmc.py

  # GPU 训练（推荐，有显卡时）：
  python train_dmc.py --cuda 0 --training_device 0

  # 继续训练：
  python train_dmc.py --load_model --xpid a3dizhu_v1

  # 多进程加速（CPU 核多时）：
  python train_dmc.py --num_actors 8

训练完成后模型保存在 experiments/dmc_result/<xpid>/model.tar
"""

import os
import sys
import argparse

# 确保可以 import rl/rlcard
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer


def train(args):
    # 创建 A3 地主环境（支持混合对手训练）
    env = rlcard.make('a3dizhu', config={
        'greedy_ratio': args.greedy_ratio,
        'random_ratio': args.random_ratio,
    })

    print(f'[A3 地主 DMC 训练]')
    print(f'  状态维度: {env.state_shape}')
    print(f'  动作维度: {env.action_shape}')
    print(f'  玩家数量: {env.num_players}')
    print(f'  贪婪混入: {args.greedy_ratio*100:.0f}%/座位')
    print(f'  随机混入: {args.random_ratio*100:.0f}%/座位')
    print(f'  实验 ID:  {args.xpid}')
    print(f'  保存目录: {args.savedir}')
    print()

    trainer = DMCTrainer(
        env,
        cuda=args.cuda,
        load_model=args.load_model,
        xpid=args.xpid,
        savedir=args.savedir,
        save_interval=args.save_interval,
        num_actor_devices=args.num_actor_devices,
        num_actors=args.num_actors,
        training_device=args.training_device,
        total_frames=args.total_frames,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    trainer.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A3 地主 DMC 训练')

    parser.add_argument(
        '--cuda', type=str, default='',
        help='可用的 CUDA 设备 ID，如 "0" 或 "0,1"；空字符串=CPU'
    )
    parser.add_argument(
        '--training_device', type=str, default='cpu',
        help='训练用的设备，"0"=GPU0，"cpu"=CPU'
    )
    parser.add_argument(
        '--num_actor_devices', type=int, default=1,
        help='模拟用的设备数量'
    )
    parser.add_argument(
        '--num_actors', type=int, default=4,
        help='每个设备的 Actor 数量（并行环境数），建议 = CPU核数/2'
    )
    parser.add_argument(
        '--load_model', action='store_true',
        help='加载已有模型继续训练'
    )
    parser.add_argument(
        '--xpid', type=str, default='a3dizhu_v1',
        help='实验 ID（用于区分不同训练运行）'
    )
    parser.add_argument(
        '--savedir', type=str, default='experiments/dmc_result',
        help='模型保存目录'
    )
    parser.add_argument(
        '--save_interval', type=int, default=30,
        help='保存间隔（分钟）'
    )
    parser.add_argument(
        '--total_frames', type=int, default=100_000_000,
        help='总训练帧数（越大训练越充分，默认1亿）'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='批次大小'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.0001,
        help='学习率'
    )
    parser.add_argument(
        '--greedy_ratio', type=float, default=0.10,
        help='每座位被贪婪 agent 替换的概率（默认 10%%）'
    )
    parser.add_argument(
        '--random_ratio', type=float, default=0.05,
        help='每座位被随机 agent 替换的概率（默认 5%%）'
    )

    args = parser.parse_args()

    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    train(args)
