"""
A3 地主 DMC 训练脚本

使用方法：
  # 4×H100 多卡训练（推荐）：
  python train_dmc.py --cuda 0,1,2,3 --num_actor_devices 3 --training_device 3 --num_actors 24

  # 单卡 H100：
  python train_dmc.py --cuda 0 --training_device 0 --num_actors 32

  # 继续训练：
  python train_dmc.py --cuda 0,1,2,3 --num_actor_devices 3 --training_device 3 --load_model

  # CPU 调试（少量帧）：
  python train_dmc.py --total_frames 1000000 --num_actors 2

训练完成后模型保存在 experiments/dmc_result/<xpid>/model.tar
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer


def train(args):
    env = rlcard.make('a3dizhu', config={
        'greedy_ratio': args.greedy_ratio,
        'random_ratio': args.random_ratio,
    })

    total_actors = args.num_actors * args.num_actor_devices
    if args.vectorized:
        total_envs = total_actors * args.envs_per_actor
        mode_str = f'向量化 ({args.envs_per_actor} envs/actor, {total_envs} total)'
    else:
        total_envs = total_actors
        mode_str = f'经典 (1 env/actor, {total_envs} total)'

    print(f'[A3 地主 DMC 训练 — 增强版]')
    print(f'  状态维度: {env.state_shape}')
    print(f'  动作维度: {env.action_shape}')
    print(f'  玩家数量: {env.num_players}')
    print(f'  ────────────────────────────')
    print(f'  Actor 模式: {mode_str}')
    print(f'  权重共享: {"是 (4位置→1网络)" if args.share_weights else "否 (4独立网络)"}')
    print(f'  Actors:   {args.num_actors}/device × {args.num_actor_devices} devices')
    print(f'  Threads:  {args.num_threads}/device/position')
    print(f'  Batch:    {args.batch_size}')
    print(f'  LR:       {args.learning_rate} → {args.min_lr} (cosine)')
    print(f'  Epsilon:  {args.initial_epsilon} → {args.final_epsilon} (linear)')
    print(f'  总帧数:   {args.total_frames:,} ({args.total_frames/1e9:.1f}B)')
    print(f'  贪婪混入: {args.greedy_ratio*100:.0f}%/座位')
    print(f'  随机混入: {args.random_ratio*100:.0f}%/座位')
    print(f'  实验 ID:  {args.xpid}')
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
        unroll_length=args.unroll_length,
        num_buffers=args.num_buffers,
        num_threads=args.num_threads,
        share_weights=args.share_weights,
        initial_epsilon=args.initial_epsilon,
        final_epsilon=args.final_epsilon,
        min_lr=args.min_lr,
        vectorized=args.vectorized,
        envs_per_actor=args.envs_per_actor,
    )

    trainer.start()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('A3 地主 DMC 训练')

    # ─── 硬件 ─────────────────────────────────────
    parser.add_argument('--cuda', type=str, default='',
                        help='CUDA 设备，如 "0,1,2,3"；空=CPU')
    parser.add_argument('--training_device', type=str, default='cpu',
                        help='训练用 GPU 编号，或 "cpu"')
    parser.add_argument('--num_actor_devices', type=int, default=1,
                        help='模拟用 GPU 数量')
    parser.add_argument('--num_actors', type=int, default=24,
                        help='每 GPU 的 Actor 数')
    parser.add_argument('--num_threads', type=int, default=4,
                        help='每设备每位置的 learner 线程数')

    # ─── 训练控制 ──────────────────────────────────
    parser.add_argument('--total_frames', type=int, default=5_000_000_000,
                        help='总帧数')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小 (必须 <= num_buffers)')
    parser.add_argument('--unroll_length', type=int, default=60,
                        help='展开长度 (A3地主每局约30-50步)')
    parser.add_argument('--num_buffers', type=int, default=50,
                        help='共享内存 buffer 数量')
    parser.add_argument('--learning_rate', type=float, default=0.0003,
                        help='初始学习率')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='最终学习率 (cosine decay)')

    # ─── 探索率退火 ────────────────────────────────
    parser.add_argument('--initial_epsilon', type=float, default=0.10,
                        help='初始探索率')
    parser.add_argument('--final_epsilon', type=float, default=0.01,
                        help='最终探索率')

    # ─── 模型 ─────────────────────────────────────
    parser.add_argument('--share_weights', type=int, default=1,
                        help='权重共享 (1=是, 0=否)')
    parser.add_argument('--load_model', action='store_true',
                        help='加载已有模型继续训练')
    parser.add_argument('--xpid', type=str, default='a3dizhu_v6',
                        help='实验 ID')
    parser.add_argument('--savedir', type=str, default='experiments/dmc_result',
                        help='保存目录')
    parser.add_argument('--save_interval', type=int, default=15,
                        help='保存间隔（分钟）')

    # ─── 向量化环境 ─────────────────────────────────
    parser.add_argument('--vectorized', type=int, default=1,
                        help='向量化 Actor (1=是, 0=否; 默认开启)')
    parser.add_argument('--envs_per_actor', type=int, default=200,
                        help='每个 Actor 进程管理的 C++ 环境数')

    # ─── 对手混入 ──────────────────────────────────
    parser.add_argument('--greedy_ratio', type=float, default=0.20,
                        help='贪婪 agent 混入概率/座位 (前期建议 20%%)')
    parser.add_argument('--random_ratio', type=float, default=0.08,
                        help='随机 agent 混入概率/座位')

    args = parser.parse_args()
    args.share_weights = bool(args.share_weights)
    args.vectorized = bool(args.vectorized)

    if args.batch_size > args.num_buffers:
        print(f'[WARNING] batch_size({args.batch_size}) > num_buffers({args.num_buffers}), '
              f'会导致死锁! 自动调整 num_buffers={args.batch_size + 10}')
        args.num_buffers = args.batch_size + 10

    if args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda

    train(args)
