''' An example of training a Deep Monte-Carlo (DMC) Agent on the environments in RLCard
'''
import os
import argparse

import torch

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer

def train(args):

    # Make the environment
    config = {'seed': args.seed}
    if args.env == 'a3dizhu':
        config['backend'] = args.backend
    env = rlcard.make(args.env, config=config)

    # Initialize the DMC trainer
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
        unroll_length=args.unroll_length,
        num_buffers=args.num_buffers,
        envs_per_actor=args.envs_per_actor,
        backend=args.backend,
        seed=args.seed,
        share_weights=args.share_weights,
        architecture=args.architecture,
        auxiliary=args.auxiliary,
        mlp_layers=args.hidden_sizes,
        weight_sync_interval=args.weight_sync_interval,
        actor_on_cpu=args.actor_on_cpu,
    )

    # Train DMC Agents
    trainer.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DMC example in RLCard")
    parser.add_argument(
        '--env',
        type=str,
        default='leduc-holdem',
        choices=[
            'blackjack',
            'leduc-holdem',
            'limit-holdem',
            'doudizhu',
            'mahjong',
            'no-limit-holdem',
            'uno',
            'gin-rummy',
            'a3dizhu'
        ],
    )
    parser.add_argument(
        '--cuda',
        type=str,
        default='',
    )
    parser.add_argument(
        '--load_model',
        action='store_true',
        help='Load an existing model',
    )
    parser.add_argument(
        '--xpid',
        default='leduc_holdem',
        help='Experiment id (default: leduc_holdem)',
    )
    parser.add_argument(
        '--savedir',
        default='experiments/dmc_result',
        help='Root dir where experiment data will be saved'
    )
    parser.add_argument(
        '--save_interval',
        default=30,
        type=int,
        help='Time interval (in minutes) at which to save the model',
    )
    parser.add_argument(
        '--num_actor_devices',
        default=1,
        type=int,
        help='The number of devices used for simulation',
    )
    parser.add_argument(
        '--num_actors',
        default=5,
        type=int,
        help='The number of actors for each simulation device',
    )
    parser.add_argument(
        '--training_device',
        default="0",
        type=str,
        help='The index of the GPU used for training models',
    )

    parser.add_argument('--total_frames', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--unroll_length', type=int, default=60)
    parser.add_argument('--num_buffers', type=int, default=64)
    parser.add_argument('--envs_per_actor', type=int, default=8)
    parser.add_argument('--backend', choices=['auto', 'python', 'cpp'], default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--share_weights', action='store_true')
    parser.add_argument('--architecture', choices=['mlp', 'resnet'])
    parser.add_argument('--auxiliary', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--hidden_sizes', type=int, nargs='+', default=[512] * 5)
    parser.add_argument('--weight_sync_interval', type=int, default=50)
    parser.add_argument('--cpu_threads', type=int, default=1)
    parser.add_argument('--actor_on_cpu', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)
