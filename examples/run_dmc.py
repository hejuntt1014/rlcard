''' An example of training a Deep Monte-Carlo (DMC) Agent on the environments in RLCard
'''
import os
import argparse
import json
import logging

import torch

import rlcard
from rlcard.agents.dmc_agent import DMCTrainer

def train(args):

    # Make the environment
    config = {'seed': args.seed}
    if args.env in ('a3dizhu', 'a3dizhu-v12'):
        config['backend'] = args.backend
    if args.env == 'a3dizhu-v12':
        config['reward_mode'] = args.reward_mode
        config['rules'] = json.loads(args.rules) if args.rules else None
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
        history_encoder=args.history_encoder,
        history_steps=args.history_steps,
        precision=args.precision,
        initial_epsilon=args.initial_epsilon,
        final_epsilon=args.final_epsilon,
        learning_rate=args.learning_rate,
        pin_memory=args.pin_memory,
        dense_learner=args.dense_learner,
        compile_learner=args.compile_learner,
        compile_mode=args.compile_mode,
        actor_cuda_graphs=args.actor_cuda_graphs,
        actor_half_weights=args.actor_half_weights,
        learner_poll_interval=args.learner_poll_interval,
        actor_poll_interval=args.actor_poll_interval,
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
            'a3dizhu',
            'a3dizhu-v12'
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
    parser.add_argument('--architecture', choices=['mlp', 'resnet', 'context'])
    parser.add_argument('--auxiliary', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--hidden_sizes', type=int, nargs='+')
    parser.add_argument('--history_encoder', choices=['mlp', 'transformer'], default='mlp')
    parser.add_argument('--history_steps', type=int, default=24)
    parser.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='fp32')
    parser.add_argument('--initial_epsilon', type=float)
    parser.add_argument('--final_epsilon', type=float)
    parser.add_argument('--learning_rate', type=float, default=.0001)
    parser.add_argument('--reward_mode', choices=['game', 'shaped'], default='game')
    parser.add_argument('--rules', help='JSON object with all three A3 rule settings; omitted means random rules')
    parser.add_argument('--weight_sync_interval', type=int, default=50)
    parser.add_argument('--cpu_threads', type=int, default=1)
    parser.add_argument('--actor_on_cpu', action='store_true')
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--dense_learner', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--compile_learner', action='store_true')
    parser.add_argument('--compile_mode', choices=['default', 'reduce-overhead', 'max-autotune'], default='default')
    parser.add_argument('--actor_cuda_graphs', action='store_true')
    parser.add_argument('--actor_half_weights', action='store_true')
    parser.add_argument('--learner_poll_interval', type=float, default=.002)
    parser.add_argument('--actor_poll_interval', type=float, default=.005)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    torch.set_num_threads(args.cpu_threads)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda
    train(args)
