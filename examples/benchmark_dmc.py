"""Measure complete DMC runs; keep hardware and model fixed across pool sizes."""
import argparse
import json
import platform
import time
from pathlib import Path

import torch
import rlcard
from rlcard.agents.dmc_agent import DMCTrainer


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--env', default='leduc-holdem')
    parser.add_argument('--envs', type=int, nargs='+', default=[1, 8, 32])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--total_frames', type=int, default=20000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--unroll_length', type=int, default=20)
    parser.add_argument('--num_actors', type=int, default=1)
    parser.add_argument('--backend', choices=['auto', 'python', 'cpp'], default='python')
    parser.add_argument('--cuda', default='')
    parser.add_argument('--training_device', default='0')
    parser.add_argument('--output', default='experiments/benchmark/results.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(configuration=vars(args), python=platform.python_version(),
                  platform=platform.platform(), torch=torch.__version__,
                  metric='end-to-end trained samples/s, including startup and final checkpoint',
                  runs=[])
    for count in args.envs:
        for repeat in range(args.repeats):
            seed = 42 + repeat
            config = {'seed': seed}
            if args.env == 'a3dizhu':
                config['backend'] = args.backend
            env = rlcard.make(args.env, config=config)
            trainer = DMCTrainer(env, cuda=args.cuda, training_device=args.training_device,
                backend=args.backend, num_actors=args.num_actors, envs_per_actor=count,
                total_frames=args.total_frames, batch_size=args.batch_size,
                unroll_length=args.unroll_length, num_buffers=max(32, args.batch_size * 2),
                mlp_layers=[128, 128], seed=seed, savedir=str(output.parent),
                xpid='envs-%d-repeat-%d' % (count, repeat))
            start = time.perf_counter()
            stats = trainer.start()
            elapsed = time.perf_counter() - start
            report['runs'].append(dict(envs_per_actor=count, repeat=repeat, seed=seed,
                                      elapsed_seconds=elapsed, samples_per_second=stats['frames']/elapsed,
                                      actual_backend=trainer.backend, statistics=stats))
            report['devices'] = ([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                                 if args.cuda else ['cpu'])
            output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
