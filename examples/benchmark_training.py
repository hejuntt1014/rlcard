"""Reproducible full-pipeline DMC timing with configurable actor/learner sizes."""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import torch
import rlcard
from rlcard.agents.dmc_agent import DMCTrainer


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--label', default='timing')
    parser.add_argument('--frames', type=int, default=2_000_000)
    parser.add_argument('--envs', type=int, default=128)
    parser.add_argument('--actors', type=int, default=1)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--unroll', type=int, default=20)
    parser.add_argument('--precision', choices=['fp32', 'bf16', 'fp16'], default='bf16')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sync', type=int, default=50)
    parser.add_argument('--trainer-options', default='{}', help='Additional DMCTrainer keyword arguments as JSON')
    args = parser.parse_args()
    torch.set_num_threads(1)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    kwargs = dict(cuda='0', training_device='0', num_actors=args.actors,
                  envs_per_actor=args.envs, total_frames=args.frames, batch_size=args.batch,
                  unroll_length=args.unroll, num_buffers=max(args.batch * 2, 64),
                  share_weights=True, precision=args.precision, seed=args.seed,
                  initial_epsilon=.08, final_epsilon=.01, stats_interval=50,
                  weight_sync_interval=args.sync, save_interval=60,
                  savedir=str(out), xpid=args.label)
    kwargs.update(json.loads(args.trainer_options))
    trainer = DMCTrainer(rlcard.make('a3dizhu-v12', config={'reward_mode': 'game'}), **kwargs)
    # Record steady-state progress separately from whole Trainer.start time.
    records = []
    original = trainer._record_stats
    def record(*a, **k):
        original(*a, **k)
        records.append(dict(time=time.perf_counter(), frames=trainer.frames))
    trainer._record_stats = record
    started = time.perf_counter()
    stats = trainer.start()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    useful = [r for r in records if r['frames'] >= args.frames * .2]
    steady = ((useful[-1]['frames'] - useful[0]['frames']) /
              (useful[-1]['time'] - useful[0]['time'])) if len(useful) > 1 else None
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        revision = 'working-copy'
    result = dict(label=args.label, python=platform.python_version(), torch=torch.__version__,
                  cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  commit=revision,
                  configuration=kwargs, seconds=elapsed, samples_per_second=stats['frames']/elapsed,
                  steady_samples_per_second=steady, statistics=stats, progress=records,
                  learner_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  timing='Trainer.start wall time including startup and final save; excludes interpreter imports')
    (out / (args.label + '.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'progress'}), flush=True)


if __name__ == '__main__':
    main()
