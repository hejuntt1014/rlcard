"""Profile native rollout Python overhead without neural-network inference."""
import argparse
import contextlib
import cProfile
import io
import json
from pathlib import Path
import pstats
import time
from types import SimpleNamespace

import rlcard
from rlcard.envs.a3dizhu.dmc import NativePool


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--envs', type=int, default=512)
    parser.add_argument('--rounds', type=int, default=150)
    parser.add_argument('--output', default='experiments/native-pool-profile')
    args = parser.parse_args()
    agent = object()
    model = SimpleNamespace(get_agent=lambda _player: agent)
    pool = NativePool(rlcard.make('a3dizhu-v12', config={'reward_mode': 'game'}), args.envs, 42, 10000)
    for _ in range(20):
        pool.round(model, 1., 4096, contextlib.nullcontext())
    profiler = cProfile.Profile()
    steps = episodes = 0
    start = time.perf_counter()
    profiler.enable()
    for _ in range(args.rounds):
        finished, count = pool.round(model, 1., 4096, contextlib.nullcontext())
        steps += count
        episodes += len(finished)
    profiler.disable()
    elapsed = time.perf_counter() - start
    report = dict(envs=args.envs, rounds=args.rounds, steps=steps, episodes=episodes,
                  seconds=elapsed, profiled_samples_per_second=steps / elapsed,
                  batch_native=pool.batch_native, indexed_native=pool.indexed_native,
                  scope='NativePool.round only; epsilon=1 skips neural inference; cProfile instrumentation included')
    text = io.StringIO()
    pstats.Stats(profiler, stream=text).strip_dirs().sort_stats('cumulative').print_stats(35)
    pstats.Stats(profiler, stream=text).strip_dirs().sort_stats('tottime').print_stats(20)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    out.with_suffix('.txt').write_text(text.getvalue(), encoding='utf-8')
    profiler.dump_stats(str(out.with_suffix('.prof')))
    print(json.dumps(report, indent=2))
    print(text.getvalue())


if __name__ == '__main__':
    main()
