"""Compare expanded and compact candidate scoring on identical A3 decisions."""
import argparse
import contextlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from rlcard.agents.dmc_agent.collector import score_actions, score_action_groups
from rlcard.agents.dmc_agent.model import DMCAgent


def decisions(count, seed):
    from a3dizhu_v12_cpp import VectorizedEngine
    vec = VectorizedEngine(count)
    vec.seed(seed)
    rng = np.random.default_rng(seed)
    for i in range(count):
        vec.reset(i)
        for _ in range(4):
            vec.step(i, 'pass')
        for _ in range(int(rng.integers(4, 30))):
            if vec.is_over(i):
                vec.reset(i)
                for _ in range(4):
                    vec.step(i, 'pass')
            _, _, _, keys = vec.prepare_batch([i])
            vec.step(i, keys[0][int(rng.integers(len(keys[0])))])
        if vec.is_over(i):
            vec.reset(i)
            for _ in range(4):
                vec.step(i, 'pass')
    return vec.prepare_batch(list(range(count)))


def measure(function, device, repeats):
    for _ in range(3):
        function()
    samples = []
    for _ in range(repeats):
        if device != 'cpu':
            torch.cuda.synchronize(int(device))
        start = time.perf_counter()
        function()
        if device != 'cpu':
            torch.cuda.synchronize(int(device))
        samples.append((time.perf_counter() - start) * 1000)
    return dict(median_ms=statistics.median(samples), samples_ms=samples)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--envs', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--history_steps', type=int, default=24)
    parser.add_argument('--output', default='experiments/context-benchmark.json')
    args = parser.parse_args()
    if args.envs < 1 or args.repeats < 1:
        parser.error('envs and repeats must be positive')
    torch.set_num_threads(1)
    obs, actions, offsets, _ = decisions(args.envs, args.seed)
    counts = np.diff(offsets)
    report = dict(python=platform.python_version(), torch=torch.__version__, device=args.device,
                  gpu=torch.cuda.get_device_name(int(args.device)) if args.device != 'cpu' else None,
                  seed=args.seed, states=len(obs), candidates=len(actions),
                  unique_state_bytes=obs.nbytes, expanded_state_bytes=int(counts.sum()) * obs.shape[1],
                  precision='fp32', history_steps=args.history_steps,
                  method='identical real states, warmup, synchronized wall time, no training',
                  models={})
    for history in ('mlp', 'transformer'):
        torch.manual_seed(args.seed)
        agent = DMCAgent([2668], [111], [768] * 5, device=args.device,
                         architecture='context', history_encoder=history, aux_classes=(3,3,3,4,4),
                         history_steps=args.history_steps)
        agent.eval()
        def expanded():
            return score_actions(agent, np.repeat(obs, counts, axis=0), actions, 4096, contextlib.nullcontext())
        def compact():
            return score_action_groups(agent, obs, actions, offsets, 4096, contextlib.nullcontext())
        expected, actual = expanded(), compact()
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)
        before, after = measure(expanded, args.device, args.repeats), measure(compact, args.device, args.repeats)
        report['models'][history] = dict(parameters=sum(p.numel() for p in agent.parameters()),
            max_abs_difference=float(np.abs(actual - expected).max()), expanded=before, compact=after,
            speed_ratio=before['median_ms'] / after['median_ms'])
        del agent
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
