"""Compare Python and NumPy segmented argmax on real native action groups."""
import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np

from examples.benchmark_context import decisions
from rlcard.agents.dmc_agent.collector import segmented_argmax


def loop_argmax(values, offsets):
    return np.asarray([values[start:end].argmax() for start, end in zip(offsets[:-1], offsets[1:])])


def measure(function, repeats):
    for _ in range(10):
        function()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(20):
            function()
        samples.append((time.perf_counter() - start) * 1e6 / 20)
    return dict(median_us=statistics.median(samples), samples_us=samples)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--states', type=int, default=512)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--output', default='experiments/segmented-argmax.json')
    args = parser.parse_args()
    _, actions, offsets, _ = decisions(args.states, 42)
    offsets = np.asarray(offsets, dtype=np.int64)
    rng = np.random.default_rng(42)
    report = dict(states=args.states, candidates=len(actions), cases={})
    for case in ('finite', 'ties', 'nan_inf'):
        values = rng.standard_normal(len(actions)).astype(np.float32)
        if case == 'ties':
            values = np.round(values)
        elif case == 'nan_inf':
            values[::7], values[1::11], values[2::13] = np.nan, np.inf, -np.inf
        np.testing.assert_array_equal(loop_argmax(values, offsets), segmented_argmax(values, offsets))
        before = measure(lambda: loop_argmax(values, offsets), args.repeats)
        after = measure(lambda: segmented_argmax(values, offsets), args.repeats)
        report['cases'][case] = dict(loop=before, segmented=after,
                                    speed_ratio=before['median_us'] / after['median_us'])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
