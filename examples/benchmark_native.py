"""CPU-only A3 native rollout throughput with reproducible output fingerprints."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
from a3dizhu_v12_cpp import VectorizedEngine


def action_keys_from_features(actions, offsets):
    """Reconstruct canonical keys for fingerprints without touching native caches."""
    names = [suit+'_'+rank for suit in ('diamond','club','heart','spade')
             for rank in ('4','5','6','7','8','9','10','J','Q','K','A','2','3')]
    flat = []
    for action in actions:
        cards = np.flatnonzero(action[:52])
        flat.append('declare' if len(cards) == 52 else
                    '|'.join(sorted(names[c] for c in cards)) if len(cards) else 'pass')
    return [flat[start:end] for start,end in zip(offsets[:-1],offsets[1:])]


def prepare_benchmark(count, seed, repeats, calls=1000):
    """Time candidate preparation on a fixed, identical collection of real states."""
    vec = VectorizedEngine(count)
    vec.seed(seed)
    rng = np.random.default_rng(seed)
    for i in range(count):
        vec.reset(i)
        for _ in range(4):
            vec.step(i, 'pass')
        for _ in range(int(rng.integers(4, 30))):
            if vec.is_over(i):
                break
            _, _, offsets, _, _ = vec.prepare_indexed_batch([i])
            vec.step_choices([i], [int(rng.integers(offsets[-1]))])
        if vec.is_over(i):
            vec.reset(i)
            for _ in range(4):
                vec.step(i, 'pass')
    indices = list(range(count))
    methods = [vec.prepare_batch_with_metadata, vec.prepare_indexed_batch]
    samples = {method.__name__: [] for method in methods}
    for method in methods:
        for _ in range(100):
            method(indices)
    for repeat in range(repeats):
        for method in (methods if repeat % 2 == 0 else methods[::-1]):
            start = time.perf_counter()
            for _ in range(calls):
                result = method(indices)
            samples[method.__name__].append((time.perf_counter()-start)*1000/calls)
    return dict(states=count, candidates=len(result[1]), calls_per_sample=calls,
                methods={name:dict(median_ms=statistics.median(values),samples_ms=values)
                         for name,values in samples.items()})


def run(count, rounds, seed, batch=False, fingerprint=False, greedy=0., shaping=True, indexed=False):
    batch = batch or indexed
    vec = VectorizedEngine(count)
    vec.seed(seed)
    vec.set_greedy_ratio(greedy)
    if hasattr(vec, 'set_reward_shaping'):
        vec.set_reward_shaping(shaping)
    for i in range(count):
        vec.reset(i)
    rng = np.random.default_rng(seed)
    digest = hashlib.sha256()
    steps = candidates = games = 0
    start = time.perf_counter()
    for _ in range(rounds):
        pending = []
        if batch:
            done, pending, players, records = vec.advance_batch()
        else:
            done, records = [], []
            for i in range(count):
                over, obs, pids, acts, aux = vec.advance_to_decision_with_data(i)
                if pids:
                    records.append((i, obs, pids, acts, aux))
                (done if over else pending).append(i)
        for i, obs, pids, acts, aux in records:
            steps += len(pids)
            if fingerprint:
                digest.update(obs.tobytes())
                digest.update(acts.tobytes())
                digest.update(aux.tobytes())
        for i in done:
            if fingerprint:
                digest.update(np.asarray(vec.get_payoffs(i)).tobytes())
            games += 1
            vec.reset(i)
        if not pending:
            continue
        if indexed:
            obs, acts, offsets, players, auxiliary = vec.prepare_indexed_batch(pending)
            choices = [int(rng.integers(offsets[row+1]-offsets[row])) for row in range(len(pending))]
            if fingerprint:
                keys = action_keys_from_features(acts, offsets)
        elif batch:
            obs, acts, offsets, keys, players, auxiliary = vec.prepare_batch_with_metadata(pending)
        else:
            obs, acts, offsets, keys = vec.prepare_batch(pending)
            players = [vec.get_player_id(i) for i in pending]
            auxiliary = np.stack([vec.get_aux_targets_for_player(i, p) for i, p in zip(pending, players)])
        if not indexed:
            selected = [group[int(rng.integers(len(group)))] for group in keys]
        if fingerprint:
            digest.update(obs.tobytes())
            digest.update(acts.tobytes())
            digest.update(auxiliary.tobytes())
            digest.update(json.dumps(keys, separators=(',', ':')).encode())
        if indexed:
            vec.step_choices(pending, choices)
        elif batch:
            vec.step_batch(pending, selected)
        else:
            for i, key in zip(pending, selected):
                vec.step(i, key)
        steps += len(pending)
        candidates += len(acts)
    elapsed = time.perf_counter() - start
    return dict(seconds=elapsed, decisions=steps, decisions_per_second=steps/elapsed,
                candidates=candidates, games=games, fingerprint=digest.hexdigest() if fingerprint else None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--envs', type=int, default=128)
    parser.add_argument('--rounds', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--indexed', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--fingerprint', action='store_true')
    parser.add_argument('--greedy', type=float, default=0.)
    parser.add_argument('--no-shaping', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    report = dict(python=platform.python_version(), platform=platform.platform(), config=vars(args))
    if args.prepare_only:
        report['preparation'] = prepare_benchmark(args.envs, args.seed, args.repeats)
    else:
        report['runs'] = [run(args.envs, args.rounds, args.seed, args.batch, args.fingerprint,
                            args.greedy, not args.no_shaping, args.indexed) for _ in range(args.repeats)]
    result = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(result, encoding='utf-8')
    print(result)


if __name__ == '__main__':
    main()
