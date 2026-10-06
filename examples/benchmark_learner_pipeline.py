"""Measure the real DMC learner on fixed native decisions, excluding actors.

Includes CPU batch gathering, H2D, forward/backward, clipping and RMSprop. Targets
are deterministic synthetic regression values: this measures speed, not policy
quality. Native decisions, selected legal actions and public auxiliary labels
are identical across cases. Compilation startup is reported separately.

Run from the checkout with ``python -m examples.benchmark_learner_pipeline``.
"""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import platform
import queue
import statistics
import time

import numpy as np
import torch

from rlcard.agents.dmc_agent.model import DMCAgent
from rlcard.agents.dmc_agent.trainer import learn, _make_training_objective
from rlcard.agents.dmc_agent.utils import BatchReader, create_buffers


CASES = {
    'fp32-sparse': ('fp32', False, False, None),
    'fp32-dense': ('fp32', True, False, None),
    'fp32-pinned': ('fp32', True, True, None),
    'bf16-sparse': ('bf16', False, False, None),
    'bf16-dense': ('bf16', True, False, None),
    'bf16-pinned': ('bf16', True, True, None),
    'bf16-compile': ('bf16', True, True, 'default'),
    'bf16-graphs': ('bf16', True, True, 'reduce-overhead'),
    'bf16-compile-loss': ('bf16', True, True, 'default-loss'),
    'bf16-graphs-loss': ('bf16', True, True, 'reduce-overhead-loss'),
}


def fixture(batch_size, unroll_length, seed):
    from a3dizhu_v12_cpp import VectorizedEngine
    count = batch_size * unroll_length
    vec = VectorizedEngine(count)
    vec.seed(seed)
    rng = np.random.default_rng(seed)
    for i in range(count):
        vec.reset(i)
        for _ in range(int(rng.integers(0, 48))):
            vec.step_random(i)
            if vec.is_over(i):
                vec.reset(i)
    observations, actions, offsets, _ = vec.prepare_batch(list(range(count)))
    selected = [rng.integers(offsets[i], offsets[i + 1]) for i in range(count)]
    labels = np.stack([vec.get_aux_targets_for_player(i, vec.get_player_id(i))
                       for i in range(count)])
    data = {
        'state': torch.from_numpy(observations.copy()),
        'action': torch.from_numpy(actions[selected].copy()),
        'aux_target': torch.from_numpy(labels),
        'target': torch.from_numpy(rng.normal(size=count).astype(np.float32)),
        'done': torch.zeros(count, dtype=torch.bool),
        'episode_return': torch.zeros(count),
    }
    return {key: value.reshape(batch_size, unroll_length, *value.shape[1:])
            for key, value in data.items()}


def run_case(name, batch, args):
    precision, dense, pinned, compile_mode = CASES[name]
    if args.device == 'cpu' and precision != 'fp32':
        raise ValueError('BF16 benchmark cases require CUDA')
    pinned = pinned and args.device != 'cpu'
    torch.manual_seed(args.seed)
    agent = DMCAgent([2668], [111], [args.width] * args.layers,
                     architecture='context', aux_classes=(3, 3, 3, 4, 4),
                     device=args.device)
    optimizer = torch.optim.RMSprop(agent.parameters(), lr=.0001, alpha=.99, eps=.00001)
    forward = agent.net.forward_with_aux_dense if dense else agent.forward_with_aux
    objective = None
    compile_loss = compile_mode is not None and compile_mode.endswith('-loss')
    if compile_loss:
        compile_mode = compile_mode[:-5]
    if compile_mode:
        if compile_loss:
            objective = _make_training_objective(forward, agent.net.aux_classes,
                (((0, 1, 2), .1), ((3,), .05), ((4,), .05)), dense)
            objective = torch.compile(objective, mode=compile_mode, fullgraph=True, dynamic=False)
        else:
            forward = torch.compile(forward, mode=compile_mode, fullgraph=True, dynamic=False)
    buffers = create_buffers(args.unroll_length, args.batch_size, [[2668]], [[111]],
                             ['cpu'], aux_size=5, dtype=torch.int8)['cpu'][0]
    for key, value in batch.items():
        buffers[key].copy_(value)
    free, full = queue.Queue(), queue.Queue()
    reader = BatchReader(free, full, buffers, args.batch_size, batch_major=True,
                         pin_memory=pinned)
    returns = [deque(maxlen=100)]

    def synchronize():
        if args.device != 'cpu':
            torch.cuda.synchronize(int(args.device))

    def step():
        for i in range(args.batch_size):
            full.put(i)
        payload = reader.get()
        for _ in range(args.batch_size):
            free.get_nowait()
        if compile_mode and hasattr(getattr(torch, 'compiler', None), 'cudagraph_mark_step_begin'):
            torch.compiler.cudagraph_mark_step_begin()
        return learn(0, {}, agent, payload, optimizer, args.device, 40, returns,
                     sync_weights=False, precision=precision,
                     aux_groups=(((0, 1, 2), .1), ((3,), .05), ((4,), .05)),
                     transfer_done=reader.mark_transferred, learner_forward=forward,
                     dense_learner=dense, learner_objective=objective)

    synchronize()
    start = time.perf_counter()
    first_loss = step()
    synchronize()
    cold_seconds = time.perf_counter() - start
    first_loss = float(first_loss)
    for _ in range(args.warmup):
        step()
    synchronize()
    if args.device != 'cpu':
        torch.cuda.reset_peak_memory_stats(int(args.device))
    blocks = []
    for _ in range(args.blocks):
        start = time.perf_counter()
        for _ in range(args.repeats):
            loss = step()
        synchronize()
        blocks.append((time.perf_counter() - start) / args.repeats)
    result = dict(precision=precision, dense=dense, pinned=pinned, compile_mode=compile_mode,
                  compile_loss=compile_loss,
                  first_step_seconds=cold_seconds, first_loss=first_loss, final_loss=float(loss),
                  block_step_ms=[seconds * 1000 for seconds in blocks],
                  median_step_ms=statistics.median(blocks) * 1000,
                  samples_per_second=args.batch_size * args.unroll_length / statistics.median(blocks),
                  peak_allocated_bytes=(torch.cuda.max_memory_allocated(int(args.device))
                                        if args.device != 'cpu' else None))
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', default='0')
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--unroll_length', type=int, default=20)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--layers', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--blocks', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='experiments/learner-pipeline.json')
    args = parser.parse_args()
    if min(args.batch_size, args.unroll_length, args.width, args.layers, args.repeats, args.blocks) < 1 or args.warmup < 0:
        parser.error('Dimensions/repetitions must be positive and warmup nonnegative')
    torch.set_num_threads(1)
    batch = fixture(args.batch_size, args.unroll_length, args.seed)
    report = dict(python=platform.python_version(), torch=torch.__version__,
                  gpu=torch.cuda.get_device_name(int(args.device)) if args.device != 'cpu' else None,
                  settings=vars(args), declaration_fraction=float((batch['state'][..., 525] > .5).float().mean()),
                  method='Fixed native decisions and synthetic targets; real learn; gather and transfer included; '
                         'no actors, process IPC or weight publishing; synchronized timing per repeat block',
                  cases={})
    root = Path(__file__).resolve().parents[1]
    sources = [Path(__file__).resolve(), *[
        root / 'rlcard' / 'agents' / 'dmc_agent' / name
        for name in ('trainer.py', 'utils.py', 'model.py', 'context_model.py', 'collector.py')
    ], *[(root / 'cpp_engine' / 'a3_v12' / name)
          for name in ('a3dizhu.cpp', 'a3dizhu.h', 'bindings.cpp')]]
    report['source_sha256'] = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources}
    for name in args.cases:
        print(json.dumps({'starting_case': name}), flush=True)
        report['cases'][name] = run_case(name, batch, args)
        print(json.dumps({name: report['cases'][name]}), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
