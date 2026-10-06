"""Compare eager and bucketed CUDA graph scoring on identical native decisions."""
import argparse
import contextlib
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from examples.benchmark_context import decisions, measure
from rlcard.agents.dmc_agent.collector import score_action_groups
from rlcard.agents.dmc_agent.context_model import ContextInferenceGraphs, DECLARE_FLAG_INDEX, ContextCompiledInference
from rlcard.agents.dmc_agent.model import DMCAgent


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', default='0')
    parser.add_argument('--envs', type=int, default=128)
    parser.add_argument('--precision', choices=('fp32', 'bf16', 'fp16'), default='bf16')
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--precast-linear', action='store_true')
    parser.add_argument('--bucket-multiple', type=int, default=64)
    parser.add_argument('--max-graphs', type=int, default=12)
    parser.add_argument('--compiled', action='store_true')
    parser.add_argument('--output', default='experiments/context-graphs.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    obs, actions, offsets, _ = decisions(args.envs, 42)
    agent = DMCAgent([2668], [111], [768] * 5, device=args.device,
                     architecture='context', aux_classes=(3, 3, 3, 4, 4))
    agent.eval()
    dtype = {'fp32': None, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    agent.inference_dtype = dtype
    if args.precast_linear:
        if dtype is None:
            parser.error('--precast-linear requires bf16 or fp16')
        for module in agent.net.modules():
            if isinstance(module, torch.nn.Linear):
                module.to(dtype=dtype)
    runner = ContextInferenceGraphs(agent.net, dtype=dtype, max_graphs=args.max_graphs,
                                    bucket_multiple=args.bucket_multiple)
    runner.declaration_flag_index = agent.net.declaration_flag_index
    graphed = copy.copy(agent)
    graphed.net = runner

    def score(policy):
        return score_action_groups(policy, obs, actions, offsets, 4096, contextlib.nullcontext())

    expected = score(agent)
    torch.cuda.synchronize()
    initial_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    actual = score(graphed)
    torch.cuda.synchronize()
    startup = time.perf_counter() - start
    tolerance = .04 if dtype is not None else 2e-5
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    # The same actor graph must see full parameter snapshots without recapture,
    # including weights that autocast might otherwise leave stale in its cache.
    snapshot = {key: value.clone() for key, value in agent.net.state_dict().items()}
    refreshed = {key: value.float().clone() for key, value in snapshot.items()}
    refreshed['output_head.weight'].mul_(1.5)
    refreshed['output_head.bias'].add_(.25)
    refreshed['obs_static_proj.0.weight'].mul_(1.05)
    agent.net.load_state_dict(refreshed)
    changed_eager, changed_graph = score(agent), score(graphed)
    np.testing.assert_allclose(changed_graph, changed_eager, rtol=tolerance, atol=tolerance)
    if np.max(np.abs(changed_graph - actual)) < .1:
        raise AssertionError('The graph did not see the new actor parameters')
    agent.net.load_state_dict(snapshot)
    del snapshot, refreshed
    checks = []
    # Switch buckets in both directions and exercise empty and mixed histories.
    # These are scoring checks, outside the timed throughput measurements.
    for count, case in ((min(17, len(obs)), 'short'), (len(obs), 'full'),
                        (min(3, len(obs)), 'empty_history'),
                        (min(33, len(obs)), 'mixed_heads'), (len(obs), 'full_again')):
        sample_obs = obs[:count].copy()
        sample_actions = actions[:offsets[count]].copy()
        if case == 'empty_history':
            sample_obs[:, 556:] = 0
        if case == 'mixed_heads':
            sample_obs[::2, 525] = 1
            for row in range(0, count, 2):
                sample_actions[offsets[row]:offsets[row + 1], :52] = 1
        a = score_action_groups(agent, sample_obs, sample_actions, offsets[:count + 1],
                                4096, contextlib.nullcontext())
        b = score_action_groups(graphed, sample_obs, sample_actions, offsets[:count + 1],
                                4096, contextlib.nullcontext())
        np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)
        checks.append(dict(case=case, states=count, max_abs_difference=float(np.max(np.abs(a - b)))))
    eager = measure(lambda: score(agent), args.device, args.repeats)
    graph = measure(lambda: score(graphed), args.device, args.repeats)
    graph_allocated = torch.cuda.memory_allocated() - initial_allocated
    graph_peak = torch.cuda.max_memory_allocated()
    graph_reserved = torch.cuda.memory_reserved()
    compiled = None
    if args.compiled:
        compiled_agent = copy.copy(agent)
        compiled_agent.net = ContextCompiledInference(agent.net)
        torch.cuda.synchronize()
        start = time.perf_counter()
        compiled_values = score(compiled_agent)
        torch.cuda.synchronize()
        compile_startup = time.perf_counter() - start
        np.testing.assert_allclose(compiled_values, expected, rtol=tolerance, atol=tolerance)
        compiled_checks = []
        for count, mixed in ((min(17, len(obs)), False), (min(33, len(obs)), True)):
            sample_obs = obs[:count].copy()
            if mixed:
                sample_obs[::2, 525] = 1
            sample_actions, sample_offsets = actions[:offsets[count]], offsets[:count + 1]
            a = score_action_groups(agent, sample_obs, sample_actions, sample_offsets,
                                    4096, contextlib.nullcontext())
            b = score_action_groups(compiled_agent, sample_obs, sample_actions, sample_offsets,
                                    4096, contextlib.nullcontext())
            np.testing.assert_allclose(a, b, rtol=tolerance, atol=tolerance)
            compiled_checks.append(dict(states=count, mixed_heads=mixed,
                                        max_abs_difference=float(np.max(np.abs(a - b)))))
        compiled = dict(first_call_seconds=compile_startup,
                        checks=compiled_checks,
                        max_abs_difference=float(np.max(np.abs(compiled_values - expected))),
                        **measure(lambda: score(compiled_agent), args.device, args.repeats))
    report = dict(torch=torch.__version__, precision=args.precision, precast_linear=args.precast_linear,
                  bucket_multiple=args.bucket_multiple, max_graphs=args.max_graphs,
                  cached_buckets=[(key[0], key[1]) for key in runner.cache],
                  checks=checks, states=len(obs),
                  candidates=len(actions), first_graph_seconds=startup,
                  max_abs_difference=float(np.max(np.abs(actual - expected))),
                  refreshed_max_abs_difference=float(np.max(np.abs(changed_graph - changed_eager))),
                  graph_cache_size=len(runner.cache), graph_hits=runner.hits,
                  graph_allocation_bytes=graph_allocated,
                  peak_allocated_bytes=graph_peak,
                  reserved_bytes=graph_reserved,
                  eager=eager, graph=graph, compiled=compiled,
                  speed_ratio=eager['median_ms'] / graph['median_ms'])
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
