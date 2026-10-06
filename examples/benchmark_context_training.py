"""Measure fixed-shape context training kernels without environment overhead."""
import argparse
import contextlib
import copy
import json
import statistics
import time
from pathlib import Path

import torch

from rlcard.agents.dmc_agent.context_model import ContextDMCNet
from rlcard.agents.dmc_agent.trainer import auxiliary_loss, compute_loss


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch', type=int, default=640)
    parser.add_argument('--precision', choices=('fp32', 'bf16', 'fp16'), default='bf16')
    parser.add_argument('--repeats', type=int, default=50)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--compile-loss', action='store_true')
    parser.add_argument('--phase', choices=('mixed', 'play', 'declare'), default='mixed')
    parser.add_argument('--compile-mode', default='default',
                        choices=('default', 'reduce-overhead', 'max-autotune'))
    parser.add_argument('--output', default='experiments/context-training-kernels.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type != 'cuda' and args.precision != 'fp32':
        parser.error('half precision measurements require CUDA')
    sync = torch.cuda.synchronize if device.type == 'cuda' else lambda: None
    autocast = (lambda: torch.autocast('cuda', dtype=(torch.bfloat16 if args.precision == 'bf16'
                                                     else torch.float16))) if args.precision != 'fp32' else contextlib.nullcontext
    original = ContextDMCNet([2668], [111]).to(device)
    # Binary inputs match the native feature range; 1/16 declaration decisions
    # exercise both heads without making the benchmark mostly declaration work.
    obs = torch.randint(0, 2, (args.batch, 2668), device=device).float()
    obs[:, 525] = 0
    obs[::16, 525] = 1
    if args.phase != 'mixed':
        obs[:, 525] = int(args.phase == 'declare')
    phase = None if args.phase == 'mixed' else args.phase
    action = torch.randint(0, 2, (args.batch, 111), device=device).float()
    target = torch.randn(args.batch, device=device)
    labels = torch.stack([torch.randint(0, n, (args.batch,), device=device)
                          for n in original.aux_classes], 1)
    labels[::7, 3:] = -1

    def loss(values, auxiliary, targets, task_labels):
        return compute_loss(values.float(), targets.float()) + auxiliary_loss(
            auxiliary.float(), task_labels, original.aux_classes,
            (((0, 1, 2), .1), ((3,), .05), ((4,), .05)))

    report = dict(torch=torch.__version__, device=str(device), batch=args.batch,
                  precision=args.precision, compile_mode=args.compile_mode, phase=args.phase,
                  loss='current trainer MSE plus masked, vectorized auxiliary CE',
                  method='same weights and fixed synthetic binary samples; forward, loss and backward; no optimizer',
                  results={})
    reference = None
    modes = ['sparse', 'dense'] + (['compiled_dense'] if args.compile else [])
    if args.compile_loss:
        modes.append('compiled_loss')
    for mode in modes:
        model = copy.deepcopy(original)
        forward = model.forward_with_aux if mode == 'sparse' else model.forward_with_aux_dense
        if mode == 'compiled_dense':
            forward = torch.compile(forward, mode=args.compile_mode, fullgraph=True)

        def objective_function(states, actions, targets, task_labels):
            with autocast():
                outputs = (forward(states, actions) if mode == 'sparse'
                           else forward(states, actions, phase=phase))
            return loss(*outputs, targets, task_labels)

        objective_forward = (torch.compile(objective_function, mode=args.compile_mode, fullgraph=True)
                             if mode == 'compiled_loss' else objective_function)

        def step():
            model.zero_grad(set_to_none=True)
            objective = objective_forward(obs, action, target, labels)
            objective.backward()
            return objective

        sync()
        start = time.perf_counter()
        objective = step()
        sync()
        startup = time.perf_counter() - start
        gradients = {name: p.grad.detach().float().clone() if p.grad is not None else torch.zeros_like(p)
                     for name, p in model.named_parameters()}
        if reference is None:
            reference = (objective.detach().float().clone(), gradients)
        differences = dict(loss_abs_difference=float((objective.detach().float() - reference[0]).abs()),
                           gradient_max_abs=max(float((gradients[name] - expected).abs().max())
                                                for name, expected in reference[1].items()))
        for _ in range(args.warmup):
            step()
        times = []
        for _ in range(args.repeats):
            sync()
            start = time.perf_counter()
            step()
            sync()
            times.append((time.perf_counter() - start) * 1000)
        report['results'][mode] = dict(first_step_seconds=startup,
                                      median_ms=statistics.median(times),
                                      samples_per_second=args.batch / (statistics.median(times) / 1000),
                                      **differences, samples_ms=times)
        print(json.dumps({mode: report['results'][mode]}, indent=2), flush=True)
        del model, forward, gradients, objective, objective_forward
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
