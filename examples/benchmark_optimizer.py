"""Time compiled RMSprop and clipping on the same fixed native decisions."""
import argparse
import json
import time
from collections import deque
from pathlib import Path

import torch

from examples.benchmark_learner_pipeline import fixture
from rlcard.agents.dmc_agent.model import DMCAgent
from rlcard.agents.dmc_agent.trainer import _make_training_objective, learn


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', default='experiments/optimizer-benchmark.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = {k: v.pin_memory() for k, v in fixture(32, 20, 42).items()}
    results = {}
    reference = None
    for mode in ('eager', 'compiled_optimizer', 'compiled_clip_optimizer'):
        torch.manual_seed(42)
        agent = DMCAgent([2668], [111], [768]*5, architecture='context',
                         aux_classes=(3,3,3,4,4), device='0')
        compiled = mode != 'eager'
        lr = torch.tensor(.0001, device='cuda') if compiled else .0001
        opt = torch.optim.RMSprop(agent.parameters(), lr=lr, alpha=.99, eps=.00001,
                                 capturable=compiled)
        objective = torch.compile(_make_training_objective(agent.net.forward_with_aux_dense,
            agent.net.aux_classes, (((0,1,2),.1),((3,),.05),((4,),.05)), True),
            mode='reduce-overhead', fullgraph=True, dynamic=False)
        original_step = opt.step
        original_clip = torch.nn.utils.clip_grad_norm_
        if mode == 'compiled_optimizer':
            opt.step = torch.compile(original_step, fullgraph=False, mode='default')
        elif mode == 'compiled_clip_optimizer':
            def update():
                original_clip(agent.parameters(), 40)
                original_step()
            opt.step = torch.compile(update, fullgraph=False, mode='default')
            torch.nn.utils.clip_grad_norm_ = lambda *a, **k: None
        returns = [deque(maxlen=100)]
        def step(index):
            rate = .0001 * (1 - min(index, 1000)/2000)
            if compiled:
                opt.param_groups[0]['lr'].fill_(rate)
            else:
                opt.param_groups[0]['lr'] = rate
            torch.compiler.cudagraph_mark_step_begin()
            return learn(0, {}, agent, data, opt, '0', 40, returns,
                         sync_weights=False, precision='bf16', dense_learner=True,
                         learner_objective=objective)
        try:
            torch.cuda.synchronize(); started = time.perf_counter()
            step(0); torch.cuda.synchronize()
            cold = time.perf_counter()-started
            for i in range(1,10): step(i)
            torch.cuda.synchronize()
            weights = {n: p.detach().clone() for n,p in agent.net.named_parameters()}
            if reference is None: reference=weights
            difference = max(float((weights[n]-p).abs().max()) for n,p in reference.items())
            blocks=[]
            for block in range(3):
                started=time.perf_counter()
                for i in range(100): loss=step(10+block*100+i)
                torch.cuda.synchronize()
                blocks.append((time.perf_counter()-started)*10)
            results[mode]=dict(cold_seconds=cold,block_ms=blocks,
                              weight_difference_after_10=difference, final_loss=float(loss))
            print(mode,results[mode],flush=True)
        finally:
            torch.nn.utils.clip_grad_norm_=original_clip
    Path(args.output).write_text(json.dumps(results,indent=2))


if __name__ == '__main__': main()
