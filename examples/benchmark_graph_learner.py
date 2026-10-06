"""Compare complete learning graph replay against an eager capturable update."""
import argparse
import copy
import json
import time
from pathlib import Path

import torch

from examples.benchmark_learner_pipeline import fixture
from rlcard.agents.dmc_agent.model import DMCAgent
from rlcard.agents.dmc_agent.graph_learner import CUDALearner
from rlcard.agents.dmc_agent.trainer import _make_training_objective


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',default='experiments/graph-learner.json')
    parser.add_argument('--compile',action='store_true')
    parser.add_argument('--compile-optimizer', action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(42)
    data={k:v.pin_memory() for k,v in fixture(32,20,42).items()}
    original=DMCAgent([2668],[111],[768]*5,architecture='context',aux_classes=(3,3,3,4,4),device='0')
    report={}; reference=None
    for mode in ('eager','graph'):
        agent=copy.deepcopy(original)
        optimizer=torch.optim.RMSprop(agent.parameters(),lr=torch.tensor(.0001,device='cuda'),
                                     alpha=.99,eps=.00001,capturable=True)
        objective=_make_training_objective(agent.net.forward_with_aux_dense,agent.net.aux_classes,
                                          (((0,1,2),.1),((3,),.05),((4,),.05)),True)
        if args.compile:
            objective=torch.compile(objective,fullgraph=True,dynamic=False,
                                    options={'triton.cudagraphs':False})
        runner=CUDALearner(agent,optimizer,objective,'bf16',40)
        if args.compile_optimizer:
            optimizer.step=torch.compile(optimizer.step, fullgraph=False,
                                         options={'triton.cudagraphs':False})
        def step(i):
            optimizer.param_groups[0]['lr'].fill_(.0001*(1-min(i,1000)/2000))
            if mode=='graph': return runner.step(data)
            inputs={k:v.flatten(0,1).to('cuda',non_blocking=True) for k,v in data.items()
                    if k in ('state','action','target','aux_target')}
            return runner._update(inputs,None)
        start=time.perf_counter();step(0);torch.cuda.synchronize()
        cold=time.perf_counter()-start
        for i in range(1,10):step(i)
        weights={n:p.detach().clone() for n,p in agent.net.named_parameters()}
        if reference is None:reference=weights
        difference=max(float((weights[n]-p).abs().max()) for n,p in reference.items())
        blocks=[]
        for b in range(3):
            torch.cuda.synchronize();start=time.perf_counter()
            for i in range(100): loss=step(10+b*100+i)
            torch.cuda.synchronize();blocks.append((time.perf_counter()-start)*10)
        report[mode]=dict(cold_seconds=cold,block_ms=blocks,weight_difference_after_10=difference,
                          final_loss=float(loss.detach()),steps=[float(s['step']) for s in optimizer.state.values()])
        print(mode,report[mode],flush=True)
    Path(args.output).write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
