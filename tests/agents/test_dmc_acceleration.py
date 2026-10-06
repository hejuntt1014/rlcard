"""Ownership and optimizer semantics of accelerated DMC paths."""
import copy
import contextlib
import tempfile
import time
import unittest
from queue import Full

import numpy as np
import torch
from torch import multiprocessing as mp

from rlcard.agents.dmc_agent.utils import SharedIndexQueue
from rlcard.agents.dmc_agent.model import DMCAgent
from rlcard.agents.dmc_agent.trainer import _make_training_objective
from rlcard.agents.dmc_agent.graph_learner import CUDALearner


def _ring_producer(ring, base):
    for start in range(0, 100, 4):
        while True:
            try:
                ring.put_many(range(base + start, base + start + 4))
                break
            except Full:
                time.sleep(.001)


def _ring_consumer(ring, result):
    values = []
    deadline = time.monotonic() + 20
    while len(values) < 200 and time.monotonic() < deadline:
        values.extend(ring.get_many(7))
        time.sleep(.001)
    result.put(values)


class TestSharedIndices(unittest.TestCase):
    def test_wraparound_and_exact_claims(self):
        ring = SharedIndexQueue(mp.get_context('spawn'), 4)
        ring.put_many([0, 1, 2, 3])
        self.assertEqual(ring.get_many(3), [0, 1, 2])
        ring.put_many([4, 5, 6])
        self.assertEqual(ring.get_many(8), [3, 4, 5, 6])
        self.assertTrue(ring.empty())

    def test_spawn_multiple_producers_no_missing_or_duplicate_slots(self):
        ctx = mp.get_context('spawn')
        ring, result = SharedIndexQueue(ctx, 8), ctx.Queue()
        processes = [ctx.Process(target=_ring_producer, args=(ring, base)) for base in (0, 100)]
        processes.append(ctx.Process(target=_ring_consumer, args=(ring, result)))
        try:
            for process in processes: process.start()
            values = result.get(timeout=30)
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(sorted(values), list(range(200)))
            for base in (0, 100):
                self.assertEqual([v for v in values if base <= v < base+100], list(range(base, base+100)))
            self.assertTrue(ring.empty())
        finally:
            for process in processes:
                if process.is_alive(): process.terminate(); process.join()
            result.close()


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA integration test')
class TestLearningReplay(unittest.TestCase):
    def test_compiled_actor_observes_in_place_parameter_refresh(self):
        from rlcard.agents.dmc_agent.context_model import ContextCompiledInference
        from rlcard.agents.dmc_agent.collector import score_action_groups
        torch.set_num_threads(1); torch.manual_seed(72)
        agent=DMCAgent([2668],[111],[16],architecture='context',aux_classes=(3,3,3,4,4),device='0')
        agent.eval();agent.inference_dtype=torch.bfloat16
        for layer in agent.net.modules():
            if isinstance(layer,torch.nn.Linear):layer.to(dtype=torch.bfloat16)
        obs=np.zeros((8,2668),dtype=np.int8);obs[:2,525]=1
        actions=np.zeros((16,111),dtype=np.int8);actions[1::2,:52]=1
        offsets=np.arange(9)*2
        def score():return score_action_groups(agent,obs,actions,offsets,4096,contextlib.nullcontext())
        expected=score()
        runner=ContextCompiledInference(agent.net)
        agent.inference_runner=runner
        before=score()
        np.testing.assert_allclose(before,expected,rtol=.04,atol=.04)
        snapshot={key:value.float().clone() for key,value in agent.net.state_dict().items()}
        snapshot['output_head.bias'].add_(1.)
        snapshot['declare_head.bias'].add_(1.)
        agent.load_state_dict(snapshot)
        after=score()
        del agent.inference_runner
        np.testing.assert_allclose(after,score(),rtol=.04,atol=.04)
        self.assertGreater(float(np.min(after-before)),.8)

    def test_accelerated_checkpoint_resumes_without_compilation(self):
        import rlcard
        from rlcard.agents.dmc_agent.trainer import DMCTrainer
        with tempfile.TemporaryDirectory() as folder:
            args=dict(cuda='0',training_device='0',savedir=folder,xpid='resume',num_actors=1,
                      envs_per_actor=3,batch_size=2,unroll_length=2,num_buffers=4,total_frames=16,
                      share_weights=True,mlp_layers=[16],precision='bf16',shared_transport=True,
                      cuda_graph_learner=True,stats_interval=2,weight_sync_interval=1)
            trainer=DMCTrainer(rlcard.make('a3dizhu-v12'),**args)
            self.assertEqual(trainer.start()['frames'],16)
            saved=torch.load(trainer.checkpointpath,map_location='cpu',weights_only=False)
            self.assertTrue(saved['optimizer_state_dict'][0]['param_groups'][0]['capturable'])
            args.update(load_model=True,total_frames=20,cuda_graph_learner=False)
            resumed=DMCTrainer(rlcard.make('a3dizhu-v12'),**args)
            self.assertEqual(resumed.start()['frames'],20)
            saved=torch.load(resumed.checkpointpath,map_location='cpu',weights_only=False)
            group=saved['optimizer_state_dict'][0]['param_groups'][0]
            self.assertFalse(group['capturable'])
            self.assertIsInstance(group['lr'],float)

    def test_phase_changes_and_learning_rate_do_not_add_warmup_updates(self):
        torch.set_num_threads(1); torch.manual_seed(31)
        reference = DMCAgent([2668], [111], [16], architecture='context',
                             aux_classes=(3,3,3,4,4), device='0')
        actual = copy.deepcopy(reference)
        opts = [torch.optim.RMSprop(agent.parameters(), lr=torch.tensor(.001, device='cuda'),
                                   alpha=.99, eps=.00001, capturable=True)
                for agent in (reference, actual)]
        groups = (((0,1,2),.1), ((3,),.05), ((4,),.05))
        objectives = [_make_training_objective(a.net.forward_with_aux_dense, a.net.aux_classes, groups, True)
                      for a in (reference, actual)]
        runner = CUDALearner(actual, opts[1], objectives[1], 'fp32', 40)
        for i, phase in enumerate(('play', None, 'declare', None, 'play')):
            data = dict(state=torch.randn(2, 2, 2668), action=torch.randn(2, 2, 111),
                        target=torch.randn(2, 2), aux_target=torch.zeros(2, 2, 5, dtype=torch.long))
            data['state'][...,525] = 1 if phase == 'declare' else 0
            if phase is None: data['state'][0,:,525] = 1
            data['aux_target'][0,0] = -1
            for opt in opts: opt.param_groups[0]['lr'].fill_(.001/(i+1))
            gpu = {k:v.flatten(0,1).cuda() for k,v in data.items()}
            opts[0].zero_grad(set_to_none=True)
            expected_loss = objectives[0](gpu['state'],gpu['action'],gpu['target'],gpu['aux_target'],phase)
            expected_loss.backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(),40)
            opts[0].step()
            loss = runner.step(data,phase)
            torch.testing.assert_close(loss, expected_loss.detach(), atol=2e-5, rtol=2e-5)
            for left,right in zip(reference.parameters(),actual.parameters()):
                torch.testing.assert_close(left,right,atol=2e-6,rtol=2e-5)
                self.assertEqual(left.grad is None, right.grad is None)
                expected_state=opts[0].state.get(left)
                self.assertEqual(float(opts[1].state[right]['step']),
                                 float(expected_state['step']) if expected_state else 0.)
        self.assertEqual(len(runner.cache),3)

    def test_compiled_rmsprop_matches_fixed_gradients_with_changing_lr(self):
        torch.manual_seed(7)
        left = torch.nn.Parameter(torch.randn(16,16,device='cuda'))
        right = torch.nn.Parameter(left.detach().clone())
        old = torch.optim.RMSprop([left],lr=.001,alpha=.99,eps=.00001)
        new = torch.optim.RMSprop([right],lr=torch.tensor(.001,device='cuda'),alpha=.99,eps=.00001,capturable=True)
        update = torch.compile(new.step,fullgraph=False,options={'triton.cudagraphs':False})
        for i in range(12):
            gradient=torch.randn_like(left) if i%3 else torch.zeros_like(left)
            left.grad=gradient.clone(); right.grad=gradient.clone()
            old.param_groups[0]['lr']=.001/(i+1)
            new.param_groups[0]['lr'].fill_(.001/(i+1))
            old.step();update()
            torch.testing.assert_close(left,right,atol=2e-6,rtol=2e-6)
            torch.testing.assert_close(old.state[left]['square_avg'],new.state[right]['square_avg'],atol=1e-7,rtol=2e-6)
