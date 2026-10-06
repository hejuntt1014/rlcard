"""Feature-schema, decision-label and compact-scoring integration regressions."""
import contextlib
import importlib.util
import queue
import tempfile
import unittest
from unittest.mock import patch
from collections import deque

import numpy as np
import torch

import rlcard
from rlcard.agents.dmc_agent.collector import score_action_groups, choose_action_indices, PythonPool
from rlcard.agents.dmc_agent.model import DMCModel
from rlcard.agents.dmc_agent.trainer import DMCTrainer, learn
from rlcard.agents.dmc_agent.utils import create_buffers, TrajectoryWriter, BatchReader
from rlcard.envs.a3dizhu.dmc import A3Adapter, NativePool


class TestCompactScoring(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_variable_candidates_encode_only_unique_states(self):
        model = DMCModel([[2668]], [[111]], [16], device='cpu',
                         architecture='context', aux_classes=(3,3,3,4,4))
        agent = model.get_agent(0)
        obs = np.zeros((3, 2668), dtype=np.float32)
        obs[0, 525] = 1
        actions = np.random.default_rng(42).integers(0, 2, (8, 111)).astype(np.float32)
        offsets = [0, 2, 3, 8]
        with torch.no_grad():
            expected = agent.forward(torch.tensor(np.repeat(obs, [2,1,5], axis=0)), torch.tensor(actions))
        with patch.object(agent.net, 'encode_state', wraps=agent.net.encode_state) as encode:
            actual = score_action_groups(agent, obs, actions, offsets, 3, contextlib.nullcontext())
        self.assertEqual(encode.call_count, 1)
        self.assertEqual(encode.call_args.args[0].shape[0], 3)
        np.testing.assert_allclose(actual, expected.numpy(), rtol=1e-5, atol=1e-6)

    def test_batch_major_and_decision_labels(self):
        buffers = create_buffers(2, 2, [[1]], [[1]], ['cpu'], aux_size=1)['cpu']
        free, full = [queue.Queue()], [queue.Queue()]
        free[0].put(0); free[0].put(1)
        writer = TrajectoryWriter(2, free, full, buffers)
        steps = [[(np.array([i], dtype=np.float32), np.array([1.]), np.array([i % 3]))
                  for i in range(4)]]
        writer.add_episode(steps, [2.], auxiliary=[np.array([99])])
        writer.flush()
        batch = BatchReader(free[0], full[0], buffers[0], 2, batch_major=True).get()
        self.assertEqual(batch['state'].flatten().tolist(), [0., 1., 2., 3.])
        self.assertEqual(batch['aux_target'].flatten().tolist(), [0, 1, 2, 0])

    def test_forced_and_random_decisions_skip_network(self):
        with patch('rlcard.agents.dmc_agent.collector.score_action_groups', side_effect=AssertionError('unneeded forward')):
            forced = choose_action_indices(None, np.zeros((3, 2)), np.zeros((3, 2)), [0,1,2,3], 0., 10, None)
            np.testing.assert_array_equal(forced, [0,0,0])
            exploratory = choose_action_indices(None, np.zeros((2, 2)), np.zeros((5, 2)), [0,2,5], 1., 10, None)
            self.assertTrue(0 <= exploratory[0] < 2 and 0 <= exploratory[1] < 3)


@unittest.skipUnless(importlib.util.find_spec('a3dizhu_v12_cpp'), 'Optional V12 extension')
class TestContextPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_fixed_rules_survive_reset_and_native_pool(self):
        rules = dict(declare_require_both_spades=True, straight_start_val=2, straight_end_val=11)
        env = rlcard.make('a3dizhu-v12', config={'rules': rules, 'seed': 3})
        for _ in range(5):
            state, _ = env.reset()
            np.testing.assert_array_equal(state['obs'][537:544], [0,1,1,0,0,0,1])
        pool = NativePool(env, 2, 3, 1000)
        for i in range(2):
            for _ in range(3):
                pool.reset(i)
                p = pool.vec.get_player_id(i)
                np.testing.assert_array_equal(pool.vec.encode_obs(i,p)[537:544], [0,1,1,0,0,0,1])

    def test_episode_auxiliary_labels_are_captured_before_declaration(self):
        import a3dizhu_v12_cpp as native
        e = native.CppEngine(); e.seed(3); e.reset()
        e.set_rules(False, 1, 12)
        before = np.array(e.get_aux_targets()[0])
        e.step('declare')
        after = np.array(e.get_aux_targets()[0])
        self.assertFalse(np.array_equal(before, after))
        buffers = create_buffers(1, 1, [[2668]], [[111]], ['cpu'], aux_size=5)['cpu']
        free, full = [queue.Queue()], [queue.Queue()]; free[0].put(0)
        writer = TrajectoryWriter(1, free, full, buffers)
        writer.add_episode([[(np.zeros(2668), np.zeros(111), before)]], [1.], auxiliary=[after])
        writer.flush()
        np.testing.assert_array_equal(buffers[0]['aux_target'][0,0], before)

    def test_v12_train_resume_and_evaluate(self):
        from examples.evaluate_dmc import load_policy, evaluate, summarize
        with tempfile.TemporaryDirectory() as directory:
            args = dict(savedir=directory, xpid='context', num_actors=1, envs_per_actor=3,
                        total_frames=32, batch_size=2, unroll_length=2, num_buffers=4,
                        mlp_layers=[16], share_weights=True, weight_sync_interval=1,
                        stats_interval=2, actor_timeout=45)
            trainer = DMCTrainer(rlcard.make('a3dizhu-v12'), **args)
            stats = trainer.start()
            self.assertEqual(stats['frames'], 32)
            self.assertTrue(np.isfinite(list(v for k,v in stats.items() if k.startswith('loss_'))).all())
            payload = torch.load(trainer.checkpointpath, weights_only=True)
            self.assertEqual(payload['model_spec']['feature_schema'], 'a3-v12-lite-v1')
            self.assertEqual(payload['model_spec']['reward_mode'], 'game')
            args.update(load_model=True, total_frames=40)
            self.assertEqual(DMCTrainer(rlcard.make('a3dizhu-v12'), **args).start()['frames'], 40)
            with self.assertRaisesRegex(ValueError, 'specification'):
                DMCTrainer(rlcard.make('a3dizhu-v12', config={'reward_mode': 'shaped'}), **args).start()
            with self.assertRaisesRegex(ValueError, 'specification'):
                DMCTrainer(rlcard.make('a3dizhu-v12', config={'rules': {
                    'declare_require_both_spades': True, 'straight_start_val': 2,
                    'straight_end_val': 11}}), **args).start()
            env = rlcard.make('a3dizhu-v12', config={'greedy_ratio': 1.})
            model, _ = load_policy(trainer.checkpointpath, env)
            result = summarize(evaluate(model, env, [4], max_steps=1000))
            self.assertEqual(result['games'], 4)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA integration test')
class TestContextCUDA(unittest.TestCase):
    def test_precisions_declaration_and_mixed_batches(self):
        torch.set_num_threads(1)
        for precision in ('fp32', 'bf16', 'fp16'):
            for mixed in (False, True):
                with self.subTest(precision=precision, mixed=mixed):
                    torch.manual_seed(42)
                    model = DMCModel([[2668]], [[111]], [32], device='0', architecture='context',
                                     aux_classes=(3,3,3,4,4))
                    agent = model.get_agent(0)
                    state = torch.rand(4, 2, 2668)
                    state[..., 525] = 1
                    if mixed:
                        state[::2, :, 525] = 0
                    action = torch.zeros(4, 2, 111)
                    action[::2, :, :52] = 1
                    batch = dict(state=state, action=action, target=torch.ones(4,2),
                                 aux_target=torch.full((4,2,5), -1),
                                 done=torch.ones(4,2,dtype=torch.bool), episode_return=torch.ones(4,2))
                    optimizer = torch.optim.RMSprop(agent.parameters(), lr=1e-4)
                    scaler = torch.amp.GradScaler('cuda') if precision == 'fp16' else None
                    initial = agent.net.declare_head.weight.detach().clone()
                    for _ in range(8):
                        loss = learn(0, {}, agent, batch, optimizer, '0', 40, [deque()],
                                     sync_weights=False, precision=precision, scaler=scaler)
                        self.assertTrue(torch.isfinite(loss))
                    self.assertFalse(torch.equal(initial, agent.net.declare_head.weight))
                    self.assertTrue(all(torch.isfinite(p).all() for p in agent.parameters()))
                    if not mixed:
                        self.assertTrue(all(p.grad is None for p in agent.net.output_head.parameters()))

    def test_low_precision_compact_scoring(self):
        for history in ('mlp', 'transformer'):
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(history=history, dtype=dtype):
                    model = DMCModel([[2668]], [[111]], [32], device='0', architecture='context',
                                     history_encoder=history, aux_classes=())
                    model.eval()
                    agent = model.get_agent(0)
                    agent.inference_dtype = dtype
                    obs = np.zeros((3, 2668), dtype=np.float32)
                    obs[0,525] = 1
                    actions = np.zeros((7,111), dtype=np.float32)
                    actions[0,:52] = 1
                    with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
                        reference = agent.forward(torch.as_tensor(np.repeat(obs,[2,1,4],axis=0),device='cuda'),
                                                  torch.as_tensor(actions,device='cuda')).float().cpu().numpy()
                    actual = score_action_groups(agent, obs, actions, [0,2,3,7], 3, contextlib.nullcontext())
                    np.testing.assert_allclose(actual, reference, atol=.03, rtol=.03)


if __name__ == '__main__':
    unittest.main()
