"""DMC transport, policy compatibility, and real spawn-process smoke tests."""
import contextlib
import copy
import importlib.util
import queue
import tempfile
import unittest
from unittest.mock import Mock, patch
from collections import deque

import numpy as np
import torch

import rlcard
from rlcard.agents.dmc_agent.model import DMCModel, DMCNet
from rlcard.agents.dmc_agent.collector import PythonPool, RLCardAdapter, action_features
from rlcard.agents.dmc_agent.trainer import DMCTrainer, auxiliary_loss, learn
from rlcard.agents.dmc_agent.utils import BatchReader, TrajectoryWriter, create_buffers


class BrokenEnv:
    name = 'broken'
    num_players = 1
    state_shape = [[2]]
    action_shape = [[2]]

    def seed(self, seed):
        pass

    def reset(self):
        raise RuntimeError('deliberate actor error')


class TestDMC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_role_shapes_and_upstream_mlp_weights(self):
        env = rlcard.make('doudizhu')
        model = DMCModel(env.state_shape, env.action_shape, [16], device='cpu',
                         architecture='mlp', aux_classes=())
        for p, shape in enumerate(env.state_shape):
            self.assertEqual(model.get_agent(p).forward(torch.zeros(2, *shape),
                             torch.zeros(2, 54)).shape, (2,))
        self.assertIn('fc_layers.0.weight', model.get_agent(0).state_dict())
        with self.assertRaisesRegex(ValueError, 'identical'):
            DMCModel(env.state_shape, env.action_shape, device='cpu', share_weights=True)

    def test_ci_metadata_and_environment_allowlist(self):
        from rlcard.agents.dmc_agent.file_writer import gather_metadata
        repo = Mock()
        repo.head.is_detached = True
        repo.commit.return_value.hexsha = 'abc'
        repo.is_dirty.return_value = False
        repo.git_dir = '.git'
        with patch('rlcard.agents.dmc_agent.file_writer.git.Repo', return_value=repo), \
             patch.dict('os.environ', {'DMC_TEST_SECRET': 'must-not-log', 'OMP_NUM_THREADS': '1'}):
            meta = gather_metadata()
        self.assertIsNone(meta['git']['branch'])
        self.assertNotIn('DMC_TEST_SECRET', meta['env'])
        self.assertEqual(meta['env']['OMP_NUM_THREADS'], '1')

    def test_action_features_and_eval_without_raw_actions(self):
        state = {'obs': np.array([.25, -.5]), 'legal_actions': {1: None, 3: None}}
        keys, features = action_features(state, [4])
        np.testing.assert_array_equal(features, [[0, 1, 0, 0], [0, 0, 0, 1]])
        model = DMCModel([[2]], [[4]], [8], device='cpu', architecture='mlp', aux_classes=())
        agent = model.get_agent(0)
        action, info = agent.eval_step(state)
        self.assertIn(action, keys)
        self.assertEqual(set(info['values']), set(keys))

    def test_float_transport_partial_batch_and_slot_ownership(self):
        buffers = create_buffers(2, 3, [[2]], [[2]], ['cpu'])['cpu']
        free, full = [queue.Queue()], [queue.Queue()]
        for index in range(3):
            free[0].put(index)
        writer = TrajectoryWriter(2, free, full, buffers)
        steps = [[(np.array([.25, -.5]), np.array([1., 0.]))] * 2]
        writer.add_episode(steps, [2.], [[.1, .2]])
        writer.flush()
        reader = BatchReader(free[0], full[0], buffers[0], 2)
        self.assertIsNone(reader.get())
        self.assertEqual(len(reader.pending), 1)
        writer.add_episode(steps, [3.])
        writer.flush()
        batch = reader.get()
        np.testing.assert_allclose(batch['target'].numpy(), [[2.3, 3.], [2.2, 3.]])
        np.testing.assert_allclose(batch['episode_return'][-1].numpy(), [2.3, 3.])
        self.assertEqual(float(batch['state'][0, 0, 0]), .25)
        buffers[0]['state'].fill_(99)
        self.assertEqual(float(batch['state'][0, 0, 0]), .25)
        self.assertEqual(free[0].qsize(), 3)

    def test_no_auxiliary_and_masked_auxiliary_updates(self):
        model = DMCModel([[2]], [[2]], [8], device='cpu', architecture='mlp', aux_classes=())
        agent = model.get_agent(0)
        batch = {'state': torch.ones(2, 2, 2), 'action': torch.ones(2, 2, 2),
                 'target': torch.ones(2, 2), 'done': torch.zeros(2, 2, dtype=torch.bool),
                 'episode_return': torch.zeros(2, 2)}
        before = copy.deepcopy(agent.state_dict())
        loss = learn(0, {}, agent, batch, torch.optim.RMSprop(agent.parameters()),
                     'cpu', 40, [deque()], sync_weights=False)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in agent.state_dict().items()))
        logits = torch.randn(4, 9, requires_grad=True)
        masked = auxiliary_loss(logits, torch.full((4, 3), -1), (3, 3, 3))
        self.assertEqual(masked.item(), 0.)
        masked.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_batch_policy_grouping(self):
        for shared in (False, True):
            env = rlcard.make('leduc-holdem')
            model = DMCModel(env.state_shape, [[env.num_actions]] * 2, [8],
                             device='cpu', architecture='mlp', aux_classes=(), share_weights=shared)
            pool = PythonPool(env, 4, 42, RLCardAdapter, 100)
            finished = []
            for _ in range(20):
                episodes, steps = pool.round(model, .2, 16, contextlib.nullcontext())
                finished.extend(episodes)
            self.assertTrue(finished)
            self.assertTrue(all(len(e[0]) == 2 for e in finished))

    def test_invalid_config_fails_before_workers(self):
        env = rlcard.make('leduc-holdem')
        for kwargs in ({'num_buffers': 1, 'batch_size': 2}, {'backend': 'cpp'},
                       {'envs_per_actor': 0}):
            with self.assertRaises(ValueError):
                DMCTrainer(env, **kwargs)

    def test_standard_environments_roll_out(self):
        for name in ('blackjack', 'limit-holdem', 'no-limit-holdem', 'doudizhu',
                     'uno', 'mahjong', 'gin-rummy'):
            with self.subTest(env=name):
                env = rlcard.make(name)
                trainer = DMCTrainer(env, mlp_layers=[8])
                pool = PythonPool(env, 2, 42, RLCardAdapter, 10000)
                model = trainer.model_func('cpu')
                for _ in range(8):
                    pool.round(model, .2, 64, contextlib.nullcontext())

    def test_a3_python_rules_record_real_actions(self):
        from rlcard.envs.a3dizhu.env import A3DizhuPyEnv
        from rlcard.envs.a3dizhu.dmc import A3Adapter
        env = A3DizhuPyEnv(dict(seed=42, allow_step_back=False, greedy_ratio=1.))
        pool = PythonPool(env, 2, 42, A3Adapter, 500)
        model = DMCModel(env.state_shape, env.action_shape, [8], device='cpu')
        finished = []
        for _ in range(150):
            episodes, _ = pool.round(model, .1, 128, contextlib.nullcontext())
            finished.extend(episodes)
            if finished:
                break
        self.assertTrue(finished)
        trajectory, _, rewards, _ = finished[0]
        self.assertTrue(any(np.any(a) for steps in trajectory for _, a in steps))
        for p in range(4):
            self.assertEqual(len(trajectory[p]), len(rewards[p]))

    @unittest.skipUnless(importlib.util.find_spec('a3dizhu_cpp'), 'Optional native extension')
    def test_native_rule_features_and_training(self):
        from a3dizhu_cpp import CppEngine, VectorizedEngine
        single, vector = CppEngine(), VectorizedEngine(1)
        for engine in (single, vector):
            engine.seed(42)
            engine.set_greedy_ratio(1.)
        single.reset()
        vector.reset(0)
        done, observations, players, actions = vector.advance_to_decision_with_actions(0)
        self.assertTrue(done)
        for obs, p, action in zip(observations, players, actions):
            self.assertEqual(p, single.get_player_id())
            np.testing.assert_array_equal(obs, single.encode_obs(p))
            key = single.get_rule_agent_action()
            np.testing.assert_array_equal(action, single.get_action_feature(key))
            single.step(key)
        self.assertTrue(single.is_over())
        for p in range(4):
            np.testing.assert_allclose(vector.get_step_rewards(0, p), single.get_step_rewards(p))
        from rlcard.envs.a3dizhu.dmc import NativePool
        env = rlcard.make('a3dizhu', config={'greedy_ratio': 1.})
        pool = NativePool(env, 2, 42, 1000)
        model = DMCModel(env.state_shape, env.action_shape, [8], device='cpu')
        episodes, _ = pool.round(model, .1, 64, contextlib.nullcontext())
        self.assertTrue(episodes)
        for steps, _, rewards, _ in episodes:
            self.assertTrue(any(np.any(a) for role in steps for _, a in role))
            self.assertEqual([len(s) for s in steps], list(map(len, rewards)))
        with tempfile.TemporaryDirectory() as path:
            trainer = DMCTrainer(rlcard.make('a3dizhu', config={'greedy_ratio': .5}),
                savedir=path, xpid='native', num_actors=1, envs_per_actor=3,
                batch_size=2, unroll_length=2, num_buffers=4, total_frames=16,
                mlp_layers=[8], backend='cpp', stats_interval=2, actor_timeout=45)
            self.assertEqual(trainer.start()['frames'], 16)

    @unittest.skipUnless(importlib.util.find_spec('pettingzoo'), 'Optional PettingZoo')
    def test_pettingzoo_aec_training(self):
        from pettingzoo.classic import tictactoe_v3
        with tempfile.TemporaryDirectory() as path:
            env = tictactoe_v3.env()
            trainer = DMCTrainer(env, is_pettingzoo_env=True, savedir=path, xpid='aec',
                num_actors=1, envs_per_actor=2, batch_size=2, unroll_length=2,
                num_buffers=4, total_frames=16, mlp_layers=[8], actor_timeout=45)
            self.assertEqual(trainer.start()['frames'], 16)
            env.close()

    def test_spawn_training_checkpoint_resume_and_shutdown(self):
        with tempfile.TemporaryDirectory() as path:
            params = dict(savedir=path, xpid='smoke', num_actors=1, envs_per_actor=3,
                          batch_size=2, unroll_length=2, num_buffers=4,
                          total_frames=16, mlp_layers=[16], backend='python',
                          initial_epsilon=.2, final_epsilon=.01, stats_interval=2,
                          weight_sync_interval=2, actor_timeout=45, share_weights=True)
            trainer = DMCTrainer(rlcard.make('leduc-holdem'), **params)
            stats = trainer.start()
            self.assertEqual(stats['frames'], 16)
            self.assertGreater(stats['actor_steps'], 0)
            self.assertAlmostEqual(stats['epsilon'], .01)
            self.assertAlmostEqual(stats['lr'], trainer.min_lr)
            self.assertTrue(all(not p.is_alive() for p in trainer.actor_processes))
            saved = torch.load(trainer.checkpointpath, weights_only=False)
            self.assertEqual(saved['scheduler_state_dicts'][0]['last_epoch'], 4)
            params.update(total_frames=24, load_model=True)
            resumed = DMCTrainer(rlcard.make('leduc-holdem'), **params)
            self.assertEqual(resumed.start()['frames'], 24)
            self.assertTrue(all(not p.is_alive() for p in resumed.actor_processes))

    def test_actor_error_reaches_parent_and_exits(self):
        with tempfile.TemporaryDirectory() as path:
            trainer = DMCTrainer(BrokenEnv(), savedir=path, xpid='broken', num_actors=1,
                                 envs_per_actor=1, batch_size=1, unroll_length=1,
                                 num_buffers=2, total_frames=2, mlp_layers=[8], actor_timeout=30)
            with self.assertRaises(RuntimeError):
                trainer.start()
            self.assertTrue(all(not p.is_alive() for p in trainer.actor_processes))


if __name__ == '__main__':
    unittest.main()
