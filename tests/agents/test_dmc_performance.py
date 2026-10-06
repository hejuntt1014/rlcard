"""Correctness boundaries of optional DMC performance paths."""
from collections import deque
import copy
import queue
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import rlcard
from rlcard.agents.dmc_agent.model import DMCModel
from rlcard.agents.dmc_agent.trainer import DMCTrainer, learn
from rlcard.agents.dmc_agent.utils import BatchReader, TrajectoryWriter, ColumnTrajectoryWriter, create_buffers


class TestPerformancePaths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_batched_writer_preserves_partial_slot_order_and_dtypes(self):
        buffers = create_buffers(3, 3, [[2]], [[2]], ['cpu'], aux_size=1)['cpu']
        free, full = [queue.Queue()], [queue.Queue()]
        for index in range(3):
            free[0].put(index)
        writer = TrajectoryWriter(3, free, full, buffers)
        expected_targets, expected_returns, expected_done = [], [], []
        offset = 0
        for length, payoff in ((7, 2.), (11, -1.)):
            trajectory = [(np.array([i + .25, -.5]), np.array([1., 0.]), np.array([i % 3]))
                          for i in range(offset, offset + length)]
            rewards = [.125] * length
            writer.add_episode([trajectory], [payoff], [rewards])
            expected_targets.extend(payoff + .125 * (length - j) for j in range(length))
            expected_returns.extend([0.] * (length - 1) + [payoff + .125 * length])
            expected_done.extend([False] * (length - 1) + [True])
            offset += length
        gathered = {key: [] for key in buffers[0]}
        for _ in range(2):
            writer.flush()
            self.assertEqual(full[0].qsize(), 3)
            for _ in range(3):
                index = full[0].get_nowait()
                for key, value in buffers[0].items():
                    gathered[key].append(value[index].clone())
                free[0].put(index)
        actual = {key: torch.cat(values) for key, values in gathered.items()}
        torch.testing.assert_close(actual['state'][:, 0], torch.arange(18) + .25)
        torch.testing.assert_close(actual['target'], torch.tensor(expected_targets))
        torch.testing.assert_close(actual['episode_return'], torch.tensor(expected_returns))
        torch.testing.assert_close(actual['done'], torch.tensor(expected_done))
        torch.testing.assert_close(actual['aux_target'][:, 0], torch.arange(18) % 3)
        self.assertFalse(writer.pending[0]['target'])

    def test_column_writer_crosses_episodes_and_noncontiguous_slots(self):
        for aux_size in (0, 1):
            with self.subTest(aux_size=aux_size):
                buffers = create_buffers(3, 3, [[2]], [[2]], ['cpu'], aux_size=aux_size)['cpu']
                free, full = [queue.Queue()], [queue.Queue()]
                for index in (2, 0, 1):
                    free[0].put(index)
                writer = ColumnTrajectoryWriter(3, free, full, buffers)
                writer.add_episode([{'target': np.empty(0, dtype=np.float32)}])
                offset = 0
                for length in (7, 11):
                    ids = np.arange(offset, offset + length, dtype=np.float32)
                    writer.add_episode([dict(
                        state=np.stack((ids + .25, ids - .5), axis=1),
                        action=np.stack((ids % 2, 1 - ids % 2), axis=1),
                        aux_target=(ids.astype(np.int64) % 3)[:, None],
                        target=ids, done=np.arange(length) == length - 1,
                        episode_return=np.where(np.arange(length) == length - 1, length, 0.),
                    )])
                    offset += length
                self.assertTrue(writer.congested())
                gathered = {key: [] for key in buffers[0]}
                for _ in range(2):
                    writer.flush()
                    for expected_index in (2, 0, 1):
                        index = full[0].get_nowait()
                        self.assertEqual(index, expected_index)
                        for key, value in buffers[0].items():
                            gathered[key].append(value[index].clone())
                        free[0].put(index)
                result = {key: torch.cat(value) for key, value in gathered.items()}
                torch.testing.assert_close(result['state'][:, 0], torch.arange(18) + .25)
                torch.testing.assert_close(result['target'], torch.arange(18, dtype=torch.float32))
                self.assertEqual(result['done'].nonzero().flatten().tolist(), [6, 17])
                self.assertEqual(result['episode_return'][[6, 17]].tolist(), [7., 11.])
                if aux_size:
                    torch.testing.assert_close(result['aux_target'][:, 0], torch.arange(18) % 3)
                self.assertEqual(writer.pending_rows, [0])
                self.assertFalse(writer.congested())

    def test_dense_training_preserves_inactive_head_gradients(self):
        for phase in ('play', 'declare', 'mixed'):
            with self.subTest(phase=phase):
                torch.manual_seed(42)
                model = DMCModel([[2668]], [[111]], [16, 16], device='cpu',
                                 architecture='context', aux_classes=(3, 3, 3, 4, 4))
                sparse = model.get_agent(0)
                dense = copy.deepcopy(sparse)
                state = torch.zeros(2, 2, 2668)
                if phase == 'declare':
                    state[..., 525] = 1
                elif phase == 'mixed':
                    state[0, :, 525] = 1
                action = torch.zeros(2, 2, 111)
                action[0, :, :52] = 1
                batch = dict(state=state, action=action, target=torch.ones(2, 2),
                             done=torch.zeros(2, 2, dtype=torch.bool), episode_return=torch.zeros(2, 2),
                             aux_target=torch.full((2, 2, 5), -1))
                results = []
                for agent, use_dense in ((sparse, False), (dense, True)):
                    results.append(learn(0, {}, agent, batch,
                        torch.optim.RMSprop(agent.parameters(), lr=.0001, eps=.00001), 'cpu', 40, [deque()],
                        sync_weights=False, dense_learner=use_dense,
                        learner_forward=agent.net.forward_with_aux_dense if use_dense else None))
                torch.testing.assert_close(results[0], results[1])
                for (name, old), (_, new) in zip(sparse.net.named_parameters(), dense.net.named_parameters()):
                    self.assertEqual(old.grad is None, new.grad is None, name)
                    if old.grad is not None:
                        torch.testing.assert_close(old.grad, new.grad, rtol=1e-4, atol=1e-6)
                    torch.testing.assert_close(old, new, rtol=1e-4, atol=1e-6)
                if phase == 'declare':
                    self.assertIsNone(dense.net.output_head.weight.grad)
                elif phase == 'play':
                    self.assertIsNone(dense.net.declare_head.weight.grad)

    @unittest.skipUnless(hasattr(torch, 'compile'), 'torch.compile unavailable')
    def test_compiled_trainer_keeps_portable_checkpoint_names(self):
        original_compile = torch.compile

        def eager_compile(function, **kwargs):
            return original_compile(function, backend='eager', fullgraph=True, dynamic=False)

        with tempfile.TemporaryDirectory() as folder:
            env = rlcard.make('leduc-holdem')
            trainer = DMCTrainer(env, savedir=folder, xpid='compiled', num_actors=1,
                                 envs_per_actor=1, batch_size=1, unroll_length=1,
                                 num_buffers=2, total_frames=2, mlp_layers=[8],
                                 compile_learner=True, actor_timeout=60)
            with patch('torch.compile', side_effect=eager_compile):
                trainer.start()
            checkpoint = torch.load(trainer.checkpointpath, map_location='cpu', weights_only=False)
            ordinary = trainer.model_func('cpu')
            for position, state in enumerate(checkpoint['model_state_dict']):
                self.assertFalse(any('_orig_mod' in key for key in state))
                self.assertEqual(set(state), set(ordinary.get_agent(position).state_dict()))
                ordinary.get_agent(position).load_state_dict(state)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA integration test')
class TestPinnedTransport(unittest.TestCase):
    def test_staging_reuse_waits_for_pending_device_reads(self):
        buffers = create_buffers(2, 2, [[2]], [[2]], ['cpu'])['cpu'][0]
        free, full = queue.Queue(), queue.Queue()
        reader = BatchReader(free, full, buffers, 2, batch_major=True, pin_memory=True)
        received = []
        addresses = []
        stream = torch.cuda.Stream(device=0)
        for value in (1., 2., 3.):
            for tensor in buffers.values():
                tensor.fill_(value)
            for index in range(2):
                full.put(index)
            batch = reader.get()
            self.assertTrue(batch['state'].is_pinned())
            addresses.append(batch['state'].data_ptr())
            for _ in range(2):
                free.get_nowait()
            with torch.cuda.stream(stream):
                # Keep the H2D read pending while the host attempts the next
                # gather, making an omitted transfer fence observable.
                if hasattr(torch.cuda, '_sleep'):
                    torch.cuda._sleep(5_000_000)
                received.append(batch['state'].to('cuda:0', non_blocking=True))
                reader.mark_transferred('cuda:0')
        stream.synchronize()
        self.assertEqual(len(set(addresses)), 1)
        for expected, result in enumerate(received, 1):
            torch.testing.assert_close(result.cpu(), torch.full((2, 2, 2), float(expected)))
