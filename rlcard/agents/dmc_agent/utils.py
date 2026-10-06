# Modifications copyright (c) 2026 hejuntt1014.
# Modified for batched training, optional backends, and runtime portability.
# Copyright 2021 RLCard Team of Texas A&M University
# Copyright 2021 DouZero Team of Kwai
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bounded CPU shared-memory transport for DMC trajectories."""
import logging
from collections import deque
from queue import Empty

import numpy as np
import torch

log = logging.getLogger(__name__)


def create_buffers(T, num_buffers, state_shape, action_shape, device_iterator,
                   aux_size=0, dtype=torch.float32):
    buffers = {}
    for device in device_iterator:
        buffers[device] = []
        for obs_shape, act_shape in zip(state_shape, action_shape):
            specs = {
                'done': ((T,), torch.bool),
                'episode_return': ((T,), torch.float32),
                'target': ((T,), torch.float32),
                'state': ((T,) + tuple(obs_shape), dtype),
                'action': ((T,) + tuple(act_shape), dtype),
            }
            if aux_size:
                specs['aux_target'] = ((T, aux_size), torch.int64)
            buffers[device].append({
                key: torch.empty((num_buffers,) + shape, dtype=kind).share_memory_()
                for key, (shape, kind) in specs.items()
            })
    return buffers


class BatchReader:
    """One consumer; hold partial batches locally instead of requeueing them."""
    def __init__(self, free_queue, full_queue, buffers, batch_size, batch_major=False,
                 pin_memory=False):
        self.free_queue = free_queue
        self.full_queue = full_queue
        self.buffers = buffers
        self.batch_size = batch_size
        self.pending = []
        self.batch_major = batch_major
        self.pin_memory = pin_memory
        self.staging = None
        self.transfer_complete = None

    def mark_transferred(self, device):
        """Protect reusable host storage until all queued H2D reads finish."""
        if self.pin_memory:
            if self.transfer_complete is None:
                self.transfer_complete = torch.cuda.Event()
            self.transfer_complete.record(torch.cuda.current_stream(device))

    def get(self):
        while len(self.pending) < self.batch_size:
            try:
                self.pending.append(self.full_queue.get_nowait())
            except Empty:
                return None
        indices = torch.tensor(self.pending, dtype=torch.long)
        if self.pin_memory and self.batch_major:
            if self.staging is None:
                self.staging = {
                    key: torch.empty((self.batch_size,) + tuple(value.shape[1:]),
                                     dtype=value.dtype, pin_memory=True)
                    for key, value in self.buffers.items()
                }
            if self.transfer_complete is not None:
                self.transfer_complete.synchronize()
            # Gather directly into pinned storage, avoiding an intermediate
            # pageable batch and allowing asynchronous H2D transfers.
            for key, value in self.buffers.items():
                torch.index_select(value, 0, indices, out=self.staging[key])
            batch = self.staging
        else:
            batch = {key: value.index_select(0, indices) for key, value in self.buffers.items()}
        if not self.batch_major:
            batch = {key: value.transpose(0, 1).contiguous() for key, value in batch.items()}
        for index in self.pending:
            self.free_queue.put(index)
        self.pending = []
        return batch


def _compute_targets_with_step_rewards(payoff, step_rewards):
    total = float(payoff)
    targets = [0.] * len(step_rewards)
    for i in range(len(step_rewards) - 1, -1, -1):
        total += float(step_rewards[i])
        targets[i] = total
    return targets


class TrajectoryWriter:
    """Flush whole unrolls without blocking one role behind another."""
    def __init__(self, T, free_queues, full_queues, buffers):
        self.T = T
        self.free_queues = free_queues
        self.full_queues = full_queues
        self.buffers = buffers
        self.numpy_buffers = [{key: value.numpy() for key, value in fields.items()}
                              for fields in buffers]
        self.pending = [{key: deque() for key in fields} for fields in buffers]

    def add_episode(self, steps, payoffs, rewards=None, auxiliary=None):
        for p, trajectory in enumerate(steps):
            n = len(trajectory)
            if not n:
                continue
            sr = [0.] * n if rewards is None else rewards[p]
            if len(sr) != n:
                raise ValueError('Step rewards and recorded actions must have identical lengths')
            out = self.pending[p]
            out['state'].extend(s[0] for s in trajectory)
            out['action'].extend(s[1] for s in trajectory)
            out['target'].extend(_compute_targets_with_step_rewards(payoffs[p], sr))
            out['done'].extend([False] * (n - 1) + [True])
            out['episode_return'].extend([0.] * (n - 1) + [float(payoffs[p]) + sum(sr)])
            if 'aux_target' in out:
                if all(len(step) >= 3 for step in trajectory):
                    out['aux_target'].extend(step[2] for step in trajectory)
                elif auxiliary is None:
                    raise ValueError('Auxiliary task enabled but episode has no labels')
                else:
                    out['aux_target'].extend([auxiliary[p]] * n)

    def flush(self):
        for p, data in enumerate(self.pending):
            while len(data['target']) >= self.T:
                # Batch a bounded number of complete unrolls while retaining
                # the existing one-index-per-slot queue ownership protocol.
                count = min(len(data['target']) // self.T, 16, max(1, 320 // self.T))
                indices = []
                for _ in range(count):
                    try:
                        indices.append(self.free_queues[p].get_nowait())
                    except Empty:
                        break
                if not indices:
                    break
                rows = len(indices) * self.T
                for key, values in data.items():
                    output = self.numpy_buffers[p][key]
                    block = np.asarray([values.popleft() for _ in range(rows)], dtype=output.dtype)
                    output[indices] = block.reshape(len(indices), *output.shape[1:])
                for index in indices:
                    self.full_queues[p].put(index)

    def congested(self):
        return any(len(data['target']) >= self.T * 2 for data in self.pending)


class ColumnTrajectoryWriter:
    """Publish complete columnar episodes without creating Python row records.

    ``add_episode`` takes one dictionary of NumPy arrays per role. Arrays are
    retained until consumed; their owner must not mutate them after publication.
    Native capsule-backed arrays keep their storage alive independently of resets.
    """
    def __init__(self, T, free_queues, full_queues, buffers):
        self.T = T
        self.free_queues, self.full_queues = free_queues, full_queues
        self.buffers = [{key: value.numpy() for key, value in fields.items()}
                        for fields in buffers]
        self.pending = [deque() for _ in buffers]
        self.pending_rows = [0] * len(buffers)

    def add_episode(self, columns):
        if len(columns) != len(self.buffers):
            raise ValueError('Column episodes must contain one entry per role')
        for p, data in enumerate(columns):
            count = len(data['target'])
            if not count:
                continue
            block = {}
            for key, destination in self.buffers[p].items():
                source = np.asarray(data[key], dtype=destination.dtype)
                if source.shape != (count,) + destination.shape[2:]:
                    raise ValueError('Column episode field has an incompatible shape: ' + key)
                block[key] = source
            self.pending[p].append([block, 0])
            self.pending_rows[p] += count

    def _copy_rows(self, position, destinations, count):
        written = 0
        while written < count:
            entry = self.pending[position][0]
            columns, offset = entry
            take = min(count - written, len(columns['target']) - offset)
            for key, destination in destinations.items():
                destination[written:written + take] = columns[key][offset:offset + take]
            written += take
            offset += take
            if offset == len(columns['target']):
                self.pending[position].popleft()
            else:
                entry[1] = offset

    def flush(self):
        for p, data in enumerate(self.buffers):
            while self.pending_rows[p] >= self.T:
                count = min(self.pending_rows[p] // self.T, 16)
                indices = []
                for _ in range(count):
                    try:
                        indices.append(self.free_queues[p].get_nowait())
                    except Empty:
                        break
                if not indices:
                    break
                # Consecutive shared slots form writable contiguous views. Copy
                # entire episode slices across these views without staging or
                # advanced-indexing copies; arbitrary slot order remains valid.
                start = 0
                while start < len(indices):
                    end = start + 1
                    while end < len(indices) and indices[end] == indices[end - 1] + 1:
                        end += 1
                    first, stop = indices[start], indices[end - 1] + 1
                    rows = (end - start) * self.T
                    destinations = {key: value[first:stop].reshape((rows,) + value.shape[2:])
                                    for key, value in data.items()}
                    self._copy_rows(p, destinations, rows)
                    start = end
                self.pending_rows[p] -= len(indices) * self.T
                for index in indices:
                    self.full_queues[p].put(index)

    def congested(self):
        return any(count >= self.T * 2 for count in self.pending_rows)


def create_optimizers(num_players, learning_rate, momentum, epsilon, alpha, learner_model):
    unique = {}
    optimizers = []
    for p in range(num_players):
        agent = learner_model.get_agent(p)
        if id(agent) not in unique:
            unique[id(agent)] = torch.optim.RMSprop(
                agent.parameters(), lr=learning_rate, momentum=momentum, eps=epsilon, alpha=alpha)
        optimizers.append(unique[id(agent)])
    return optimizers
