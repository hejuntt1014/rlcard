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

"""DMC training with per-role batching and optional native environment backends."""
import contextlib
import importlib.util
import math
import os
import time
from collections import deque
from queue import Empty

import numpy as np
import torch
from torch import multiprocessing as mp
from torch import nn
import torch.nn.functional as F

from .collector import RLCardAdapter, PettingZooAdapter, actor_worker, environment_source
from .file_writer import FileWriter
from .model import DMCModel
from .utils import BatchReader, create_buffers, create_optimizers, log


def compute_loss(logits, targets):
    return ((logits - targets) ** 2).mean()


class FrameScheduler:
    """Cosine learning rate as a function of consumed training samples."""
    def __init__(self, optimizer, learning_rate, min_lr, total_frames, batch_frames):
        self.optimizer = optimizer
        self.learning_rate, self.min_lr = learning_rate, min_lr
        self.total_frames, self.batch_frames = total_frames, batch_frames
        self.frames = self.last_epoch = 0

    def step(self, frames):
        self.frames = frames
        self.last_epoch = frames // self.batch_frames
        progress = min(frames / self.total_frames, 1.)
        rate = self.min_lr + .5 * (self.learning_rate - self.min_lr) * (1 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group['lr'] = rate

    def state_dict(self):
        return dict(frames=self.frames, last_epoch=self.last_epoch)

    def load_state_dict(self, state):
        self.step(state['frames'])


def auxiliary_loss(logits, labels, classes):
    """Masked categorical tasks without host-side tests on CUDA tensors."""
    offset = 0
    losses, active = [], []
    for i, width in enumerate(classes):
        valid = labels[:, i] >= 0
        safe_labels = labels[:, i].clamp(min=0)
        loss = F.cross_entropy(logits[:, offset:offset + width], safe_labels, reduction='none')
        count = valid.sum()
        losses.append((loss * valid).sum() / count.clamp(min=1))
        active.append(count > 0)
        offset += width
    return torch.stack(losses).sum() / torch.stack(active).sum().clamp(min=1)


def learn(position, actor_models, agent, batch, optimizer, training_device,
          max_grad_norm, mean_episode_return_buf, lock=None, sync_weights=True,
          aux_weight=.1):
    device = 'cuda:' + str(training_device) if training_device != 'cpu' else 'cpu'
    state = batch['state'].to(device).flatten(0, 1).float()
    action = batch['action'].to(device).flatten(0, 1).float()
    target = batch['target'].to(device).flatten(0, 1)
    returns = batch['episode_return'][batch['done']]
    if returns.numel():
        mean_episode_return_buf[position].append(float(returns.mean()))
    with lock if lock is not None else contextlib.nullcontext():
        values, aux = agent.forward_with_aux(state, action)
        loss = compute_loss(values, target)
        if aux is not None and 'aux_target' in batch:
            labels = batch['aux_target'].to(device).flatten(0, 1).long()
            loss = loss + aux_weight * auxiliary_loss(aux, labels, agent.net.aux_classes)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()
        if sync_weights:
            for model in actor_models.values():
                model.get_agent(position).load_state_dict(agent.state_dict())
    return loss.detach()


class DMCTrainer:
    """Batched DMC for RLCard and PettingZoo AEC environments.

    Independent role policies are the default. Weight sharing is opt-in and
    requires equal shapes and compatible role semantics. `backend='auto'` uses
    native A3 batching when available and otherwise uses Python environments.
    `vectorized=False` selects one environment per actor.
    """
    def __init__(self, env, cuda='', is_pettingzoo_env=False, load_model=False,
                 xpid='dmc', save_interval=30, num_actor_devices=1, num_actors=5,
                 training_device='0', savedir='experiments/dmc_result',
                 total_frames=100000000000, exp_epsilon=.01, batch_size=32,
                 unroll_length=100, num_buffers=50, num_threads=4,
                 max_grad_norm=40, learning_rate=.0001, alpha=.99, momentum=0,
                 epsilon=.00001, share_weights=False, initial_epsilon=None,
                 final_epsilon=None, epsilon_decay_ratio=.8, min_lr=1e-6,
                 vectorized=True, envs_per_actor=8, backend='auto', seed=0,
                 architecture=None, mlp_layers=None, auxiliary=None,
                 weight_sync_interval=50, stats_interval=50,
                 max_inference_actions=4096, max_episode_steps=10000,
                 actor_timeout=120, adapter_class=None, env_factory=None, actor_on_cpu=False):
        positive = dict(batch_size=batch_size, unroll_length=unroll_length,
                        num_buffers=num_buffers, num_actors=num_actors,
                        num_actor_devices=num_actor_devices, envs_per_actor=envs_per_actor,
                        weight_sync_interval=weight_sync_interval, stats_interval=stats_interval,
                        max_inference_actions=max_inference_actions, max_episode_steps=max_episode_steps)
        if any(not isinstance(v, int) or v <= 0 for v in positive.values()):
            raise ValueError('Counts must be positive integers: ' + str(positive))
        if num_buffers < batch_size:
            raise ValueError('num_buffers must be >= batch_size')
        if total_frames <= 0 or actor_timeout <= 0 or save_interval <= 0:
            raise ValueError('total_frames, actor_timeout and save_interval must be positive')
        if not 0 <= seed < 2**32 or not 0 < epsilon_decay_ratio <= 1:
            raise ValueError('Invalid seed or epsilon_decay_ratio')
        initial_epsilon = exp_epsilon if initial_epsilon is None else initial_epsilon
        final_epsilon = initial_epsilon if final_epsilon is None else final_epsilon
        if not (0 <= initial_epsilon <= 1 and 0 <= final_epsilon <= 1):
            raise ValueError('Exploration probabilities must be in [0, 1]')
        if not 0 <= min_lr <= learning_rate or learning_rate <= 0:
            raise ValueError('Require 0 <= min_lr <= learning_rate and learning_rate > 0')
        if backend not in ('auto', 'python', 'cpp'):
            raise ValueError('backend must be auto, python or cpp')
        self.env = env
        self.env_source = env_factory or environment_source(env)
        self.is_pettingzoo_env = is_pettingzoo_env
        is_a3 = not is_pettingzoo_env and getattr(env, 'name', '') == 'a3dizhu'
        if is_pettingzoo_env:
            env.reset(seed=seed)
            self.num_players = len(env.possible_agents)
            self.state_shape = [list(env.observation_space(p)['observation'].shape)
                                for p in env.possible_agents]
            self.action_shape = [[env.action_space(p).n] for p in env.possible_agents]
            default_adapter = PettingZooAdapter
        else:
            self.num_players = env.num_players
            self.state_shape = env.state_shape
            self.action_shape = [list(s) if s is not None else [env.num_actions]
                                 for s in env.action_shape]
            default_adapter = RLCardAdapter
            if is_a3:
                from rlcard.envs.a3dizhu.dmc import A3Adapter
                default_adapter = A3Adapter
        self.adapter_class = adapter_class or default_adapter
        self.state_shape = [[int(n) for n in s] for s in self.state_shape]
        self.action_shape = [[int(n) for n in s] for s in self.action_shape]
        native = False
        if is_a3 and importlib.util.find_spec('a3dizhu_cpp') is not None:
            from a3dizhu_cpp import VectorizedEngine
            native = hasattr(VectorizedEngine, 'advance_to_decision_with_actions')
        if backend == 'cpp' and (not native or adapter_class is not None):
            raise ValueError('Native A3 backend unavailable: build the current C++ extension')
        self.backend = ('cpp' if native and adapter_class is None else 'python') if backend == 'auto' else backend
        self.architecture = architecture or ('resnet' if is_a3 else 'mlp')
        self.aux_classes = (3, 3, 3) if (is_a3 if auxiliary is None else auxiliary) else ()
        if self.aux_classes and not is_a3:
            raise ValueError('Built-in auxiliary labels are only available for A3')
        self.mlp_layers = list(mlp_layers or [512] * 5)
        self.share_weights = share_weights
        if share_weights and (any(tuple(s) != tuple(self.state_shape[0]) for s in self.state_shape)
                              or any(tuple(s) != tuple(self.action_shape[0]) for s in self.action_shape)):
            raise ValueError('Shared policies require identical observation and action shapes')
        if self.architecture == 'mlp' and self.aux_classes:
            raise ValueError('Use auxiliary=False with an MLP')
        if cuda:
            os.environ['CUDA_VISIBLE_DEVICES'] = cuda
            if not torch.cuda.is_available():
                raise ValueError('CUDA requested but unavailable')
            if not actor_on_cpu and num_actor_devices > torch.cuda.device_count():
                raise ValueError('num_actor_devices exceeds visible CUDA devices')
            if training_device != 'cpu' and not 0 <= int(training_device) < torch.cuda.device_count():
                raise ValueError('training_device is not a visible CUDA device')
            self.device_iterator = ['cpu'] if actor_on_cpu else list(range(num_actor_devices))
            self.training_device = str(training_device)
        else:
            self.device_iterator = ['cpu']
            self.training_device = 'cpu'
        self.T, self.B = unroll_length, batch_size
        self.num_buffers, self.num_actors = num_buffers, num_actors
        self.envs_per_actor = envs_per_actor if vectorized else 1
        self.total_frames, self.seed = total_frames, seed
        self.initial_epsilon, self.final_epsilon = initial_epsilon, final_epsilon
        self.epsilon_decay_frames = max(1, int(total_frames * epsilon_decay_ratio))
        self.learning_rate, self.min_lr = learning_rate, min_lr
        self.alpha, self.momentum, self.epsilon = alpha, momentum, epsilon
        self.max_grad_norm = max_grad_norm
        self.weight_sync_interval, self.stats_interval = weight_sync_interval, stats_interval
        self.max_inference_actions, self.max_episode_steps = max_inference_actions, max_episode_steps
        self.actor_timeout = actor_timeout
        self.load_model, self.save_interval = load_model, save_interval
        self.savedir, self.xpid = savedir, xpid
        self.checkpointpath = os.path.join(os.path.expanduser(savedir), xpid, 'model.tar')
        self.storage_dtype = torch.int8 if is_a3 else torch.float32
        self.frames = 0
        self.stats = {key + str(p): 0. for p in range(self.num_players)
                      for key in ('loss_', 'mean_episode_return_')}
        self.actor_processes = []
        self.mean_episode_return_buf = [deque(maxlen=100) for _ in range(self.num_players)]

    def model_func(self, device):
        return DMCModel(self.state_shape, self.action_shape, self.mlp_layers,
                        self.initial_epsilon, str(device), self.share_weights,
                        self.architecture, self.aux_classes)

    def _get_epsilon(self, frames):
        fraction = min(frames / self.epsilon_decay_frames, 1.)
        return self.initial_epsilon + fraction * (self.final_epsilon - self.initial_epsilon)

    def _model_spec(self):
        return dict(state_shape=[list(s) for s in self.state_shape],
                    action_shape=self.action_shape, mlp_layers=self.mlp_layers,
                    architecture=self.architecture, aux_classes=list(self.aux_classes),
                    share_weights=self.share_weights)

    def start(self):
        torch.manual_seed(self.seed)
        learner = self.model_func(self.training_device)
        optimizers = create_optimizers(self.num_players, self.learning_rate, self.momentum,
                                      self.epsilon, self.alpha, learner)
        unique_opts = list({id(o): o for o in optimizers}.values())
        schedulers = [FrameScheduler(o, self.learning_rate, self.min_lr,
                                     self.total_frames, self.T * self.B) for o in unique_opts]
        self.frames = 0
        if self.load_model:
            if not os.path.isfile(self.checkpointpath):
                raise FileNotFoundError(self.checkpointpath)
            state = torch.load(self.checkpointpath, map_location='cpu', weights_only=False)
            if 'model_spec' in state and state['model_spec'] != self._model_spec():
                raise ValueError('Checkpoint model specification does not match this run')
            if bool(state.get('share_weights', False)) != self.share_weights:
                raise ValueError('Checkpoint weight sharing does not match this run')
            if len(state['model_state_dict']) != self.num_players:
                raise ValueError('Checkpoint role count does not match this run')
            for p in range(self.num_players):
                learner.get_agent(p).load_state_dict(state['model_state_dict'][p])
                optimizers[p].load_state_dict(state['optimizer_state_dict'][p])
            self.frames = state['frames']
            self.stats = state.get('stats', {})
            for scheduler, saved in zip(schedulers, state.get('scheduler_state_dicts', [])):
                scheduler.load_state_dict(saved)
            for scheduler in schedulers:
                scheduler.step(self.frames)
        ctx = mp.get_context('spawn')
        stop = ctx.Event()
        exploration = ctx.Value('d', self._get_epsilon(self.frames))
        errors = ctx.Queue()
        buffers = create_buffers(self.T, self.num_buffers, self.state_shape, self.action_shape,
                                 self.device_iterator, len(self.aux_classes), self.storage_dtype)
        models, locks, free, full = {}, {}, {}, {}
        readers, counters = [], []
        for device in self.device_iterator:
            models[device] = self.model_func(device)
            models[device].share_memory()
            models[device].eval()
            locks[device] = ctx.Lock()
            free[device] = [ctx.Queue(maxsize=self.num_buffers) for _ in range(self.num_players)]
            full[device] = [ctx.Queue(maxsize=self.num_buffers) for _ in range(self.num_players)]
            for p in range(self.num_players):
                models[device].get_agent(p).load_state_dict(learner.get_agent(p).state_dict())
                for i in range(self.num_buffers):
                    free[device][p].put(i)
                readers.append((device, p, BatchReader(free[device][p], full[device][p],
                                                       buffers[device][p], self.B)))
            if device != 'cpu':
                torch.cuda.synchronize(device)
        self.plogger = FileWriter(self.xpid, xp_args=dict(
            model=self._model_spec(), backend=self.backend, seed=self.seed,
            batch_size=self.B, unroll_length=self.T, envs_per_actor=self.envs_per_actor,
            num_actors=self.num_actors, actor_devices=self.device_iterator,
            training_device=self.training_device, weight_sync_interval=self.weight_sync_interval),
            rootdir=self.savedir)
        self.actor_processes = []
        updates = self.frames // (self.T * self.B)
        successful = False
        last_checkpoint = last_log = last_batch = time.monotonic()
        start_time, start_frames = last_log, self.frames
        loss_totals = [[] for _ in range(self.num_players)]

        def sync_models():
            for device, model in models.items():
                with locks[device]:
                    for p in ([0] if self.share_weights else range(self.num_players)):
                        model.get_agent(p).load_state_dict(learner.get_agent(p).state_dict())
                    if device != 'cpu':
                        torch.cuda.synchronize(device)

        def checkpoint():
            payload = dict(model_state_dict=[a.state_dict() for a in learner.get_agents()],
                           optimizer_state_dict=[o.state_dict() for o in optimizers],
                           scheduler_state_dicts=[s.state_dict() for s in schedulers],
                           frames=self.frames, stats=self.stats, share_weights=self.share_weights,
                           model_spec=self._model_spec(), seed=self.seed)
            temporary = self.checkpointpath + '.tmp'
            torch.save(payload, temporary)
            os.replace(temporary, self.checkpointpath)

        try:
            if self.frames < self.total_frames:
                for device in self.device_iterator:
                    for _ in range(self.num_actors):
                        actor_id = len(self.actor_processes)
                        actor_seed = int(np.random.SeedSequence([self.seed, actor_id]).generate_state(1)[0])
                        actor_seed %= 2**32 - self.envs_per_actor
                        counter = ctx.Array('q', 2)
                        counters.append(counter)
                        actor = ctx.Process(target=actor_worker, name='dmc-actor-%d' % actor_id,
                            args=(actor_id, actor_seed, self.env_source, models[device], locks[device],
                                  buffers[device], free[device], full[device], stop, exploration,
                                  errors, counter, self.T, self.envs_per_actor, self.backend,
                                  self.adapter_class, self.max_inference_actions, self.max_episode_steps))
                        actor.start()
                        self.actor_processes.append(actor)
            while self.frames < self.total_frames:
                try:
                    actor_id, detail = errors.get_nowait()
                    raise RuntimeError('Actor %d failed:\n%s' % (actor_id, detail))
                except Empty:
                    pass
                for actor in self.actor_processes:
                    if actor.exitcode is not None:
                        raise RuntimeError('%s exited unexpectedly (%s)' % (actor.name, actor.exitcode))
                trained = False
                for device, p, reader in readers:
                    if self.frames >= self.total_frames:
                        break
                    batch = reader.get()
                    if batch is None:
                        continue
                    loss = learn(p, {}, learner.get_agent(p), batch, optimizers[p],
                                 self.training_device, self.max_grad_norm,
                                 self.mean_episode_return_buf, sync_weights=False)
                    loss_totals[p].append(loss)
                    self.frames += self.T * self.B
                    updates += 1
                    for scheduler in schedulers:
                        scheduler.step(self.frames)
                    exploration.value = self._get_epsilon(self.frames)
                    if updates % self.weight_sync_interval == 0:
                        sync_models()
                    trained = True
                    last_batch = time.monotonic()
                    if updates % self.stats_interval == 0:
                        self._record_stats(loss_totals, counters, updates, exploration.value,
                                           optimizers[0].param_groups[0]['lr'], start_time, start_frames)
                if not trained:
                    stop.wait(.002)
                now = time.monotonic()
                if now - last_batch > self.actor_timeout:
                    raise TimeoutError('No training batch within actor_timeout; check actor supply and batch size')
                if now - last_checkpoint >= self.save_interval * 60:
                    checkpoint()
                    last_checkpoint = now
                if now - last_log >= 5:
                    log.info('Trained %d samples (%.0f samples/s)', self.frames,
                             (self.frames - start_frames) / (now - start_time))
                    last_log = now
            successful = True
        except KeyboardInterrupt:
            log.info('Stopping training and saving checkpoint')
        finally:
            stop.set()
            deadline = time.monotonic() + 5
            for actor in self.actor_processes:
                actor.join(timeout=max(0, deadline - time.monotonic()))
            for actor in self.actor_processes:
                if actor.is_alive():
                    actor.terminate()
                    actor.join(timeout=5)
            self._record_stats(loss_totals, counters, updates, exploration.value,
                               optimizers[0].param_groups[0]['lr'], start_time, start_frames)
            checkpoint()
            for queues in (free, full):
                for group in queues.values():
                    for q in group:
                        q.cancel_join_thread()
                        q.close()
            errors.close()
            self.plogger.close(successful=successful)
        return self.stats

    def _record_stats(self, losses, counters, updates, exploration, lr, start_time, start_frames):
        self.stats.update(frames=self.frames, learner_updates=updates, epsilon=exploration, lr=lr,
                          training_samples_per_second=(self.frames - start_frames) /
                          max(time.monotonic() - start_time, 1e-9),
                          actor_steps=sum(c[0] for c in counters),
                          episodes=sum(c[1] for c in counters))
        for p in range(self.num_players):
            if losses[p]:
                self.stats['loss_%d' % p] = torch.stack(losses[p]).mean().item()
                losses[p].clear()
            returns = self.mean_episode_return_buf[p]
            if returns:
                self.stats['mean_episode_return_%d' % p] = float(np.mean(returns))
        self.plogger.log(dict(self.stats))
