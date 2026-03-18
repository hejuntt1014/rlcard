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

import os
import threading
import time
import timeit
import pprint
import math
from collections import deque

import torch
from torch import multiprocessing as mp
from torch import nn

from .file_writer import FileWriter
from .model import DMCModel
from .pettingzoo_model import DMCModelPettingZoo
from .utils import (
    get_batch,
    create_buffers,
    create_optimizers,
    act,
    log,
)
from .pettingzoo_utils import (
    create_buffers_pettingzoo,
    act_pettingzoo,
)

import torch.nn.functional as F

AUX_LOSS_WEIGHT = 0.1

def compute_loss(logits, targets):
    loss = ((logits - targets)**2).mean()
    return loss

def learn(
    position,
    actor_models,
    agent,
    batch,
    optimizer,
    training_device,
    max_grad_norm,
    mean_episode_return_buf,
    lock
):
    """Performs a learning (optimization) step."""
    device = "cuda:"+str(training_device) if training_device != "cpu" else "cpu"
    state = torch.flatten(batch['state'].to(device), 0, 1).float()
    action = torch.flatten(batch['action'].to(device), 0, 1).float()
    target = torch.flatten(batch['target'].to(device), 0, 1)
    aux_target = torch.flatten(batch['aux_target'].to(device), 0, 1).long()
    episode_returns = batch['episode_return'][batch['done']]
    mean_episode_return_buf[position].append(torch.mean(episode_returns).to(device))

    with lock:
        values, aux_logits = agent.forward_with_aux(state, action)
        q_loss = compute_loss(values, target)

        aux_loss = torch.tensor(0.0, device=device)
        aux_count = 0
        for i in range(3):
            mask = aux_target[:, i] >= 0
            if mask.any():
                aux_loss += F.cross_entropy(
                    aux_logits[:, i*3:(i+1)*3][mask],
                    aux_target[:, i][mask],
                )
                aux_count += 1
        if aux_count > 0:
            aux_loss = aux_loss / aux_count

        loss = q_loss + AUX_LOSS_WEIGHT * aux_loss

        stats = {
            'mean_episode_return_'+str(position): torch.mean(torch.stack([_r for _r in mean_episode_return_buf[position]])).item(),
            'loss_'+str(position): loss.item(),
        }

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        optimizer.step()

        for actor_model in actor_models.values():
            actor_model.get_agent(position).load_state_dict(agent.state_dict())
        return stats


class DMCTrainer:
    """Deep Monte-Carlo Trainer（A3 地主增强版）

    相对原版 RLCard DMC 的改进：
      - 权重共享: 4 个位置共用 1 个网络（share_weights=True）
      - Epsilon 退火: 探索率从 initial_epsilon 线性衰减到 final_epsilon
      - 学习率调度: Cosine Annealing 到 min_lr
      - 残差网络: DMCNet 使用 ResBlock + LayerNorm
    """
    def __init__(
        self,
        env,
        cuda="",
        is_pettingzoo_env=False,
        load_model=False,
        xpid='dmc',
        save_interval=30,
        num_actor_devices=1,
        num_actors=5,
        training_device="0",
        savedir='experiments/dmc_result',
        total_frames=100000000000,
        exp_epsilon=0.01,
        batch_size=32,
        unroll_length=100,
        num_buffers=50,
        num_threads=4,
        max_grad_norm=40,
        learning_rate=0.0001,
        alpha=0.99,
        momentum=0,
        epsilon=0.00001,
        # ─── 新增参数 ─────────────────────────
        share_weights=True,
        initial_epsilon=0.1,
        final_epsilon=0.01,
        epsilon_decay_ratio=0.8,
        min_lr=1e-6,
    ):
        self.env = env

        self.plogger = FileWriter(
            xpid=xpid,
            rootdir=savedir,
        )

        self.checkpointpath = os.path.expandvars(
            os.path.expanduser('%s/%s/%s' % (savedir, xpid, 'model.tar')))

        self.T = unroll_length
        self.B = batch_size

        self.xpid = xpid
        self.load_model = load_model
        self.savedir = savedir
        self.save_interval = save_interval
        self.num_actor_devices = num_actor_devices
        self.num_actors = num_actors
        self.training_device = training_device
        self.total_frames = total_frames
        self.exp_epsilon = exp_epsilon
        self.num_buffers = num_buffers
        self.num_threads = num_threads
        self.max_grad_norm = max_grad_norm
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.momentum = momentum
        self.epsilon = epsilon

        self.share_weights = share_weights
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.epsilon_decay_frames = int(total_frames * epsilon_decay_ratio)
        self.min_lr = min_lr

        self.is_pettingzoo_env = is_pettingzoo_env
        if not self.is_pettingzoo_env:
            self.num_players = self.env.num_players
            self.action_shape = self.env.action_shape
            if self.action_shape[0] == None:
                self.action_shape = [[self.env.num_actions] for _ in range(self.num_players)]

            def model_func(device):
                return DMCModel(
                    self.env.state_shape,
                    self.action_shape,
                    exp_epsilon=self.initial_epsilon,
                    device=str(device),
                    share_weights=self.share_weights,
                )
        else:
            self.num_players = self.env.num_agents

            def model_func(device):
                return DMCModelPettingZoo(
                    self.env,
                    exp_epsilon=self.initial_epsilon,
                    device=device
                )
        self.model_func = model_func

        self.mean_episode_return_buf = [deque(maxlen=100) for _ in range(self.num_players)]

        if cuda == "":
            self.device_iterator = ['cpu']
            self.training_device = "cpu"
        else:
            self.device_iterator = range(num_actor_devices)

    def _get_epsilon(self, frames):
        """线性退火：initial_epsilon → final_epsilon"""
        if frames >= self.epsilon_decay_frames:
            return self.final_epsilon
        frac = frames / self.epsilon_decay_frames
        return self.initial_epsilon - (self.initial_epsilon - self.final_epsilon) * frac

    def _update_actor_epsilon(self, models, eps):
        """更新所有 actor 模型的探索率"""
        for model in models.values():
            seen = set()
            for agent in model.get_agents():
                if id(agent) not in seen:
                    agent.exp_epsilon = eps
                    seen.add(id(agent))

    def start(self):
        models = {}
        for device in self.device_iterator:
            model = self.model_func(device)
            model.share_memory()
            model.eval()
            models[device] = model

        if not self.is_pettingzoo_env:
            buffers = create_buffers(
                self.T, self.num_buffers, self.env.state_shape,
                self.action_shape, self.device_iterator,
            )
        else:
            buffers = create_buffers_pettingzoo(
                self.T, self.num_buffers, self.env, self.device_iterator,
            )

        actor_processes = []
        ctx = mp.get_context('spawn')
        free_queue = {}
        full_queue = {}
        for device in self.device_iterator:
            _free_queue = [ctx.SimpleQueue() for _ in range(self.num_players)]
            _full_queue = [ctx.SimpleQueue() for _ in range(self.num_players)]
            free_queue[device] = _free_queue
            full_queue[device] = _full_queue

        learner_model = self.model_func(self.training_device)

        # 权重共享时只需 1 个 optimizer
        if self.share_weights and not self.is_pettingzoo_env:
            single_opt = torch.optim.RMSprop(
                learner_model.parameters(0),
                lr=self.learning_rate,
                momentum=self.momentum,
                eps=self.epsilon,
                alpha=self.alpha,
            )
            optimizers = [single_opt] * self.num_players

            total_steps = self.total_frames // (self.T * self.B) + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                single_opt, T_max=total_steps, eta_min=self.min_lr,
            )
        else:
            optimizers = create_optimizers(
                self.num_players, self.learning_rate,
                self.momentum, self.epsilon, self.alpha, learner_model,
            )
            total_steps = self.total_frames // (self.T * self.B) + 1
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[0], T_max=total_steps, eta_min=self.min_lr,
            )

        stat_keys = []
        for p in range(self.num_players):
            stat_keys.append('mean_episode_return_'+str(p))
            stat_keys.append('loss_'+str(p))
        stat_keys.extend(['epsilon', 'lr'])
        frames, stats = 0, {k: 0 for k in stat_keys}

        if self.load_model and os.path.exists(self.checkpointpath):
            checkpoint_states = torch.load(
                self.checkpointpath,
                map_location="cuda:"+str(self.training_device) if self.training_device != "cpu" else "cpu"
            )
            for p in range(self.num_players):
                learner_model.get_agent(p).load_state_dict(checkpoint_states["model_state_dict"][p])
                for device in self.device_iterator:
                    models[device].get_agent(p).load_state_dict(learner_model.get_agent(p).state_dict())
            if not self.share_weights:
                for p in range(self.num_players):
                    optimizers[p].load_state_dict(checkpoint_states["optimizer_state_dict"][p])
            else:
                optimizers[0].load_state_dict(checkpoint_states["optimizer_state_dict"][0])
            stats.update(checkpoint_states.get("stats", {}))
            frames = checkpoint_states.get("frames", 0)
            log.info(f"Resuming preempted job at frame {frames:,}")

        for device in self.device_iterator:
            for i in range(self.num_actors):
                actor = ctx.Process(
                    target=act_pettingzoo if self.is_pettingzoo_env else act,
                    args=(i, device, self.T, free_queue[device], full_queue[device], models[device], buffers[device], self.env))
                actor.start()
                actor_processes.append(actor)

        # 权重共享时所有位置用同一把锁
        if self.share_weights and not self.is_pettingzoo_env:
            shared_position_lock = threading.Lock()
            position_locks = [shared_position_lock] * self.num_players
        else:
            position_locks = [threading.Lock() for _ in range(self.num_players)]

        def batch_and_learn(i, device, position, local_lock, position_lock, lock=threading.Lock()):
            nonlocal frames, stats
            while frames < self.total_frames:
                batch = get_batch(
                    free_queue[device][position],
                    full_queue[device][position],
                    buffers[device][position],
                    self.B, local_lock
                )
                _stats = learn(
                    position, models,
                    learner_model.get_agent(position),
                    batch, optimizers[position],
                    self.training_device, self.max_grad_norm,
                    self.mean_episode_return_buf, position_lock
                )

                with lock:
                    for k in _stats:
                        stats[k] = _stats[k]

                    if self.share_weights and position == 0:
                        scheduler.step()

                    to_log = dict(frames=frames)
                    to_log.update({k: stats[k] for k in stat_keys})
                    self.plogger.log(to_log)
                    frames += self.T * self.B

        for device in self.device_iterator:
            for m in range(self.num_buffers):
                for p in range(self.num_players):
                    free_queue[device][p].put(m)

        threads = []
        locks = {device: [threading.Lock() for _ in range(self.num_players)] for device in self.device_iterator}

        for device in self.device_iterator:
            for i in range(self.num_threads):
                for position in range(self.num_players):
                    thread = threading.Thread(
                        target=batch_and_learn,
                        name='batch-and-learn-%d' % i,
                        args=(i, device, position, locks[device][position], position_locks[position]))
                    thread.start()
                    threads.append(thread)

        def checkpoint(frames):
            log.info('Saving checkpoint to %s', self.checkpointpath)
            _agents = learner_model.get_agents()
            seen = set()
            model_dicts = []
            for a in _agents:
                model_dicts.append(a.state_dict())
                seen.add(id(a))

            opt_dicts = []
            seen_opt = set()
            for o in optimizers:
                if id(o) not in seen_opt:
                    opt_dicts.append(o.state_dict())
                    seen_opt.add(id(o))
                else:
                    opt_dicts.append(opt_dicts[0])

            torch.save({
                'model_state_dict': model_dicts,
                'optimizer_state_dict': opt_dicts,
                'stats': stats,
                'frames': frames,
                'share_weights': self.share_weights,
            }, self.checkpointpath)

        timer = timeit.default_timer
        try:
            last_checkpoint_time = timer() - self.save_interval * 60
            while frames < self.total_frames:
                start_frames = frames
                start_time = timer()
                time.sleep(5)

                # Epsilon 退火
                new_eps = self._get_epsilon(frames)
                self._update_actor_epsilon(models, new_eps)
                stats['epsilon'] = new_eps
                current_lr = optimizers[0].param_groups[0]['lr']
                stats['lr'] = current_lr

                if timer() - last_checkpoint_time > self.save_interval * 60:
                    checkpoint(frames)
                    last_checkpoint_time = timer()

                end_time = timer()
                fps = (frames - start_frames) / (end_time - start_time)
                log.info(
                    'After %i (%.1fM) frames: @ %.1f fps | eps=%.3f lr=%.2e | Stats:\n%s',
                    frames, frames / 1e6, fps, new_eps, current_lr,
                    pprint.pformat({k: v for k, v in stats.items() if not k.startswith('epsilon') and not k.startswith('lr')}),
                )
        except KeyboardInterrupt:
            return
        else:
            for thread in threads:
                thread.join()
            log.info('Learning finished after %d frames.', frames)

        checkpoint(frames)
        self.plogger.close()
