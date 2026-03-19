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

import logging
import time
import traceback

import numpy as np
import torch

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

def get_batch(
    free_queue,
    full_queue,
    buffers,
    batch_size,
    lock
):
    with lock:
        indices = [full_queue.get() for _ in range(batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def create_buffers(
    T,
    num_buffers,
    state_shape,
    action_shape,
    device_iterator,
):
    buffers = {}
    for device in device_iterator:
        buffers[device] = []
        for player_id in range(len(state_shape)):
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                state=dict(size=(T,)+tuple(state_shape[player_id]), dtype=torch.int8),
                action=dict(size=(T,)+tuple(action_shape[player_id]), dtype=torch.int8),
                aux_target=dict(size=(T, 3), dtype=torch.int8),
            )
            _buffers = {key: [] for key in specs}
            for _ in range(num_buffers):
                for key in _buffers:
                    if device == "cpu":
                        _buffer = torch.empty(**specs[key]).to('cpu').share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to('cuda:'+str(device)).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device].append(_buffers)
    return buffers

def create_optimizers(
    num_players,
    learning_rate,
    momentum,
    epsilon,
    alpha,
    learner_model
):
    optimizers = []
    for player_id in range(num_players):
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(player_id),
            lr=learning_rate,
            momentum=momentum,
            eps=epsilon,
            alpha=alpha)
        optimizers.append(optimizer)
    return optimizers

def _compute_targets_with_step_rewards(payoff, step_rewards):
    """将终局 payoff 与 per-step 中间奖励合并为每步的 target。

    target[t] = payoff + sum(step_rewards[t:])

    这样早期的合作行为能看到更多后续中间奖励的累积，
    提供比纯终局 payoff 更精细的信用分配信号。
    """
    n = len(step_rewards)
    if n == 0:
        return []
    suffix_sum = [0.0] * (n + 1)
    for t in range(n - 1, -1, -1):
        suffix_sum[t] = suffix_sum[t + 1] + step_rewards[t]
    return [payoff + suffix_sum[t] for t in range(n)]


def _flush_buffers(T, free_queue, full_queue, buffers,
                   done_buf, episode_return_buf, target_buf,
                   state_buf, action_buf, aux_target_buf, size,
                   num_players):
    """Drain per-player accumulators into shared-memory ring buffers.

    Uses ``SimpleQueue._reader.poll(0)`` for a reliable non-blocking
    check so the actor never stalls when no free slot is available.
    """
    for p in range(num_players):
        while size[p] >= T:
            if not free_queue[p]._reader.poll(0):
                break
            index = free_queue[p].get()
            for t in range(T):
                buffers[p]['done'][index][t, ...] = done_buf[p][t]
                buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                buffers[p]['target'][index][t, ...] = target_buf[p][t]
                buffers[p]['state'][index][t, ...] = state_buf[p][t]
                buffers[p]['action'][index][t, ...] = action_buf[p][t]
                buffers[p]['aux_target'][index][t, ...] = aux_target_buf[p][t]
            full_queue[p].put(index)
            done_buf[p] = done_buf[p][T:]
            episode_return_buf[p] = episode_return_buf[p][T:]
            target_buf[p] = target_buf[p][T:]
            state_buf[p] = state_buf[p][T:]
            action_buf[p] = action_buf[p][T:]
            aux_target_buf[p] = aux_target_buf[p][T:]
            size[p] -= T


def act(
    i,
    device,
    T,
    free_queue,
    full_queue,
    model,
    buffers,
    env
):
    try:
        log.info('Device %s Actor %i started.', str(device), i)

        # Configure environment
        env.seed(i)
        env.set_agents(model.get_agents())

        done_buf = [[] for _ in range(env.num_players)]
        episode_return_buf = [[] for _ in range(env.num_players)]
        target_buf = [[] for _ in range(env.num_players)]
        state_buf = [[] for _ in range(env.num_players)]
        action_buf = [[] for _ in range(env.num_players)]
        aux_target_buf = [[] for _ in range(env.num_players)]
        size = [0 for _ in range(env.num_players)]

        while True:
            trajectories, payoffs, step_rewards = env.run(is_training=True)
            aux_targets = env.get_aux_targets()
            for p in range(env.num_players):
                num_steps = len(trajectories[p][:-1]) // 2
                size[p] += num_steps
                diff = size[p] - len(target_buf[p])
                if diff > 0:
                    done_buf[p].extend([False for _ in range(diff-1)])
                    done_buf[p].append(True)
                    episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                    episode_return_buf[p].append(float(payoffs[p]))

                    sr = step_rewards[p] if p < len(step_rewards) else []
                    if len(sr) < diff:
                        sr = sr + [0.0] * (diff - len(sr))
                    elif len(sr) > diff:
                        sr = sr[:diff]
                    targets = _compute_targets_with_step_rewards(
                        float(payoffs[p]), sr
                    )
                    target_buf[p].extend(targets)

                    at = torch.from_numpy(aux_targets[p])
                    for i in range(0, len(trajectories[p])-2, 2):
                        state = trajectories[p][i]['obs']
                        action = env.get_action_feature(trajectories[p][i+1])
                        state_buf[p].append(torch.from_numpy(state))
                        action_buf[p].append(torch.from_numpy(action))
                        aux_target_buf[p].append(at)
                
                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    buffers[p]['done'][index][:] = torch.tensor(done_buf[p][:T])
                    buffers[p]['episode_return'][index][:] = torch.tensor(
                        episode_return_buf[p][:T], dtype=torch.float32)
                    buffers[p]['target'][index][:] = torch.tensor(
                        target_buf[p][:T], dtype=torch.float32)
                    buffers[p]['state'][index][:] = torch.stack(state_buf[p][:T])
                    buffers[p]['action'][index][:] = torch.stack(action_buf[p][:T])
                    buffers[p]['aux_target'][index][:] = torch.stack(aux_target_buf[p][:T])
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    state_buf[p] = state_buf[p][T:]
                    action_buf[p] = action_buf[p][T:]
                    aux_target_buf[p] = aux_target_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e


# ═══════════════════════════════════════════════════════════════════════════
#  Vectorized Actor — batched GPU inference over many C++ environments
# ═══════════════════════════════════════════════════════════════════════════

_ACTION_DIM = 52

def act_vectorized(
    i,
    device,
    T,
    free_queue,
    full_queue,
    model,
    buffers,
    num_envs,
    env_config,
):
    """One process managing *num_envs* C++ environments with batched inference.

    Replaces the original ``act()`` which ran a single env per process.
    Reduces process count by ~10-30x while batching GPU forward passes,
    eliminating both the CPU process-storm and the GPU data-starvation.
    """
    try:
        from a3dizhu_cpp import VectorizedEngine
        log.info('Device %s VecActor %i started (%d envs).', str(device), i, num_envs)

        vec = VectorizedEngine(num_envs)
        vec.set_greedy_ratio(env_config.get('greedy_ratio', 0.0))
        vec.set_random_ratio(env_config.get('random_ratio', 0.0))
        vec.seed(i * num_envs + 42)

        num_players = 4
        agent = model.get_agent(0)
        gpu_device = 'cuda:' + str(device) if device != 'cpu' else 'cpu'

        done_buf           = [[] for _ in range(num_players)]
        episode_return_buf = [[] for _ in range(num_players)]
        target_buf         = [[] for _ in range(num_players)]
        state_buf          = [[] for _ in range(num_players)]
        action_buf         = [[] for _ in range(num_players)]
        aux_target_buf     = [[] for _ in range(num_players)]
        size               = [0  for _ in range(num_players)]

        # Per-env, per-player trajectory: list of (obs, action_feat)
        env_steps = [[[] for _ in range(num_players)] for _ in range(num_envs)]

        for e in range(num_envs):
            vec.reset(e)

        # Stagger: fast-forward each env by a random number of steps so
        # games finish at different times instead of all at once.
        for e in range(num_envs):
            skip = np.random.randint(0, 60)
            for _ in range(skip):
                if vec.is_over(e):
                    vec.reset(e)
                done, _, _ = vec.advance_to_decision(e)
                if done:
                    vec.reset(e)
                    continue
                vec.step_random(e)
        env_steps = [[[] for _ in range(num_players)] for _ in range(num_envs)]
        log.info('Device %s VecActor %i stagger done.', str(device), i)

        _round = 0
        _t_log = time.time()

        while True:
            _t0 = time.time()

            # ── Phase 1: advance every env through rule-agent turns ──
            pending = []
            newly_done = []

            for e in range(num_envs):
                is_done, rule_obs, rule_pids = vec.advance_to_decision(e)
                n_rule = len(rule_pids)
                if n_rule > 0:
                    zeros = np.zeros(_ACTION_DIM, dtype=np.int8)
                    for si in range(n_rule):
                        env_steps[e][rule_pids[si]].append(
                            (rule_obs[si].copy(), zeros))

                if is_done:
                    newly_done.append(e)
                else:
                    pending.append(e)

            _t1 = time.time()

            # ── Phase 2: harvest finished games → fill training buffers ──
            for e in newly_done:
                _collect_game(vec, e, env_steps[e],
                              done_buf, episode_return_buf, target_buf,
                              state_buf, action_buf, aux_target_buf, size,
                              num_players)
                env_steps[e] = [[] for _ in range(num_players)]
                vec.reset(e)

            _t2 = time.time()

            # ── Phase 3: batched GPU inference for all pending envs ──
            if pending:
                obs_exp, act_flat, obs_raw, offsets, all_keys = \
                    vec.prepare_batch(pending)

                with torch.no_grad():
                    q_values = agent.net.forward(
                        torch.from_numpy(np.ascontiguousarray(obs_exp)).to(gpu_device).float(),
                        torch.from_numpy(np.ascontiguousarray(act_flat)).to(gpu_device).float(),
                    ).cpu().numpy()

                eps = agent.exp_epsilon
                for idx, e in enumerate(pending):
                    pid = vec.get_player_id(e)
                    start = offsets[idx]
                    end   = offsets[idx + 1]
                    n_act = end - start

                    if eps > 0 and np.random.rand() < eps:
                        ai = np.random.randint(n_act)
                    else:
                        ai = int(np.argmax(q_values[start:end]))

                    chosen_key = all_keys[idx][ai]
                    obs_i   = obs_raw[idx].copy()
                    afeat_i = act_flat[start + ai].copy()
                    env_steps[e][pid].append((obs_i, afeat_i))

                    vec.step(e, chosen_key)

            _t3 = time.time()

            # ── Phase 4: flush full unroll windows to learner ──
            _flush_buffers(T, free_queue, full_queue, buffers,
                           done_buf, episode_return_buf, target_buf,
                           state_buf, action_buf, aux_target_buf, size,
                           num_players)

            # Backpressure: sleep proportionally to backlog depth so
            # actors self-throttle to match the learner's consumption.
            _max_size = max(size)
            if _max_size > T * 3:
                time.sleep(min((_max_size / T) * 0.002, 0.5))

            _t4 = time.time()

            _round += 1
            if time.time() - _t_log > 30.0:
                log.info(
                    'VecActor %i dev %s round %d | '
                    'P1=%.0fms P2=%.0fms P3=%.0fms P4=%.0fms total=%.0fms | '
                    'pending=%d done=%d sizes=%s',
                    i, str(device), _round,
                    (_t1-_t0)*1000, (_t2-_t1)*1000,
                    (_t3-_t2)*1000, (_t4-_t3)*1000, (_t4-_t0)*1000,
                    len(pending), len(newly_done), size[:])
                _t_log = time.time()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in vec worker %i', i)
        traceback.print_exc()
        print()
        raise e


def _collect_game(vec, e, steps_per_player,
                  done_buf, episode_return_buf, target_buf,
                  state_buf, action_buf, aux_target_buf, size,
                  num_players):
    """Move one finished game's trajectory into the per-player accumulators."""
    payoffs     = vec.get_training_payoffs(e)
    aux_targets = vec.get_aux_targets(e)

    for p in range(num_players):
        steps = steps_per_player[p]
        n = len(steps)
        if n == 0:
            continue

        size[p] += n

        done_buf[p].extend([False] * (n - 1))
        done_buf[p].append(True)
        episode_return_buf[p].extend([0.0] * (n - 1))
        episode_return_buf[p].append(float(payoffs[p]))

        sr = list(vec.get_step_rewards(e, p))
        if len(sr) < n:
            sr += [0.0] * (n - len(sr))
        elif len(sr) > n:
            sr = sr[:n]
        targets = _compute_targets_with_step_rewards(float(payoffs[p]), sr)
        target_buf[p].extend(targets)

        at = torch.from_numpy(np.array(aux_targets[p], dtype=np.int8))
        for obs, afeat in steps:
            state_buf[p].append(torch.from_numpy(obs))
            action_buf[p].append(torch.from_numpy(afeat))
            aux_target_buf[p].append(at)
