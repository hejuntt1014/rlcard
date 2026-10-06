"""Experimental actor threads with one shared GPU policy and private environments."""
import contextlib
import copy
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .utils import ColumnTrajectoryWriter


def run_threaded_actor(seed, env, model, shared_model, model_lock, local_version,
                       policy_version, buffers, free_queues, full_queues, stop,
                       epsilon, counters, T, count, max_actions, max_episode_steps,
                       actor_poll_interval, thread_count):
    from rlcard.envs.a3dizhu.dmc import NativePool
    locks = [threading.Lock() for _ in range(thread_count)]
    local_stop = threading.Event()
    device = model.get_agent(0).device
    torch.cuda.current_stream(device).synchronize()

    def run(index):
        thread_seed = int(np.random.SeedSequence([seed, index]).generate_state(1)[0])
        thread_seed %= 2**32 - count
        local_model = copy.copy(model)
        copies = {}
        rng = np.random.RandomState(thread_seed)
        for agent in model.get_agents():
            if id(agent) not in copies:
                copies[id(agent)] = copy.copy(agent)
                copies[id(agent)].rng = rng
        local_model.agents = [copies[id(a)] for a in model.get_agents()]
        pool = NativePool(env() if callable(env) else env, count, thread_seed, max_episode_steps)
        if not pool.columnar:
            raise ValueError('Threaded actors require the current columnar native extension')
        writer = ColumnTrajectoryWriter(T, free_queues, full_queues, buffers)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            while not stop.is_set() and not local_stop.is_set():
                writer.flush()
                if writer.congested():
                    stop.wait(actor_poll_interval)
                    continue
                episodes, steps = pool.round(local_model, epsilon.value, max_actions, locks[index])
                for episode in episodes:
                    writer.add_episode(episode)
                with counters.get_lock():
                    counters[0] += steps
                    counters[1] += len(episodes)

    executor = ThreadPoolExecutor(max_workers=thread_count)
    futures = [executor.submit(run, i) for i in range(thread_count)]
    try:
        while not stop.wait(.001):
            for future in futures:
                if future.done():
                    future.result()
                    return
            if policy_version.value != local_version:
                # Readers synchronize their CUDA stream before releasing their
                # inference lock. Taking every lock excludes partially refreshed
                # parameters without serializing readers against one another.
                with contextlib.ExitStack() as stack:
                    for lock in locks:
                        stack.enter_context(lock)
                    with model_lock:
                        for p in ([0] if model.shared else range(len(model.get_agents()))):
                            model.get_agent(p).load_state_dict(shared_model.get_agent(p).state_dict())
                        local_version = policy_version.value
                        torch.cuda.current_stream(device).synchronize()
    finally:
        local_stop.set()
        executor.shutdown(wait=True)
