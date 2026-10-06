# Batched DMC training

DMC supports batches of independent card games in each actor process. Decisions
are grouped by policy, legal state-action pairs are scored together, and a single
learner consumes fixed-length trajectory blocks from CPU shared memory.

## Install

Python 3.10 or later is required. Check out the training branch:

```bash
git clone --branch feat/batched-dmc https://github.com/hejuntt1014/rlcard.git
cd rlcard
python -m pip install -e ".[torch]"
```

The native A3 backend is optional and requires a C++17 compiler:

```bash
python -m pip install pybind11
python setup_cpp.py build_ext --inplace
```

Use GCC or Clang on Linux, or Visual Studio C++ Build Tools on Windows. Python
environments and the generic trainer work without this extension.

## CPU training

```bash
python -m examples.run_dmc --env leduc-holdem --num_actors 2 --envs_per_actor 8 --total_frames 100000 --batch_size 16 --unroll_length 20 --num_buffers 64 --hidden_sizes 128 128 --xpid leduc_cpu
```

Independent policies are used for each role by default. Ordinary RLCard games use
an MLP and terminal returns. Generic observation and action buffers use float32,
so fractional features are preserved. A3 uses integer buffers and optional
game-specific reward and auxiliary-label adapters.

## GPU training

```bash
python -m examples.run_dmc --env doudizhu --cuda 0,1 --num_actor_devices 1 --training_device 1 --num_actors 2 --envs_per_actor 32 --total_frames 10000000 --xpid doudizhu_gpu
```

Device indices refer to the devices listed in `--cuda`. Actors use the first
`num_actor_devices` devices; the learner uses `training_device`. A single GPU can
run both roles. Environment simulation runs on the CPU. This is actor/learner
parallelism, not distributed data-parallel optimization.

For a GPU with 8 GB of memory, start with CPU actors and a small learner batch:

```bash
python -m examples.run_dmc --env leduc-holdem --cuda 0 --training_device 0 --actor_on_cpu --num_actors 2 --envs_per_actor 8 --batch_size 16 --unroll_length 20 --num_buffers 64 --hidden_sizes 128 128 --total_frames 100000 --xpid single_gpu
```

`--actor_on_cpu` keeps actor models and inference on the CPU, so only the learner
uses GPU memory. Increase the batch and model sizes after measuring peak memory
on the target hardware. Free GPU memory, not the card's nominal capacity,
determines the usable batch size.

## A3 native training

```bash
python -m examples.run_dmc --env a3dizhu --backend cpp --share_weights --envs_per_actor 200 --num_actors 2 --batch_size 64 --num_buffers 128 --total_frames 1000000 --xpid a3_native
```

`backend=auto` selects native batching when the extension exposes the required
interface. Otherwise it uses the Python collector. `backend=cpp` fails before
starting workers when the extension is unavailable. The A3 environment itself
accepts `config={'backend': 'python'}` to select pure Python rules explicitly.

Rule opponents contribute their actual selected actions. Step rewards must align
exactly with recorded decisions. Completed games supply targets; incomplete games
and incomplete unroll blocks are not counted as trained samples.

## Python API

```python
import rlcard
from rlcard.agents.dmc_agent import DMCTrainer

if __name__ == '__main__':
    env = rlcard.make('leduc-holdem', config={'seed': 42})
    trainer = DMCTrainer(
        env, num_actors=2, envs_per_actor=8,
        total_frames=100000, batch_size=16, unroll_length=20,
        num_buffers=64, seed=42,
    )
    statistics = trainer.start()
```

The main guard is required for spawned workers. Custom environments can provide
a picklable top-level `env_factory` that constructs a fresh environment, and an
`adapter_class` implementing the `RLCardAdapter` interface. Environments made with
`rlcard.make` retain their construction configuration for worker creation.

For PettingZoo AEC environments, pass `is_pettingzoo_env=True`. The collector uses
action masks, processes termination and truncation turns, and accumulates
undiscounted rewards after each agent's actions. Parallel PettingZoo environments
are not accepted by this adapter.

## Configuration

| Argument | Meaning |
| --- | --- |
| `share_weights` | Explicitly share one policy across all roles; shapes and role semantics must match |
| `architecture` | `mlp` or `resnet`; defaults to MLP for standard games, ResNet for A3 |
| `auxiliary` | Optional A3 classification targets; use `False` with an A3 MLP |
| `vectorized` | When false, use one environment per actor |
| `envs_per_actor` | Number of games managed by one process |
| `actor_on_cpu` | Keep actors on the CPU while training on the selected GPU |
| `batch_size` | Number of unroll blocks in a learner batch |
| `unroll_length` | Number of decision samples per block |
| `num_buffers` | Slots per actor device and role; must be at least `batch_size` |
| `weight_sync_interval` | Learner updates between actor weight publications |
| `max_inference_actions` | Maximum candidate pairs in one model forward call |
| `actor_timeout` | Seconds without a completed learner batch before failure |
| `max_episode_steps` | Maximum recorded actions in a game before failure; no fabricated truncated targets |

Weight sharing is inappropriate for differently shaped roles such as the landlord
and farmers in standard Dou Dizhu. Equal shapes alone do not establish game
symmetry. `num_threads` is accepted by the trainer, but the learner uses one update
loop. Policy publication uses versioned CPU snapshots and per-device locks. GPU
actors keep local CUDA models and refresh them from published snapshots, so CUDA
tensors do not cross process boundaries. CPU actors read the shared policy under
the same lock used for publication.

Increase pool sizes while the learner lacks data, then measure again. More actors
or larger batches are not guaranteed to improve throughput. Queue backpressure
pauses sampling; it does not silently drop completed samples. The shared-memory
allocation has one tensor per field, device and role, rather than per buffer slot.

## Checkpoints and metrics

Checkpoints are written atomically to `savedir/xpid/model.tar`. They include model
specification, weights, optimizer state, frame-based learning-rate state and
statistics. `--load_model` requires a matching model specification and an existing
checkpoint. Resume restarts actor games; it is not a bitwise replay of interrupted
random streams. The learning-rate horizon follows the configured `total_frames`.

`frames` counts learner samples, and may exceed the requested target by less than
one batch. `learner_updates` counts batches. `actor_steps` and `episodes` count
sampling work in the current invocation, including samples not yet learned.
Exploration is held in a shared scalar visible to all workers. The trainer shuts
down workers and saves its current state on completion or interruption; worker
exceptions are reported to the caller.

## Benchmarks

```bash
python -m examples.benchmark_dmc --env leduc-holdem --envs 1 8 32 --repeats 3 --total_frames 100000
```

The JSON report records configuration, runtime versions, devices, elapsed time,
and sample counts for each run. End-to-end throughput includes process startup and
the final checkpoint. Use sufficiently long runs to limit startup effects. Compare
the same game, model, precision, hardware, batch size and reward settings. Run
benchmarks without other training jobs on the same hardware.

Throughput is separate from policy quality: evaluate checkpoints against fixed
opponents with fixed deals, multiple seeds, and a fixed game-score definition.
No universal speedup factor is asserted for the supported games.

## Validation

```bash
python -m pytest tests/agents/test_dmc.py -q
python test_parity.py
```

The native parity test requires the extension. Optional PettingZoo integration
tests require `pettingzoo[classic]`. GPU throughput and numerical behavior should
be validated on the intended training hardware.
