# RTX 5060: complete learning replay and shared actor policies

Measured on 2026-10-06 against `a88e66f`, on the RTX 5060 8 GB and i5-14400F
used in the [first study](training-optimization.md). PyTorch is 2.13.0+cu130,
driver 610.57.04. Model dimensions, 24-step history, original game rewards,
auxiliary objectives and **640 samples per update** are unchanged. This study
uses no subagents.

## Matched repeated measurements

Each row learns 10,000,000 samples with BF16, MPS and 2,048 environments.
The reference has four actor processes; the threaded configuration has two
processes with two independent environment threads each. Timers include actor
startup, compilation/capture, shutdown and final saving, but exclude imports and
MPS controller setup. Steady measurements start after 20% of the sample budget.
The compiler disk cache may be warm. Runs execute sequentially.

| Configuration | Seed | Wall samples/s | Steady samples/s | Whole-device peak |
| --- | ---: | ---: | ---: | ---: |
| Previous version `a88e66f` | 42 | 161,485 | 173,546 | 2,360 MiB |
| Previous version `a88e66f` | 43 | 161,975 | 173,362 | 2,344 MiB |
| Complete replay, compiled actors, four processes | 42 | 170,411 | 223,573 | 2,512 MiB |
| Complete replay, compiled actors, four processes | 43 | 189,459 | 224,579 | 2,496 MiB |
| Complete replay, compiled actors, two processes x two threads | 42 | 190,435 | 229,688 | 2,220 MiB |
| Complete replay, compiled actors, two processes x two threads | 43 | 190,309 | 228,604 | 2,279 MiB |

The threaded configuration averages **190,372/s wall** and **229,146/s steady**,
versus **161,730/s** and **173,454/s** for the reference: approximately **17.7%**
and **32.1%** faster respectively. The first four-process compiled run shows why
startup cannot be hidden: steady speeds are similar, but wall time includes
additional compilation. Compare these matched runs rather than dividing by the
first study's shorter runs.

GPU memory is sampled every 0.5 seconds, includes about 806 MiB of background
allocation, and may miss brief peaks. These are throughput and correctness
measurements, not playing-strength evaluations. Fusion can change low-order
numerical results; changing actor scheduling changes self-play trajectories.

## Implemented paths

- **Compiled RMSprop:** `compile_optimizer=True` compiles the existing optimizer.
  The frame scheduler updates a device learning-rate tensor in place, avoiding
  recompilation at every new rate. Checkpoint loading normalizes LR and step
  devices when switching back to ordinary training.
- **Complete-update CUDA replay:** `cuda_graph_learner=True` captures forward,
  loss, backward, clipping and parameter updates. Separate fixed-shape graphs
  handle mixed, play-only and declaration-only batches. Warmup weights and
  optimizer state are restored, so each real batch is learned once. Inactive
  heads keep absent gradients. FP32 and BF16 are supported; dynamic FP16 scaling
  is excluded from this explicit replay path.
- **Shared ownership rings:** `shared_transport=True` uses bounded shared arrays
  for free/full indices. Batched lock-protected transfers avoid pickling and
  feeder threads and claim exactly the needed slots. A slot is returned only
  after its contents have been gathered; pinned staging still has transfer events.
- **Compiled actors:** `compile_actor=True` compiles state encoding and the two
  action heads with dynamic candidate counts and in-place parameter refresh.
  It is mutually exclusive with `actor_cuda_graphs`.
- **Threaded native actors:** each process can have private environment pools,
  writers, random streams and CUDA streams sharing one GPU policy. Publication
  takes every inference lock and synchronizes the update stream before releasing
  readers. Selected C++ operations release the GIL. Each native engine and rollout
  buffer must have one owner; concurrent mutation of the same instance is unsupported.

These paths are optional. Existing CPU and generic RLCard defaults remain.
Threaded actors require native A3 context environments and CUDA, and cannot use
the per-actor CUDA-graph cache. Their environment count is
`num_actor_devices * num_actors * actor_threads * envs_per_actor`.

## Rejected or limited options

- A reusable pinned actor-input prototype had no steady-throughput benefit and
  was removed. Learner pinned staging remains.
- One process with four threads was slower than two processes with two threads;
  shared weights do not eliminate Python scheduling overhead.
- Increasing the selected setup to three/four processes with two threads gave
  229,602/228,934 steady samples/s, essentially unchanged, with higher memory use.
- Shared rings alone gave little benefit before the optimizer was accelerated.
- Capturing an unfused optimizer retains its GPU work; CUDA replay alone does
  not fuse kernels. Compiled actors also add significant startup cost.

## Reproduce

Build the native extension, enter the private MPS environment from the first
study, and run:

```bash
python -m examples.run_dmc --env a3dizhu-v12 --backend cpp \
  --cuda 0 --training_device 0 --num_actors 2 --actor_threads 2 \
  --envs_per_actor 512 --batch_size 32 --unroll_length 20 --num_buffers 64 \
  --share_weights --history_steps 24 --precision bf16 --actor_half_weights \
  --shared_transport --compile_learner --compile_optimizer --cuda_graph_learner \
  --compile_actor --initial_epsilon .08 --final_epsilon .01 \
  --total_frames 1000000000 --xpid a3_fast
```

Raw ablations, repeated runs and microbenchmarks are in
[training-throughput-round2.json](training-throughput-round2.json).
`examples.benchmark_optimizer` and `examples.benchmark_graph_learner` reproduce
the isolated update experiments.

References: [PyTorch RMSprop](https://docs.pytorch.org/docs/main/generated/torch.optim.RMSprop.html),
[NVIDIA CUDA-graph guidance](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/handling-dynamic-patterns.html).

## Validation

After implementation converged, the Windows native build and full suite passed:
**251 passed, 11 CUDA tests skipped**. The PVE run passed **42 native/transport/CUDA
tests**, including multi-producer ring ownership, varying learning rates, phase
switches, no extra warmup updates, compiled RMSprop with identical gradients,
compiled actor parameter refresh, and accelerated checkpoint resume through the
ordinary learner. Only targeted development checks preceded this final round.
