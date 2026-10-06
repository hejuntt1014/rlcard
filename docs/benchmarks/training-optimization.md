# RTX 5060 training throughput study

Measured on 2026-10-06 using an RTX 5060 8 GB, Core i5-14400F, Python 3.13.5,
PyTorch 2.13.0+cu130 and driver 610.57.04. The reference is commit `6152975`.
All training measurements use A3 context MLP24, width 768 with four residual
blocks, original game rewards, all five auxiliary tasks and BF16. The model has
6,204,180 parameters; its history, legal action set and reward target are unchanged.

## Measurement contract

Throughput counts **samples consumed by the learner**, not speculative actor
steps, candidate actions, games or GPU-only kernel operations. Wall time covers
`Trainer.start`, including actor startup, compilation when selected, shutdown and
the final checkpoint. Python imports and starting the external MPS controller
are outside this timer. Steady throughput uses progress after the first 20% of
the sample budget, including subsequent weight refreshes and final shutdown.
Torch's on-disk compilation cache may be warm; these are not clean-install
compilation timings.

Runs execute sequentially on the same machine. Whole-device memory is sampled
every 0.5 seconds and includes approximately 806 MiB of background allocation;
sampling can miss brief peaks. The repeated comparisons use two training seeds
with 5,000,320 learned samples per run. Timing experiments are not playing-strength
evaluations. Batch size, actor count, transport length and stale-policy exposure
can affect learning even when the network and number of samples are unchanged.

## Results

The reference implementation at its original one-actor/128-environment setting
reproduced **32,378 samples/s**, or **33,803/s** after warmup, over 2 million
learned samples. Tuning concurrency alone is therefore a substantial part of
the overall gain; it must not all be credited to code changes.

All rows below use **4 actors x 512 environments, batch 32 x unroll 20**, so they
retain the same 640 samples per optimizer update. Each run learns 5,000,320 samples.

| Implementation/runtime | Seed | Wall samples/s | Steady samples/s | Sampled whole-device peak |
| --- | ---: | ---: | ---: | ---: |
| Reference `6152975` | 42 | 67,943 | 70,834 | 2,410 MiB |
| Reference `6152975` | 43 | 68,220 | 71,063 | 2,370 MiB |
| Compiled objective, actor half weights, row transport | 42 | 114,091 | 127,524 | 2,326 MiB |
| Compiled objective, actor half weights, row transport | 43 | 115,459 | 127,910 | 2,310 MiB |
| Same row transport with MPS | 42 | 137,609 | 158,309 | 2,360 MiB |
| Same row transport with MPS | 43 | 140,257 | 161,977 | 2,344 MiB |
| Native column transport, compiled objective, actor half weights, MPS | 42 | 149,110 | 171,053 | 2,360 MiB |
| Native column transport, compiled objective, actor half weights, MPS | 43 | 148,838 | 171,582 | 2,402 MiB |

The first column-transport run without MPS included additional compilation startup:
101,120/s wall versus 131,985/s steady. Startup remains included in wall results.
The final seed-42 MPS configuration is **4.61x** the original one-actor wall result,
but that comparison combines concurrency, runtime and code improvements.
The final two-seed mean is **148,974/s wall** and **171,318/s steady**, approximately
**2.19x** the equally parallel reference wall result. The seed-43 column run
without MPS measured 118,434/s wall and 131,739/s steady. All exploratory profiles,
timing samples and configurations are recorded in
[training-throughput-20261006.json](training-throughput-20261006.json).

More aggressive measured options are not automatically better:

| Experiment | Wall samples/s | Steady samples/s | Whole-device peak | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Column transport, MPS, 4 actors, actor graphs, 640 samples/update | 147,974 | 170,372 | 5,926 MiB | Extra memory did not improve throughput |
| Column transport, MPS, 6 actors, 640 samples/update | 145,525 | 168,032 | 2,897 MiB | More actors did not improve this setting |
| Row transport, MPS, 8 actors, 640 samples/update | 130,388 | 155,500 | 3,435 MiB | Concurrency overhead |
| Row transport, MPS, 12 actors, 640 samples/update | 100,027 | 121,727 | 4,523 MiB | Substantial concurrency overhead |
| Column transport, MPS, 6 actors, 5,120 samples/update, eager learner | 163,549 | 181,825 | 3,429 MiB | Eight times fewer optimizer updates per frame |

The large-batch row is an optional throughput experiment, not evidence of equal
learning progress. No playing-strength ranking is inferred from training loss.

## Reproduce a comparison

Build the native extension and run this command from the desired checkout:

```bash
python -m examples.benchmark_training --output experiments/performance \
  --label reference --frames 5000000 --actors 4 --envs 512 --batch 32 --unroll 20

python -m examples.benchmark_training --output experiments/performance \
  --label optimized --frames 5000000 --actors 4 --envs 512 --batch 32 --unroll 20 \
  --trainer-options '{"actor_half_weights":true,"compile_learner":true,"compile_mode":"reduce-overhead"}'
```

The reference checkout must use the same benchmark script. The optional compiler
needs the PyTorch/Triton toolchain, including the matching Python development
headers. Checkpoint weights remain FP32 and ordinary evaluation does not require
compilation or MPS.

The measured 5060 configuration can be used for a longer run with:

```bash
python -m examples.run_dmc --env a3dizhu-v12 --backend cpp \
  --cuda 0 --training_device 0 --num_actors 4 --envs_per_actor 512 \
  --batch_size 32 --unroll_length 20 --num_buffers 64 --share_weights \
  --history_steps 24 --precision bf16 --actor_half_weights \
  --compile_learner --compile_mode reduce-overhead \
  --initial_epsilon .08 --final_epsilon .01 --total_frames 1000000000 \
  --xpid a3_fast
```

Run it inside the private MPS environment below to match the final MPS results.
Do not enable actor graphs just to occupy more memory: the measured combination
used more memory without additional throughput.

Linux MPS can be tested without changing the trainer or installing a system
service. Use a private pipe directory and shut down that controller after its
clients have exited:

```bash
export PATH="/usr/sbin:$PATH"
export CUDA_VISIBLE_DEVICES=0
export CUDA_MPS_PIPE_DIRECTORY="$(mktemp -d /tmp/rlcard-mps.XXXXXX)"
export CUDA_MPS_LOG_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY/log"
mkdir -p "$CUDA_MPS_LOG_DIRECTORY"
nvidia-cuda-mps-control -d
# Run the selected training command here with these environment variables.
echo get_server_list | nvidia-cuda-mps-control
echo quit | nvidia-cuda-mps-control
```

The server executable is under `/usr/sbin` on this host, so that directory must
be in the controller's PATH. Controller startup alone does not establish that
MPS is active; the measurements verify a server and client connections.
See [NVIDIA's MPS reference](https://docs.nvidia.com/deploy/mps/appendix-tools-and-interface-reference.html).

## Implemented work

- Native batching advances rule players, captures pre-action labels and executes
  chosen actions through batch interfaces. Indexed actions bypass string
  creation, transfer and parsing; reset, rule changes and moves invalidate the
  candidate cache. Invalid indexed batches are validated before any move executes.
- Native batch arrays own their C++ allocation through a NumPy capsule. Original
  payoff training skips unused shaping computation and old-state copies.
  Rank masks replace repeated straight scans; card-name and rank-order tables
  avoid temporary string and set allocations.
- Trajectories are packed into shared NumPy views in bounded groups of unrolls,
  preserving the existing slot ownership and queue order. Learners gather into
  reusable pinned buffers; transfer events protect those buffers until H2D reads
  complete. Reused weight snapshots publish a complete version after one fence.
- The A3 native rollout buffer stores selected observations, actions and public
  labels in contiguous C++ columns. Completed episodes transfer ownership through
  capsules. The column writer consumes array slices across episode and unroll
  boundaries, eliminating Python objects and three NumPy copies per decision.
  Original and shaped targets retain the reference double-precision accumulation
  order before FP32 storage. A hybrid segmented argmax preserves first ties/NaNs
  and the original random-number stream.
- Context training selects homogeneous phases on the CPU and uses fixed-shape
  mixed-phase computation. Unused heads retain absent gradients on homogeneous
  batches. Auxiliary classification tasks share one cross-entropy operation,
  retaining their masks and task weights.
- Optional compilation covers the forward pass and complete loss. Gradient
  clipping, RMSprop, the frame scheduler and checkpoint storage remain explicit.
  Compiled scalar losses are copied before retention in statistics.
- Optional actor half weights keep only Linear parameters in BF16/FP16; LayerNorm
  and learner master weights stay FP32. Optional CUDA Graphs use bounded caches,
  finer candidate buckets, in-place weight refresh and safe output ownership.

The native CPU-only benchmark improves from 170,831 to 383,460 decisions/s on the
Windows development machine. This **2.24x isolated result is not a full training
speedup**. Observation/action/label/payoff fingerprints agree with the reference
for both self-play and rule-opponent rollouts. See
[native measurements](native-cpu-optimization.json) and
[trajectory packing measurements](trajectory-writer.json).
Column recording and writing measurements are in
[native-column-rollout.json](native-column-rollout.json) and
[column-writer.json](column-writer.json). Their isolated speedups must not be
multiplied to predict full-pipeline throughput.

Compilation and CUDA Graphs reduce launch overhead but can add startup cost and
memory. BF16 fusion, GEMM selection and padding can change low-order numerical
results and close action rankings. They do not promise identical training
trajectories. See the [PyTorch compiler reference](https://docs.pytorch.org/docs/stable/generated/torch.compile)
and [performance tuning guide](https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html).

## Interpretation and limits

`batch_size * unroll_length` determines samples per optimizer update. Raising
that product is a separate tuning experiment: it reduces the update count for
the same number of frames. The principal comparison keeps it at 640. The transport
unroll is unrelated to the 24-decision observation history.

Increasing one actor from 512 to 2048 environments did not improve the short-run
result. More actors were useful because environment work and inference could
overlap. Restricting processes to performance CPU cores and reducing queue polling
delays had small, single-run effects and are not defaults. Graphs improved isolated
scoring, but a four-actor large-batch run used approximately 6.4 GiB instead of
2.9 GiB without a clear end-to-end improvement. Occupied memory is acceptable
when it provides throughput; memory savings alone are not the selection criterion.

Model shrinking, shorter history, dropped learning records, approximate action
pruning, changed rewards and less frequent training were not used to establish
the principal speedup. FP8/INT8 actors, distillation and policy-guided pruning
need separate playing-quality validation. Activation checkpointing trades extra
compute for memory and is not useful for this measured memory footprint.

## Validation

The final Windows native build and complete suite passed: **249 passed, 7 CUDA
tests skipped**. The Linux/5060 run exercised **36 native, transport, training and
CUDA tests**. One test's hard-coded LayerNorm output dtype assumption differed
on PyTorch 2.13; it was changed to check the actual autocast context and eager
reference output. The affected four-test graph group then passed. No production
code change was required for that test correction.

Validation includes pre-action auxiliary labels, reward and target parity,
capsule lifetime, indexed action invalidation, partial and unordered transport
slots, pinned-buffer reuse, graph output ownership and parameter refresh,
homogeneous-phase gradient absence, and portable compiled checkpoints.
