# A3 context policy

`a3dizhu-v12` supplies structured A3 observations to a policy with reusable state
encoding. The default history encoder is an MLP. A Transformer encoder is available
as a controlled comparison on the same features and game rules.

See [validation results](benchmarks/context-validation.md) and the
[implementation review](context-review.md) for tested behavior and remaining
experimental questions.

## Model and environment

| Component | Configuration |
| --- | --- |
| Observation | 556 static features and 24 history records of 88 features; 2,668 total |
| Action | 111 features, including card bits, type, attributes and afterstate summaries |
| Static encoder | 556 → 256 |
| History MLP | Masked 24 × 88 records, flattened in order → 256 |
| State context | Concatenated static/history representations → 512 |
| Action encoder | 111 → 128 |
| Play scorer | Context/action fusion → width-768 residual network → Q |
| Declaration scorer | State context → values for pass and declare |
| Auxiliary tasks | Public relation labels and publicly known spade-3/spade-A ownership |

The default model has 6,204,180 parameters (about 23.67 MiB of FP32 weights). It
does not give the actor opponents' hidden hands. Unknown auxiliary labels are
masked, and each label is captured at its decision state. These auxiliary tasks
reconstruct public information; their presence does not establish hidden-card
inference ability or improved playing strength.

`--history_steps 16` uses the most recent 16 valid records and has 6,023,956
parameters (about 22.98 MiB). The environment still supplies the same 24-record
feature protocol. Early games retain their available records; trailing padding is
not mistaken for recent history. The default window is 24,
and the checkpoint records the selected window. Changing it requires a new model.

Candidate groups carry one observation per decision, a flat action array and
offsets. Each state is encoded once, and its context is reused across candidate
actions. Declaration candidates use only the declaration head. Forced and
epsilon-exploration choices skip network inference but still produce training
samples. Invalid history slots are masked; chronological positions remain intact.
`max_inference_actions` limits candidate rows per scoring forward. Unique states,
candidate features and their indices are transferred as complete groups, so this
setting is not a hard limit on total GPU memory.

The native engine is required. `a3dizhu` remains available with its 850/52 feature
layout and Python fallback. Feature schemas and model architectures are distinct;
their checkpoints are not interchangeable.

## Train

```bash
python setup_cpp.py build_ext --inplace
python -m examples.run_dmc --env a3dizhu-v12 --backend cpp --cuda 0 --training_device 0 --num_actors 1 --envs_per_actor 128 --share_weights --batch_size 32 --unroll_length 20 --num_buffers 64 --initial_epsilon 0.08 --final_epsilon 0.01 --total_frames 1000000 --xpid a3_context
```

The default precision is FP32. `--precision bf16` enables CUDA autocast in the
learner and GPU actors. `--precision fp16` additionally uses gradient scaling for
the learner. Checkpoint weights remain FP32; reduced-precision playing quality
requires independent evaluation. CPU actors remain FP32.

Select `--history_encoder transformer` for the attention comparison, or
`--architecture resnet` for a flat-observation residual policy on the same features.
`--no-auxiliary` disables auxiliary heads and losses. Use separate experiment IDs
for different model specifications.

## Execution controls

The A3 native collector automatically uses columnar rollout storage when the
extension exposes `RolloutBuffer`. Observations, selected actions and decision-time
auxiliary labels are recorded in contiguous C++ arrays. Completed episodes carry
per-role NumPy columns directly to the shared-memory writer, which copies whole
slices across unroll boundaries. Finished arrays retain their own storage across
environment resets. Episode-length limits and reward/trajectory lengths are
checked before publishing training data. Extensions without this interface use
the compatible row-based collector and writer.

Candidate actions use native indices when supported, avoiding action-string
conversion during self-play. Fixed-rule resets invalidate cached candidates.
Original-game-reward training skips cooperative step-reward calculations while
retaining the same terminal game payoffs and decision records.

Pinned host batches and dense learner scoring are enabled by default
(`pin_memory=True`, `dense_learner=True`). Pinned buffers are reused only after
their device reads finish. Dense scoring keeps mixed declaration/play batches at
a fixed shape; homogeneous batches skip the inactive head and preserve absent
gradients for that head.

The following options are disabled by default and can be enabled independently:

| Option | Behavior |
| --- | --- |
| `--compile_learner` | Compile state/action encoding, Q prediction, return regression loss and auxiliary losses together; gradient clipping and RMSprop remain eager |
| `--actor_half_weights` | Store only actor `Linear` weights in BF16/FP16; normalization parameters stay FP32; requires CUDA actors and matching `--precision` |
| `--actor_cuda_graphs` | Reuse bounded CUDA inference graphs for context encoding and action scoring; requires CUDA actors |

Learner compilation requires PyTorch 2.x and a compatible `torch.compile` backend
and compiler toolchain. `--compile_mode default` is the default mode;
`reduce-overhead` and `max-autotune` are also available. Compilation and graph
capture have startup costs and can increase memory use. Oversized inputs and
additional sizes after the graph cache is full use eager inference. Saved weights
remain FP32 with ordinary parameter names regardless of these execution options.

The learner's `--learner_poll_interval` defaults to 0.002 seconds; the actor's
`--actor_poll_interval` defaults to 0.005 seconds. Smaller values trade additional
CPU polling for potentially shorter idle periods. The CLI uses `--cpu_threads 1`
by default. In the Python API, `num_threads=None` preserves an existing PyTorch
thread setting, and a positive value sets the learner process's thread count.

`history_steps` and `unroll_length` control different things. The history window
selects up to 24 records within each observation. Unroll length controls how many
decision samples are packed into a shared-memory block. Each learner update uses
`batch_size * unroll_length` samples; increasing this product reduces optimizer
updates per frame. Changing 32 × 20 to 8 × 80 preserves 640 samples per update,
without changing the model's history window. Throughput comparisons with larger
update batches require separate playing-quality evaluation.

## Rules and rewards

By default, games sample the eight combinations of three rule switches. A fixed
rule dictionary must contain:

```python
rules = {
    'declare_require_both_spades': True,
    'straight_start_val': 2,  # 1: starts at 3; 2: starts at 4
    'straight_end_val': 11,  # 11: ends at K; 12: ends at A
}
env = rlcard.make('a3dizhu-v12', config={'rules': rules, 'reward_mode': 'game'})
```

CLI users can pass the dictionary as JSON through `--rules`. Fixed rules are
reapplied after every reset, including batched native resets. Training checkpoints
record the rules, feature schema, reward mode, history encoder and auxiliary-label
contract; incompatible resume settings fail before workers start.

The default `--reward_mode game` uses the original terminal game payoff.
`--reward_mode shaped` includes rank bonuses and cooperative step rewards. It is
an experiment option, not an established improvement. In declared games, both
partners on the anti-declarant side are recognized by cooperative rewards.

Afterstate features include heuristic rank-group and pattern summaries. They are
not an exact minimum-move solver. Card classification respects the current straight
range regardless of whether a card combination was previously cached.

## Evaluate playing quality

See the [second RTX 5060 study](benchmarks/training-optimization-round2.md) for
optional complete-update CUDA replay and threaded native actors. When using
`actor_threads > 1`, `envs_per_actor` is the environment count per thread.

```bash
python -m examples.evaluate_dmc experiments/dmc_result/a3_context/model.tar --env a3dizhu-v12 --seeds 100 --output experiments/evaluation/context.json
```

Each seed is evaluated with the model in all four seats against three native
greedy players. Scores use original game payoffs, not training reward shaping.
Reports include scores, wins/draws/losses, declaration behavior, rule and mode
breakdowns. Standard errors are computed across seed averages because games with
rotated seats share a deal.

Pass multiple compatible checkpoint paths to obtain paired score differences on
identical seeds and seats. Illegal opponent or model actions stop evaluation and
save a reproduction trace; they are never replaced silently. Greedy-opponent
results are a sanity baseline, not evidence of superiority over trained V9/V12
policies. Use multiple training seeds, equal GPU-hour budgets, and a stronger
opponent pool for that conclusion.

## Measure inference work

```bash
python -m examples.benchmark_context --device 0 --envs 128 --repeats 20
```

Repeat with `--history_steps 16` to compare the shorter window on the same seeded
decisions. Window length must also be compared by paired game scores; inference
speed alone does not identify the stronger policy.

The benchmark obtains real legal decisions, verifies expanded and compact
forward outputs agree, and times both paths using the same network weights.
It reports unique/expanded observation bytes and per-run timings. This isolates
actor scoring behavior; it does not measure end-to-end learning speed or policy
quality. The existing ONNX export script targets the classic A3 model; this policy
currently uses its PyTorch interfaces for training and evaluation.
