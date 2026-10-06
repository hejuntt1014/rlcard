# A3 context policy

`a3dizhu-v12` supplies structured A3 observations to a policy with reusable state
encoding. The default history encoder is an MLP. A Transformer encoder is available
as a controlled comparison on the same features and game rules.

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

Candidate groups carry one observation per decision, a flat action array and
offsets. Each state is encoded once, and its context is reused across candidate
actions. Declaration candidates use only the declaration head. Forced and
epsilon-exploration choices skip network inference but still produce training
samples. Invalid history slots are masked; chronological positions remain intact.

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

The benchmark obtains real legal decisions, verifies expanded and compact
forward outputs agree, and times both paths using the same network weights.
It reports unique/expanded observation bytes and per-run timings. This isolates
actor scoring behavior; it does not measure end-to-end learning speed or policy
quality. The existing ONNX export script targets the classic A3 model; this policy
currently uses its PyTorch interfaces for training and evaluation.
