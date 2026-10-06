# A3 context validation

Measured on 2026-10-06, RTX 5060 8 GB, Core i5-14400F, Python 3.13.5,
PyTorch 2.13.0+cu130. These are implementation and short-learning checks,
not a ranking against trained V9 or V12 agents.

## Candidate scoring

One fixed fixture contains 128 real decisions and 519 legal candidate actions.
Each network is compared against itself using identical weights. FP32 outputs
agree within 0.000003 absolute difference. After warmup, each path was timed 20
times with CUDA synchronization. Reported values are medians.

| History encoder/window | Expanded observations | Reused state context |
| --- | ---: | ---: |
| MLP, 24 | 2.35 ms | 2.09 ms |
| MLP, 16 | 2.40 ms | 2.19 ms |
| Transformer, 24 | 7.30 ms | 3.41 ms |
| Transformer, 16 | 5.84 ms | 3.22 ms |

The compact native representation uses 341,504 observation bytes instead of
1,384,692 expanded observation bytes for this fixture. Actual savings depend on
the number of legal actions. These are actor-scoring times, not complete training
speedups. Window cropping and fixed overhead can offset a smaller MLP input.

Raw timing samples: [24 steps](context-inference-24.json),
[16 steps](context-inference-16.json).

## Equal-sample short training

All runs use the A3 native backend, shared width-768 residual policy, one GPU actor
with 128 environments, batch size 32, unroll length 20, and 1,000,320 learned
samples. There is one training seed (42) per configuration. Timing is wall time
inside `Trainer.start`, including actors and checkpointing but excluding interpreter
imports and evaluation. Runs occurred sequentially; small timing differences can
reflect warm caches or normal measurement variation.

| Configuration | Wall time | Learned samples/s | Sampled whole-GPU peak |
| --- | ---: | ---: | ---: |
| MLP24 FP32 | 36.19 s | 27,641 | 1,316 MiB |
| MLP16 FP32 | 35.99 s | 27,798 | 1,316 MiB |
| MLP24 BF16 | 30.97 s | 32,300 | 1,354 MiB |

Whole-device memory includes approximately 714 MiB of background allocation and
was sampled every 0.5 seconds. BF16 was faster in this run but did not reduce the
sampled memory peak. Weight files remain FP32.

The 16-step and 24-step FP32 runs differ by only about 0.6% in throughput; this is
not evidence of a meaningful performance advantage for either window.

## Fixed-deal evaluation

Initial and trained checkpoints were evaluated using seeds 100000–100019, rotating
the model through all four seats against three native greedy opponents: 80 games
per checkpoint. Metrics use original game payoffs, not reward shaping. Uncertainty
is clustered by seed, because the four seat games share a deal.

| Configuration | Initial mean payoff | Trained mean payoff | Paired gain ± standard error |
| --- | ---: | ---: | ---: |
| MLP24 FP32 | -2.138 | +0.046 | +2.183 ± 0.418 |
| MLP16 FP32 | -2.329 | +0.025 | +2.354 ± 0.402 |
| MLP24 BF16 | -2.138 | -0.004 | +2.133 ± 0.384 |

For trained checkpoints, MLP16 minus MLP24 is -0.021 ± 0.127 payoff; BF16 minus
FP32 is -0.050 ± 0.118. These small differences do not establish a winner.
Improvement over random initialization confirms a useful learning signal in this
check. It does not prove superiority over the greedy opponent, V9, V12, or any
strong policy. Multiple training seeds, longer budgets and stronger opponents are
required for that conclusion.

The behavior breakdown is important: initial checkpoints declared in 37/80 or
42/80 evaluation games and won only one of those declarations. All three trained
checkpoints declared in 0/80 games. The return improvement therefore accompanies
avoiding poor declarations; these runs do not establish better ordinary card play
or a well-calibrated ability to recognize profitable declaration opportunities.

[Raw training statistics and all evaluation games](context-training.json).

## Validated runtime paths

- CPU and CUDA compact scoring, including reused/expanded output and gradient checks.
- FP32, BF16 and FP16 declaration-only and mixed-phase updates on CUDA, including
  gradient scaling for FP16.
- A complete Transformer16 BF16 training smoke run (12,800 samples).
- Native game/action/label consistency, all eight rule settings, and fixed-rule resets.

The default remains 24-step MLP with FP32 and original game rewards. The 16-step
window, Transformer history, BF16/FP16 and shaped rewards remain selectable
experiments rather than claims of improved playing strength.
