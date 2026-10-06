# A3 context implementation review

This review covers the native sampling path, model and learning signals, and
training/evaluation correctness. Findings were cross-checked across these three
areas before integration.

| Area | Included behavior | Evidence |
| --- | --- | --- |
| State reuse | Unique observations, candidate offsets, state encoding once per decision | Expanded/compact outputs and gradients agree; CUDA timing fixture |
| Actor work | Forced and exploratory choices skip the value network while retaining samples | Tests reject unexpected forwards on those decisions |
| CPU concurrency | Worker-local policy copies; locks only for versioned snapshot refresh | Spawn training and recovery tests |
| Experience transport | Block writes and batch-major reads without the second full-batch transpose copy | State/action/target/label ordering tests |
| GPU policy publication | Each unique learner policy copied to CPU once before per-device publication | CPU/GPU actor short training |
| History | Chronological MLP; configurable effective window; safe optional masked Transformer | Padding, empty history, order and real early-game window tests |
| Decision branches | Declaration and play branches score only their relevant actions | Branch-execution and mixed-gradient tests |
| Labels | Decision-time public labels, not one terminal label repeated across all decisions | Declaration transition regression and native/single-engine alignment |
| Rewards | Original game score by default; optional shaping recognizes both anti-declarant partners | Step-reward regressions and game/shaped comparisons |
| Rules | Fixed rules persist across resets and are part of checkpoint validation | Native/Python-pool reset and incompatible-resume tests |
| Legal policy actions | Greedy policies cannot pass when a last card must beat | Explicit-deal regression, portable across standard libraries |
| Card features | Correct full-house and multi-suit straight-flush summaries | Deterministic pattern fixtures |
| Rule-aware classification | Cached and uncached keys obey the same straight range | Full game and boundary fixtures across all eight variants |
| Native isolation | Separate C++ namespaces and Python-local types | Cross-module engine objects rejected with TypeError |
| Evaluation | Fixed deals, seat rotation, raw payoffs, paired statistics and failure traces | Checkpoint/schema and complete-game tests |

## Experiments kept optional

Short validation does not identify a stronger 16- or 24-step policy. FP32 remains
the default; BF16 and FP16 need deployment-specific checks. Auxiliary public-label
reconstruction and shaped rewards have explicit ablation switches. The Transformer
is a comparison model, not an assumed improvement.

Pinned staging buffers and asynchronous transfer overlap require separate buffer
lifetime tests and profiling. They are not enabled merely by adding a
`non_blocking` argument. Stronger-opponent evaluation, multiple training seeds and
equal-GPU-hour comparisons are necessary before making playing-strength claims.

See [model usage](a3-context.md) and [measured validation](benchmarks/context-validation.md).
