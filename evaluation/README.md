# A3 CPU evaluation

This package evaluates shared A3 context checkpoints against `rule_ultra` using
seeded mixed tables: 3+1, 2+2 and 1+3 engine assignments. The game and reference
policy are local sources; it does not import or modify the separate A3 project.
Win rates in JSON are fractions and count engine-controlled player appearances,
not independent head-to-head matches.

## Install

From the repository root, install the Python export dependencies in the training
environment and the Node dependencies on the evaluation machine:

```bash
python -m pip install -r evaluation/requirements.txt
cd evaluation
npm ci
npm run check
npm test
```

CPU inference is explicit. The default is two worker processes, one ONNX compute
thread per process and eight concurrent games coalesced into each inference batch.
One native inference call runs at a time per worker. No uncontrolled ORT worker
pool is created in this configuration. Node's game logic and native inference can
overlap, so the process count is not an exact operating-system CPU percentage cap.

## Track a training run

Create an untracked `evaluation/local.json` with:

```json
{
  "ssh_host": "pve",
  "remote_repo": "/path/to/rlcard",
  "remote_python": "/path/to/venv/bin/python",
  "run_dir": "/path/to/run",
  "output_dir": "evaluation/results/tracking"
}
```

Then run from the repository root:

```bash
python scripts/track_training.py --games 2000 --interval 5
```

`--interval` is in minutes. The dashboard is at `http://127.0.0.1:8765`.
Use `--workers 1` for a lower CPU budget, `--port` for another dashboard port,
or `--once --no-dashboard` for one evaluation. Worker count and ONNX thread count
are independently configurable. The tracker uses a lock to prevent overlapping
instances in the same result directory and skips already evaluated model hashes.
It keeps immutable, checksum-verified snapshots and writes JSON atomically.

Exports read one open checkpoint generation and are skipped when the checkpoint
has not changed. Exporting requires a shared `a3-v12-lite-v1` context model;
it validates dynamic-state/candidate ONNX outputs against PyTorch. The evaluator
encodes each state's history once, then gathers its context for legal candidates.
Random streams are scoped to each game, so batching and worker scheduling do not
change the seeded deals. Failed or fallback decisions mark results invalid.

Local model evaluation:

```bash
python -m evaluation.export_model --checkpoint /path/to/model.tar --output evaluation/artifacts/model.onnx
python scripts/track_training.py --model evaluation/artifacts/model.onnx --games 2000 --once --no-dashboard
```

The adjacent JSON manifest is required. `--layout legacy` on the Node CLI is only
for a compatible original three-input V12 ONNX model, and `--legacy-threading`
reproduces its unrestricted runtime pool for performance comparison.

## Finite training jobs

On a Linux CUDA host with MPS installed:

```bash
python scripts/train_job.py --run-dir /path/to/new-run --frames 3000000000
```

The runner uses the measured two-process/two-thread A3 configuration, publishes
checkpoints every five minutes and writes `status.json`, `job.json`, `training.log`
and `logs.csv`. A separate `initial_policy.tar` supports a frame-zero evaluation;
it is not a resumable optimizer checkpoint. The runner refuses to overwrite a
previous run, owns a private MPS controller, and stops that controller on exit.
SIGTERM requests an orderly training stop and checkpoint save. Under systemd,
use `KillMode=mixed` and a sufficiently long stop timeout for this behavior.

## Measured CPU use

On the Windows development machine (Core Ultra 7 258V), with identical model,
seed and 128 games, the original-style seven-process/default-ORT-pool setup took
19.37 s and 126.84 game CPU-seconds. Two processes with one ORT thread took
6.98 s and 13.86 game CPU-seconds. Aggregate game outcomes were identical.

With the same new MLP checkpoint, batching eight games reduced 128-game time
from 2.30 s to 1.45 s, with identical aggregate outcomes. A complete 2,000-game
evaluation took 15.85 s and 32.95 game CPU-seconds (about 2.08 active cores on
average), with no ONNX, legality, timeout or forced-action fallbacks.
CPU-seconds exclude model/session startup. Different-model timings are not used
to claim a same-model speedup. See [measurements.json](measurements.json).
