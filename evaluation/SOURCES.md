# Source provenance

The reference game, card/hand rules, benchmark statistics, context encoding and
`rule_ultra` policy are derived from the user-supplied A3地主 project. Only their
local dependency closure is included under `src/domain`, `src/benchmark` and
`shared/src`. The original project is unchanged. Its server configuration,
credentials, database, web application and model snapshots are not dependencies.

The process scheduler, deterministic async random streams, inference batching,
CPU resource settings, checkpoint exporter, tracker, dashboard and job supervisor
are implemented in this repository. The compact action encoder uses the native
training feature contract for full-house afterstates. Legacy model comparison
retains the original feature behavior.

ONNX Runtime, TypeScript, tsx and their dependencies retain their own licenses.
The repository's existing RLCard and DMC notices remain applicable.
