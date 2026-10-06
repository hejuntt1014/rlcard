import { fork, type ChildProcess } from 'node:child_process';
import { availableParallelism } from 'node:os';
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';
import { createHash } from 'node:crypto';
import { createStatsMap, mergeRawStats, finalizeStats } from './benchmark/BenchmarkStats.js';
import { positiveInteger } from './runtime.js';

const args = process.argv.slice(2);
function option(name: string): string | undefined { const at = args.indexOf(name); return at < 0 ? undefined : args[at+1]; }
if (args.includes('--help')) {
  console.log('tsx src/cli.ts --model FILE --games 2000 --workers 2 --onnx-threads 1 --seed 42 --output FILE [--layout compact|legacy]');
  process.exit(0);
}
const model = resolve(option('--model') ?? 'artifacts/model.onnx');
const games = positiveInteger(option('--games'), 2000, 'games');
const workers = Math.min(games, positiveInteger(option('--workers'), Math.min(2, availableParallelism()), 'workers'));
const threads = positiveInteger(option('--onnx-threads'), 1, 'onnx-threads');
const batchGames = positiveInteger(option('--batch-games'), 8, 'batch-games');
const seed = Number(option('--seed') ?? 42);
if (!Number.isSafeInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error('seed must be uint32');
const layout = option('--layout') ?? 'compact';
if (!['compact', 'legacy'].includes(layout)) throw new Error('layout must be compact or legacy');
const output = resolve(option('--output') ?? 'results/evaluation.json');
const before = performance.now();
const processes: ChildProcess[] = [];
const stats = createStatsMap(['rule_rl', 'rule_ultra']);
let next = 0, completed = 0, cpuSeconds = 0;
const progress = { nextPrint: 100 };
const legacyThreading = args.includes('--legacy-threading');
const env = { ...process.env, A3_EVAL_MODEL: model, A3_EVAL_LAYOUT: layout,
  A3_EVAL_THREADS: String(threads), A3_EVAL_LEGACY_THREADS: legacyThreading ? '1' : '0',
  A3_EVAL_BATCH_GAMES: String(batchGames),
  A3_EVAL_STRICT: '1', OMP_NUM_THREADS: String(threads), MKL_NUM_THREADS: String(threads) };
const workerPath = fileURLToPath(new URL('./worker.ts', import.meta.url));

async function runWorker(): Promise<void> {
  const proc = fork(workerPath, [], { execArgv: ['--import', 'tsx'], env,
    stdio: ['ignore', 'ignore', 'inherit', 'ipc'] });
  processes.push(proc);
  await new Promise<void>((resolveJob, reject) => {
    let expected = -1;
    let timer: NodeJS.Timeout;
    const arm = () => { clearTimeout(timer); timer = setTimeout(() => reject(new Error('Evaluation worker timed out')), 120000); };
    const dispatch = () => {
      if (next >= games) { clearTimeout(timer); resolveJob(); return; }
      const start = next; next = Math.min(games, start + 16); expected = start;
      arm(); proc.send({ type: 'batch', job: start, indices: Array.from({ length: next-start }, (_, i) => start+i), seed });
    };
    arm();
    proc.on('error', error => { clearTimeout(timer); reject(error); });
    proc.on('exit', code => { clearTimeout(timer); if (completed < games) reject(new Error(`Worker exited: ${code}`)); });
    proc.on('message', (message: any) => {
      if (message.type === 'error') { clearTimeout(timer); reject(new Error(message.error)); return; }
      if (message.type === 'ready') { dispatch(); return; }
      if (message.type === 'done') {
        if (message.job !== expected) { reject(new Error('Unexpected worker response')); return; }
        mergeRawStats(stats, message.stats);
        completed += message.games; cpuSeconds += message.cpuSeconds;
        if (completed >= progress.nextPrint) { console.log(`${completed}/${games} games`); progress.nextPrint += 100; }
        dispatch();
      }
    });
  });
}

try {
  // Fail before spawning when the model is missing.
  const modelBytes = readFileSync(model, { flag: 'r' });
  if (layout === 'compact') {
    const manifest = JSON.parse(readFileSync(model.replace(/\.onnx$/, '.json'), 'utf8'));
    if (manifest.layout !== 'batched-context-v1' || manifest.model_spec?.feature_schema !== 'a3-v12-lite-v1')
      throw new Error('Unsupported evaluation model schema');
    if (manifest.sha256 !== createHash('sha256').update(modelBytes).digest('hex'))
      throw new Error('Evaluation model does not match its manifest');
  }
  await Promise.all(Array.from({ length: workers }, runWorker));
  const engines = Object.fromEntries(finalizeStats(stats));
  const invalid = Object.values(engines).some(s => s.onnxFallbackTurns || s.fallbackTurns || s.timeoutTurns || s.errorTurns || s.forceTurns);
  const seconds = (performance.now() - before) / 1000;
  const result = { protocol: 'seeded-mix-standard-v1', games: completed, seed, model, layout,
    nodeVersion: process.version, platform: process.platform,
    workers, batchGames: layout === 'legacy' ? 1 : batchGames, onnxThreads: legacyThreading ? 'runtime-default' : threads, legacyThreading,
    wallSeconds: seconds, gameCpuSeconds: cpuSeconds, averageGameCpuCores: cpuSeconds / seconds,
    valid: completed === games && !invalid, engines };
  mkdirSync(dirname(output), { recursive: true });
  writeFileSync(output, JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result, null, 2));
  if (!result.valid) process.exitCode = 2;
} finally {
  for (const proc of processes) proc.kill();
}
