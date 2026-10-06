import RuleRLBrain, { initializeModel } from './domain/game/engines/rules/RuleRLBrain.js';
import RuleUltraBrain from './domain/game/engines/rules/RuleUltraBrain.js';
import { BenchmarkRunner } from './benchmark/BenchmarkRunner.js';
import { createStatsMap, accumulateStats } from './benchmark/BenchmarkStats.js';
import type { AIEngineType } from '@a3/shared';
import type { BotBrain } from './domain/game/BotBrain.js';
import { seededRandom } from './runtime.js';
import { AsyncLocalStorage } from 'node:async_hooks';

const random = new AsyncLocalStorage<() => number>();
const originalRandom = Math.random;
Math.random = () => (random.getStore() ?? originalRandom)();

const ids: AIEngineType[] = ['rule_rl', 'rule_ultra'];
const engines = new Map<AIEngineType, BotBrain>([['rule_rl', new RuleRLBrain()], ['rule_ultra', new RuleUltraBrain()]]);
const runner = new BenchmarkRunner(engines, false);
let busy = false;
process.on('message', async (message: any) => {
  if (message.type !== 'batch') return;
  if (busy) { process.send?.({ type: 'error', error: 'Overlapping evaluation batches' }); return; }
  busy = true;
  const cpu = process.cpuUsage();
  const stats = createStatsMap(ids);
  try {
    const indices: number[] = message.indices;
    const concurrency = process.env.A3_EVAL_LAYOUT === 'legacy' ? 1 : Number(process.env.A3_EVAL_BATCH_GAMES ?? 8);
    const results = new Map<number, Awaited<ReturnType<BenchmarkRunner['runGame']>>>();
    for (let start = 0; start < indices.length; start += concurrency) {
      await Promise.all(indices.slice(start, start+concurrency).map(index => random.run(seededRandom((message.seed + index) >>> 0), async () => {
      const rlSeats = 3 - index % 3;
      const seats: AIEngineType[] = Array.from({ length: 4 }, (_, i) => i < rlSeats ? 'rule_rl' : 'rule_ultra');
      for (let j = 3; j > 0; --j) {
        const k = Math.floor(Math.random() * (j + 1));
        [seats[j], seats[k]] = [seats[k]!, seats[j]!];
      }
      const result = await runner.runGame(seats.map((engineId, seatIndex) => ({ engineId, seatIndex })));
      if (result.error) throw new Error(`Game ${index}: ${result.error}`);
      results.set(index,result);
      })));
    }
    // Aggregate in game order, independent of promise/worker completion order.
    accumulateStats(stats, indices.map(index => results.get(index)!));
    const used = process.cpuUsage(cpu);
    process.send?.({ type: 'done', job: message.job, games: message.indices.length,
      stats: [...stats.values()], cpuSeconds: (used.user + used.system) / 1e6 });
  } catch (error) {
    process.send?.({ type: 'error', error: String(error), job: message.job });
  } finally { busy = false; }
});
initializeModel().then(() => process.send?.({ type: 'ready' })).catch(error => {
  process.send?.({ type: 'error', error: String(error) });
  process.exitCode = 1;
});
