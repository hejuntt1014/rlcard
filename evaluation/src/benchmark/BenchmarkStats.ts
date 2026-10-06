import type { AIEngineType } from '@a3/shared';
import { TeamSide } from '@a3/shared';
import type { GameResult, DecisionSourceCounts, InvalidPlayRecord } from './BenchmarkRunner.js';

export interface EngineStats {
  engineId: AIEngineType;
  gamesPlayed: number;
  wins: number;
  losses: number;
  draws: number;
  winRate: number;
  declareCount: number;
  declareWins: number;
  declareRate: number;
  declareWinRate: number;
  avgRank: number;
  avgScore: number;
  firstFinishCount: number;
  firstFinishRate: number;
  soloGames: number;
  soloWins: number;
  totalScore: number;
  /** total turns this engine made decisions */
  totalTurns: number;
  /** turns where the engine had an internal ONNX/inference failure and used its own legal fallback */
  onnxFallbackTurns: number;
  /** turns where engine output was invalid and fell back to rule_default */
  fallbackTurns: number;
  /** turns where engine timed out */
  timeoutTurns: number;
  /** turns where engine threw an error */
  errorTurns: number;
  /** turns where a free-play was forced (engine returned pass on new round) */
  forceTurns: number;
}

export interface DuelResult {
  engineA: AIEngineType;
  engineB: AIEngineType;
  statsA: EngineStats;
  statsB: EngineStats;
  totalGames: number;
  errors: number;
}

export interface EloRating {
  engineId: AIEngineType;
  rating: number;
  gamesPlayed: number;
}

export interface EloPairSummary {
  engineA: AIEngineType;
  engineB: AIEngineType;
  actualA: number;
  actualB: number;
  validGames: number;
}

function createEmptyEngineStats(engineId: AIEngineType): EngineStats {
  return {
    engineId,
    gamesPlayed: 0,
    wins: 0,
    losses: 0,
    draws: 0,
    winRate: 0,
    declareCount: 0,
    declareWins: 0,
    declareRate: 0,
    declareWinRate: 0,
    avgRank: 0,
    avgScore: 0,
    firstFinishCount: 0,
    firstFinishRate: 0,
    soloGames: 0,
    soloWins: 0,
    totalScore: 0,
    totalTurns: 0,
    onnxFallbackTurns: 0,
    fallbackTurns: 0,
    timeoutTurns: 0,
    errorTurns: 0,
    forceTurns: 0,
  };
}

export function createStatsMap(engineIds: AIEngineType[]): Map<AIEngineType, EngineStats> {
  const statsMap = new Map<AIEngineType, EngineStats>();

  for (const eid of engineIds) {
    statsMap.set(eid, createEmptyEngineStats(eid));
  }

  return statsMap;
}

export function accumulateStats(
  statsMap: Map<AIEngineType, EngineStats>,
  results: Iterable<GameResult>,
): void {
  for (const result of results) {
    // always merge decision stats (even for error games)
    for (const [engineId, ds] of Object.entries(result.decisionStats ?? {})) {
      const stats = statsMap.get(engineId as AIEngineType);
      if (!stats) continue;
      stats.totalTurns += (ds as DecisionSourceCounts).total;
      stats.onnxFallbackTurns += (ds as DecisionSourceCounts).onnxFallback ?? 0;
      stats.fallbackTurns += (ds as DecisionSourceCounts).fallback;
      stats.timeoutTurns += (ds as DecisionSourceCounts).timeout;
      stats.errorTurns += (ds as DecisionSourceCounts).error;
      stats.forceTurns += (ds as DecisionSourceCounts).force;
    }

    if (result.error) continue;

    if (result.declarerEngineId && result.declarerPlayerId) {
      const declareStats = statsMap.get(result.declarerEngineId);
      if (declareStats) {
        declareStats.declareCount++;
        if ((result.scores[result.declarerPlayerId] ?? 0) > 0) {
          declareStats.declareWins++;
        }
      }
    }

    for (const [playerId, engineId] of Object.entries(result.playerEngines)) {
      const stats = statsMap.get(engineId as AIEngineType);
      if (!stats) continue;

      stats.gamesPlayed++;

      const rank = result.rankings.indexOf(playerId) + 1;
      stats.avgRank += rank;

      const score = result.scores[playerId] ?? 0;
      stats.totalScore += score;

      if (score > 0) stats.wins++;
      else if (score < 0) stats.losses++;
      else stats.draws++;

      if (rank === 1) stats.firstFinishCount++;

      const team = result.playerTeams[playerId];
      if (team === TeamSide.Solo) {
        stats.soloGames++;
        if (score > 0) stats.soloWins++;
      }
    }
  }
}

export function finalizeStats(statsMap: Map<AIEngineType, EngineStats>): Map<AIEngineType, EngineStats> {
  for (const stats of statsMap.values()) {
    if (stats.gamesPlayed > 0) {
      stats.winRate = stats.wins / stats.gamesPlayed;
      stats.declareRate = stats.declareCount / stats.gamesPlayed;
      stats.declareWinRate = stats.declareCount > 0 ? stats.declareWins / stats.declareCount : 0;
      stats.avgRank = stats.avgRank / stats.gamesPlayed;
      stats.avgScore = stats.totalScore / stats.gamesPlayed;
      stats.firstFinishRate = stats.firstFinishCount / stats.gamesPlayed;
    }
  }

  return statsMap;
}

export function mergeRawStats(
  target: Map<AIEngineType, EngineStats>,
  source: Iterable<EngineStats>,
): void {
  for (const incoming of source) {
    const existing = target.get(incoming.engineId);
    if (!existing) {
      target.set(incoming.engineId, {
        engineId: incoming.engineId,
        gamesPlayed: incoming.gamesPlayed,
        wins: incoming.wins,
        losses: incoming.losses,
        draws: incoming.draws,
        winRate: 0,
        declareCount: incoming.declareCount ?? 0,
        declareWins: incoming.declareWins ?? 0,
        declareRate: 0,
        declareWinRate: 0,
        avgRank: incoming.avgRank,
        avgScore: 0,
        firstFinishCount: incoming.firstFinishCount,
        firstFinishRate: 0,
        soloGames: incoming.soloGames,
        soloWins: incoming.soloWins,
        totalScore: incoming.totalScore,
        totalTurns: incoming.totalTurns ?? 0,
        onnxFallbackTurns: incoming.onnxFallbackTurns ?? 0,
        fallbackTurns: incoming.fallbackTurns ?? 0,
        timeoutTurns: incoming.timeoutTurns ?? 0,
        errorTurns: incoming.errorTurns ?? 0,
        forceTurns: incoming.forceTurns ?? 0,
      });
      continue;
    }

    existing.gamesPlayed += incoming.gamesPlayed;
    existing.wins += incoming.wins;
    existing.losses += incoming.losses;
    existing.draws += incoming.draws;
    existing.declareCount += incoming.declareCount ?? 0;
    existing.declareWins += incoming.declareWins ?? 0;
    existing.avgRank += incoming.avgRank;
    existing.firstFinishCount += incoming.firstFinishCount;
    existing.soloGames += incoming.soloGames;
    existing.soloWins += incoming.soloWins;
    existing.totalScore += incoming.totalScore;
    existing.totalTurns += incoming.totalTurns ?? 0;
    existing.onnxFallbackTurns += incoming.onnxFallbackTurns ?? 0;
    existing.fallbackTurns += incoming.fallbackTurns ?? 0;
    existing.timeoutTurns += incoming.timeoutTurns ?? 0;
    existing.errorTurns += incoming.errorTurns ?? 0;
    existing.forceTurns += incoming.forceTurns ?? 0;
  }
}

export function aggregateStats(results: GameResult[], engineIds: AIEngineType[]): Map<AIEngineType, EngineStats> {
  const statsMap = createStatsMap(engineIds);
  accumulateStats(statsMap, results);
  return finalizeStats(statsMap);
}

export function computeDuelResultFromStats(
  engineA: AIEngineType,
  engineB: AIEngineType,
  statsMap: Map<AIEngineType, EngineStats>,
  totalGames: number,
  errors: number,
): DuelResult {
  return {
    engineA,
    engineB,
    statsA: statsMap.get(engineA) ?? createEmptyEngineStats(engineA),
    statsB: statsMap.get(engineB) ?? createEmptyEngineStats(engineB),
    totalGames,
    errors,
  };
}

export function computeDuelResult(
  results: GameResult[],
  engineA: AIEngineType,
  engineB: AIEngineType,
): DuelResult {
  const statsMap = aggregateStats(results, [engineA, engineB]);
  return computeDuelResultFromStats(
    engineA,
    engineB,
    statsMap,
    results.length,
    results.filter((r) => r.error).length,
  );
}

export function summarizePairForElo(
  results: Iterable<GameResult>,
  engineA: AIEngineType,
  engineB: AIEngineType,
): EloPairSummary {
  let scoreA = 0;
  let scoreB = 0;
  let validGames = 0;

  for (const result of results) {
    if (result.error) continue;

    let aTotal = 0;
    let bTotal = 0;
    let aCount = 0;
    let bCount = 0;

    for (const [playerId, engineId] of Object.entries(result.playerEngines)) {
      const gameScore = result.scores[playerId] ?? 0;
      if (engineId === engineA) {
        aTotal += gameScore;
        aCount++;
      } else if (engineId === engineB) {
        bTotal += gameScore;
        bCount++;
      }
    }

    if (aCount === 0 || bCount === 0) continue;
    validGames++;

    const aAvg = aTotal / aCount;
    const bAvg = bTotal / bCount;

    if (aAvg > bAvg) scoreA++;
    else if (bAvg > aAvg) scoreB++;
    else {
      scoreA += 0.5;
      scoreB += 0.5;
    }
  }

  return {
    engineA,
    engineB,
    actualA: validGames > 0 ? scoreA / validGames : 0.5,
    actualB: validGames > 0 ? scoreB / validGames : 0.5,
    validGames,
  };
}

export function computeEloRatingsFromSummaries(
  pairSummaries: EloPairSummary[],
): EloRating[] {
  const K = 32;
  const ratings = new Map<AIEngineType, { rating: number; games: number }>();

  for (const pair of pairSummaries) {
    if (!ratings.has(pair.engineA)) ratings.set(pair.engineA, { rating: 1500, games: 0 });
    if (!ratings.has(pair.engineB)) ratings.set(pair.engineB, { rating: 1500, games: 0 });
  }

  const ROUNDS = 10;
  for (let round = 0; round < ROUNDS; round++) {
    for (const pair of pairSummaries) {
      if (pair.validGames === 0) continue;

      const rA = ratings.get(pair.engineA)!;
      const rB = ratings.get(pair.engineB)!;

      const expectedA = 1 / (1 + Math.pow(10, (rB.rating - rA.rating) / 400));
      const expectedB = 1 - expectedA;

      rA.rating += K * (pair.actualA - expectedA);
      rB.rating += K * (pair.actualB - expectedB);
      rA.games += pair.validGames;
      rB.games += pair.validGames;
    }
  }

  return [...ratings.entries()]
    .map(([engineId, data]) => ({
      engineId,
      rating: Math.round(data.rating),
      gamesPlayed: data.games / ROUNDS,
    }))
    .sort((a, b) => b.rating - a.rating);
}

export function computeEloRatings(
  pairResults: Array<{ engineA: AIEngineType; engineB: AIEngineType; results: GameResult[] }>,
): EloRating[] {
  return computeEloRatingsFromSummaries(
    pairResults.map((pair) => summarizePairForElo(pair.results, pair.engineA, pair.engineB)),
  );
}

function formatSoloLine(stats: EngineStats, w: number): string {
  if (stats.soloGames === 0) return padRight(`    独食: 未触发`, w);
  const soloWinRate = stats.soloGames > 0 ? pct(stats.soloWins / stats.soloGames) : '-';
  const soloLosses = stats.soloGames - stats.soloWins;
  return padRight(`    独食: ${stats.soloGames}局  ${stats.soloWins}胜${soloLosses}负  独食胜率: ${soloWinRate}`, w);
}

function formatDeclareLine(stats: EngineStats, w: number): string {
  if (stats.gamesPlayed === 0) return padRight(`    报牌: 无数据`, w);
  if (stats.declareCount === 0) return padRight(`    报牌: 0/${stats.gamesPlayed} (${pct(0)})  报牌胜率: -`, w);
  return padRight(
    `    报牌: ${stats.declareCount}/${stats.gamesPlayed} (${pct(stats.declareRate)})  报牌胜率: ${pct(stats.declareWinRate)}`,
    w,
  );
}

function formatDeclareCell(stats: EngineStats): string {
  if (stats.gamesPlayed === 0) return 'N/A';
  const winRate = stats.declareCount > 0 ? pct(stats.declareWinRate) : '-';
  return `${stats.declareWins}/${stats.declareCount} ${pct(stats.declareRate)} ${winRate}`;
}

function formatDecisionLine(stats: EngineStats, w: number): string {
  const t = stats.totalTurns;
  if (t === 0) return padRight(`    决策: 无数据`, w);
  const abnormal = stats.fallbackTurns + stats.timeoutTurns + stats.errorTurns + stats.forceTurns;
  const parts: string[] = [];
  if (stats.onnxFallbackTurns > 0) parts.push(`ONNX回退${stats.onnxFallbackTurns}次`);
  if (stats.fallbackTurns > 0) parts.push(`非法出牌回退${stats.fallbackTurns}次`);
  if (stats.timeoutTurns > 0) parts.push(`超时${stats.timeoutTurns}次`);
  if (stats.errorTurns > 0) parts.push(`异常${stats.errorTurns}次`);
  if (stats.forceTurns > 0) parts.push(`强制出牌${stats.forceTurns}次`);
  if (parts.length === 0) return padRight(`    决策: ${t}回合  ✓ 全部正常`, w);
  const suffix = abnormal > 0
    ? ` (异常率${pct(abnormal / t)})`
    : (stats.onnxFallbackTurns > 0 ? ` (ONNX回退率${pct(stats.onnxFallbackTurns / t)})` : '');
  return padRight(`    决策: ${t}回合  ⚠ ${parts.join(' ')}${suffix}`, w);
}

export function formatDuelReport(duel: DuelResult): string {
  const lines: string[] = [];
  const w = 74;

  lines.push('╔' + '═'.repeat(w) + '╗');
  lines.push('║' + center(`Duel: ${duel.engineA} vs ${duel.engineB}`, w) + '║');
  lines.push('║' + center(`${duel.totalGames} games (${duel.errors} errors)`, w) + '║');
  lines.push('╠' + '═'.repeat(w) + '╣');

  const a = duel.statsA;
  const b = duel.statsB;

  lines.push('║' + padRight(`  ${a.engineId}`, w) + '║');
  lines.push('║' + padRight(`    胜率: ${pct(a.winRate)}  (${a.wins}胜 ${a.losses}负 ${a.draws}平)`, w) + '║');
  lines.push('║' + padRight(`    平均排名: ${a.avgRank.toFixed(2)}  平均得分: ${a.avgScore.toFixed(1)}`, w) + '║');
  lines.push('║' + padRight(`    第1名: ${a.firstFinishCount}/${a.gamesPlayed} (${pct(a.firstFinishRate)})`, w) + '║');
  lines.push('║' + formatSoloLine(a, w) + '║');
  lines.push('║' + formatDeclareLine(a, w) + '║');
  lines.push('║' + formatDecisionLine(a, w) + '║');

  lines.push('║' + ' '.repeat(w) + '║');

  lines.push('║' + padRight(`  ${b.engineId}`, w) + '║');
  lines.push('║' + padRight(`    胜率: ${pct(b.winRate)}  (${b.wins}胜 ${b.losses}负 ${b.draws}平)`, w) + '║');
  lines.push('║' + padRight(`    平均排名: ${b.avgRank.toFixed(2)}  平均得分: ${b.avgScore.toFixed(1)}`, w) + '║');
  lines.push('║' + padRight(`    第1名: ${b.firstFinishCount}/${b.gamesPlayed} (${pct(b.firstFinishRate)})`, w) + '║');
  lines.push('║' + formatSoloLine(b, w) + '║');
  lines.push('║' + formatDeclareLine(b, w) + '║');
  lines.push('║' + formatDecisionLine(b, w) + '║');

  lines.push('╚' + '═'.repeat(w) + '╝');
  return lines.join('\n');
}

export function formatTournamentReport(
  eloRatings: EloRating[],
  allStats: Map<AIEngineType, EngineStats>,
): string {
  const lines: string[] = [];
  const totalWidth = 152;

  lines.push('');
  lines.push('🏆 Tournament Elo Rankings');
  lines.push('─'.repeat(totalWidth));
  lines.push(
    padRight(' #', 4) +
    padRight('Engine', 28) +
    padRight('Elo', 8) +
    padRight('WinRate', 10) +
    padRight('AvgRank', 10) +
    padRight('1st', 16) +
    padRight('独食(胜/总 胜率)', 18) +
    padRight('报牌(胜/总 报牌率 胜率)', 28) +
    padRight('决策问题(ONNX/非法/超时/异常/强制)', 40),
  );
  lines.push('─'.repeat(totalWidth));

  for (let i = 0; i < eloRatings.length; i++) {
    const elo = eloRatings[i]!;
    const stats = allStats.get(elo.engineId);
    const wr = stats ? pct(stats.winRate) : 'N/A';
    const ar = stats ? stats.avgRank.toFixed(2) : 'N/A';
    const first = stats ? `${stats.firstFinishCount} (${pct(stats.firstFinishRate)})` : 'N/A';
    const solo = stats && stats.soloGames > 0
      ? `${stats.soloWins}/${stats.soloGames} ${pct(stats.soloWins / stats.soloGames)}`
      : (stats ? '-' : 'N/A');
    const declare = stats ? formatDeclareCell(stats) : 'N/A';
    let decisionCol = 'N/A';
    if (stats) {
      const abnormal = stats.fallbackTurns + stats.timeoutTurns + stats.errorTurns + stats.forceTurns;
      if (abnormal === 0 && stats.onnxFallbackTurns === 0) {
        decisionCol = '✓';
      } else {
        decisionCol = `${stats.onnxFallbackTurns}/${stats.fallbackTurns}/${stats.timeoutTurns}/${stats.errorTurns}/${stats.forceTurns}`;
        if (stats.totalTurns > 0 && abnormal > 0) {
          decisionCol += ` (${pct(abnormal / stats.totalTurns)})`;
        } else if (stats.totalTurns > 0 && stats.onnxFallbackTurns > 0) {
          decisionCol += ` (${pct(stats.onnxFallbackTurns / stats.totalTurns)})`;
        }
      }
    }
    lines.push(
      padRight(` ${i + 1}`, 4) +
      padRight(elo.engineId, 28) +
      padRight(String(elo.rating), 8) +
      padRight(wr, 10) +
      padRight(ar, 10) +
      padRight(first, 16) +
      padRight(solo, 18) +
      padRight(declare, 28) +
      padRight(decisionCol, 40),
    );
  }

  lines.push('─'.repeat(totalWidth));
  return lines.join('\n');
}

export function formatFFAReport(stats: Map<AIEngineType, EngineStats>): string {
  const lines: string[] = [];
  const sorted = [...stats.values()].sort((a, b) => b.avgScore - a.avgScore);

  lines.push('');
  lines.push('🎮 FFA Results');
  lines.push('─'.repeat(150));
  lines.push(
    padRight(' #', 4) +
    padRight('Engine', 28) +
    padRight('WinRate', 10) +
    padRight('AvgRank', 10) +
    padRight('AvgScore', 10) +
    padRight('1st', 24) +
    padRight('报牌(胜/总 报牌率 胜率)', 28) +
    padRight('决策问题(ONNX/非法/超时/异常/强制)', 40),
  );
  lines.push('─'.repeat(150));

  for (let i = 0; i < sorted.length; i++) {
    const s = sorted[i]!;
    const abnormal = s.fallbackTurns + s.timeoutTurns + s.errorTurns + s.forceTurns;
    let decisionCol: string;
    if (abnormal === 0 && s.onnxFallbackTurns === 0) {
      decisionCol = '✓';
    } else {
      decisionCol = `${s.onnxFallbackTurns}/${s.fallbackTurns}/${s.timeoutTurns}/${s.errorTurns}/${s.forceTurns}`;
      if (s.totalTurns > 0 && abnormal > 0) decisionCol += ` (${pct(abnormal / s.totalTurns)})`;
      else if (s.totalTurns > 0 && s.onnxFallbackTurns > 0) decisionCol += ` (${pct(s.onnxFallbackTurns / s.totalTurns)})`;
    }
    lines.push(
      padRight(` ${i + 1}`, 4) +
      padRight(s.engineId, 28) +
      padRight(pct(s.winRate), 10) +
      padRight(s.avgRank.toFixed(2), 10) +
      padRight(s.avgScore.toFixed(1), 10) +
      padRight(`${s.firstFinishCount}/${s.gamesPlayed} (${pct(s.firstFinishRate)})`, 24) +
      padRight(formatDeclareCell(s), 28) +
      padRight(decisionCol, 40),
    );
  }

  lines.push('─'.repeat(150));
  return lines.join('\n');
}

function pct(v: number): string {
  return (v * 100).toFixed(1) + '%';
}

/** 计算字符串在终端中的显示宽度（中文等宽字符占 2 列）*/
function displayWidth(s: string): number {
  let width = 0;
  for (const char of s) {
    const cp = char.codePointAt(0) ?? 0;
    if (
      (cp >= 0x1100 && cp <= 0x115F) ||   // Hangul Jamo
      (cp >= 0x2E80 && cp <= 0x303E) ||   // CJK 部首、符号
      (cp >= 0x3041 && cp <= 0x33BF) ||   // 日文假名、CJK
      (cp >= 0x33FF && cp <= 0xA4CF) ||   // CJK 扩展
      (cp >= 0xA960 && cp <= 0xA97F) ||   // Hangul
      (cp >= 0xAC00 && cp <= 0xD7FF) ||   // 韩文音节
      (cp >= 0xF900 && cp <= 0xFAFF) ||   // CJK 兼容
      (cp >= 0xFE10 && cp <= 0xFE6F) ||   // 竖排形式、小形式
      (cp >= 0xFF01 && cp <= 0xFF60) ||   // 全角字母
      (cp >= 0xFFE0 && cp <= 0xFFE6) ||   // 全角符号
      (cp >= 0x1F300 && cp <= 0x1F9FF) || // Emoji
      (cp >= 0x20000 && cp <= 0x3FFFD)    // CJK 扩展 B+
    ) {
      width += 2;
    } else {
      width += 1;
    }
  }
  return width;
}

function center(s: string, w: number): string {
  const pad = Math.max(0, w - displayWidth(s));
  const left = Math.floor(pad / 2);
  return ' '.repeat(left) + s + ' '.repeat(pad - left);
}

function padRight(s: string, w: number): string {
  const dw = displayWidth(s);
  return dw >= w ? s : s + ' '.repeat(w - dw);
}

// ─── Invalid Play Analysis ────────────────────────────────────────────────────

function cardStr(c: { suit: string; rank: string }): string {
  const s: Record<string, string> = { spade: '♠', heart: '♥', club: '♣', diamond: '♦' };
  return `${s[c.suit] ?? c.suit[0]}${c.rank}`;
}

function cardsStr(cards: { suit: string; rank: string }[]): string {
  return cards.map(cardStr).join(' ');
}

/**
 * Collect all invalid play records from a batch of results,
 * filtered to the specified engine (or all engines if null).
 */
export function collectInvalidPlays(
  results: GameResult[],
  engineId: AIEngineType | null = null,
): InvalidPlayRecord[] {
  const all: InvalidPlayRecord[] = [];
  for (const r of results) {
    if (!r.invalidPlays) continue;
    for (const ip of r.invalidPlays) {
      if (engineId === null || ip.engineId === engineId) all.push(ip);
    }
  }
  return all;
}

/**
 * Format a detailed analysis report of invalid play attempts.
 * Groups by validation error type, then shows individual examples.
 */
export function formatInvalidPlayReport(
  records: InvalidPlayRecord[],
  engineId: AIEngineType | null,
  maxExamplesPerGroup = 5,
): string {
  const lines: string[] = [];
  const title = engineId ? `Invalid Play Analysis: ${engineId}` : 'Invalid Play Analysis (all engines)';
  lines.push('');
  lines.push(`🔍 ${title}`);
  lines.push(`   共 ${records.length} 次非法出牌`);

  if (records.length === 0) {
    lines.push('   ✅ 无非法出牌记录');
    return lines.join('\n');
  }

  // Group by validation error
  const byError = new Map<string, InvalidPlayRecord[]>();
  for (const r of records) {
    const g = byError.get(r.validationError) ?? [];
    g.push(r);
    byError.set(r.validationError, g);
  }

  lines.push('');
  lines.push('  ┌─ 按错误类型统计 ─────────────────────────────────────┐');
  for (const [err, grp] of [...byError.entries()].sort((a, b) => b[1].length - a[1].length)) {
    lines.push(`  │  ${padRight(err, 30)} ${grp.length} 次`);
  }
  lines.push('  └──────────────────────────────────────────────────────┘');

  // For each error group, show examples grouped by context
  for (const [err, grp] of [...byError.entries()].sort((a, b) => b[1].length - a[1].length)) {
    lines.push('');
    lines.push(`  ══ "${err}" (${grp.length}次) ══`);

    // Sub-group: free-play vs follow-play
    const freePlays = grp.filter(r => r.isNewRound || r.isFirstTurn);
    const followPlays = grp.filter(r => !r.isNewRound && !r.isFirstTurn);

    if (freePlays.length > 0) {
      lines.push(`  ── 自由出牌时 (${freePlays.length}次) ─────────────────────`);
      for (const r of freePlays.slice(0, maxExamplesPerGroup)) {
        lines.push(`    Turn ${r.turn}${r.isFirstTurn ? ' [首手]' : ' [新轮]'}`);
        lines.push(`      手牌(${r.playerHand.length}): ${cardsStr(r.playerHand)}`);
        lines.push(`      尝试打出: ${cardsStr(r.attempted)}`);
      }
      if (freePlays.length > maxExamplesPerGroup) {
        lines.push(`      ... 还有 ${freePlays.length - maxExamplesPerGroup} 例`);
      }
    }

    if (followPlays.length > 0) {
      lines.push(`  ── 跟牌时 (${followPlays.length}次) ─────────────────────────`);
      for (const r of followPlays.slice(0, maxExamplesPerGroup)) {
        lines.push(`    Turn ${r.turn}`);
        lines.push(`      手牌(${r.playerHand.length}): ${cardsStr(r.playerHand)}`);
        lines.push(`      上家: ${r.lastPlayCards ? cardsStr(r.lastPlayCards) : '?'} (${r.lastPlayHandType ?? '?'})`);
        lines.push(`      尝试打出: ${cardsStr(r.attempted)}`);
      }
      if (followPlays.length > maxExamplesPerGroup) {
        lines.push(`      ... 还有 ${followPlays.length - maxExamplesPerGroup} 例`);
      }
    }
  }

  return lines.join('\n');
}
