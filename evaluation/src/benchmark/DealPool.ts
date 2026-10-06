/**
 * 预生成牌局池（Fixed Deal Pool）
 *
 * 原理：预先生成 N 个发牌方案，循环赛中所有引擎对使用相同牌局。
 * 效果：
 *   - 消除发牌运气对排名的影响（Common Random Numbers 方差消减技术）
 *   - 每局打两次（A先坐♠A3队 / B先坐♠A3队），消除先手偏差
 *   - 100% 纯对决（不再有混杂队伍局），统计效率从 1/3 提升到 100%
 *
 * 牌局池大小建议：
 *   - 快速测试：50 局（实际跑 100 局/对）
 *   - 标准评估：200 局（实际跑 400 局/对）
 *   - 精准排名：500 局（实际跑 1000 局/对）
 */

import type { Card, AIEngineType } from '@a3/shared';
import { dealCards } from '../domain/card/index.js';
import { containsSpadeThree, containsSpadeAce } from '../domain/card/index.js';
import type { EngineAssignment } from './BenchmarkRunner.js';

export interface PooledDeal {
  hands: [Card[], Card[], Card[], Card[]];
  /** 普通2v2：♠3 和 ♠A 分别在哪个座位 */
  a3Seats: [number, number] | null;
  /** 独食1v3：同时持有♠3+♠A 的座位 */
  soloSeat: number | null;
}

export interface GameBatchItem {
  assignments: EngineAssignment[];
  /** 预生成手牌（按座位索引 0-3 对应） */
  hands?: [Card[], Card[], Card[], Card[]];
}

/**
 * 预生成 count 个发牌方案，记录每局的 a3 队座位信息
 */
export function generateDealPool(count: number): PooledDeal[] {
  const pool: PooledDeal[] = [];
  for (let i = 0; i < count; i++) {
    const { hands } = dealCards();

    let spade3Seat = -1;
    let spadeASeat = -1;
    for (let s = 0; s < 4; s++) {
      if (containsSpadeThree(hands[s]!)) spade3Seat = s;
      if (containsSpadeAce(hands[s]!)) spadeASeat = s;
    }

    const isSolo = spade3Seat >= 0 && spade3Seat === spadeASeat;
    pool.push({
      hands,
      a3Seats: isSolo ? null : ([spade3Seat, spadeASeat] as [number, number]),
      soloSeat: isSolo ? spade3Seat : null,
    });
  }
  return pool;
}

/**
 * 从预生成牌局池生成 duel 对战批次：
 *   - 普通2v2局：每局打两次（A在a3队一次、B在a3队一次），彻底消除先手偏差
 *   - 独食1v3局：交替让 engineA / engineB 做独食者
 *
 * 结果：100% 纯引擎对决，发牌完全相同，只有策略不同。
 */
export function generateDuelFromPool(
  engineA: AIEngineType,
  engineB: AIEngineType,
  pool: PooledDeal[],
): GameBatchItem[] {
  const games: GameBatchItem[] = [];

  for (const deal of pool) {
    if (deal.soloSeat !== null) {
      // 独食局：A做独食者 + B做独食者，各一局
      games.push({
        assignments: buildSoloAssignments(deal.soloSeat, engineA, engineB),
        hands: deal.hands,
      });
      games.push({
        assignments: buildSoloAssignments(deal.soloSeat, engineB, engineA),
        hands: deal.hands,
      });
    } else {
      const [s3, sA] = deal.a3Seats!;
      const a3Set = new Set([s3, sA]);
      const opSeats = ([0, 1, 2, 3] as const).filter(s => !a3Set.has(s));

      // Round 1：engineA 坐 a3 队（拿♠3+♠A 的两个座位）
      games.push({
        assignments: [
          { seatIndex: s3, engineId: engineA },
          { seatIndex: sA, engineId: engineA },
          { seatIndex: opSeats[0]!, engineId: engineB },
          { seatIndex: opSeats[1]!, engineId: engineB },
        ],
        hands: deal.hands,
      });

      // Round 2：engineB 坐 a3 队，同一手牌
      games.push({
        assignments: [
          { seatIndex: s3, engineId: engineB },
          { seatIndex: sA, engineId: engineB },
          { seatIndex: opSeats[0]!, engineId: engineA },
          { seatIndex: opSeats[1]!, engineId: engineA },
        ],
        hands: deal.hands,
      });
    }
  }

  return games;
}

/**
 * 将普通 EngineAssignment[][] 转换为 GameBatchItem[]（无预设手牌）
 */
export function wrapAssignments(assignments: EngineAssignment[][]): GameBatchItem[] {
  return assignments.map(a => ({ assignments: a }));
}

function buildSoloAssignments(
  soloSeat: number,
  soloEngine: AIEngineType,
  opponentEngine: AIEngineType,
): EngineAssignment[] {
  return [0, 1, 2, 3].map(seat => ({
    seatIndex: seat,
    engineId: seat === soloSeat ? soloEngine : opponentEngine,
  }));
}
