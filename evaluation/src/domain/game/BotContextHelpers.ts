/**
 * BotContext 构建辅助函数 — BotController 与 BenchmarkRunner 共用
 *
 * 将关键的、必须保持一致的逻辑抽到此处，避免两边各自维护导致漂移。
 */

import { TeamSide, type Card } from '@a3/shared';
import { isSpadeThree, isSpadeAce } from '../card/index.js';
import type { ActionRecord } from './BotBrain.js';

interface RawAction {
  playerId: string;
  action: 'play' | 'pass';
  cards: Card[];
  handType: string;
}

/**
 * 从完整的出牌历史构建：
 *   - playedCardsByPlayer: 各玩家已出的牌
 *   - observedTeams: 可观测的队伍信息（♠3/♠A 暴露机制 + 报牌公开信息）
 *   - playedOutCards: 所有已打出的牌
 *
 * 与 Python 训练环境 GameState._compute_observed_teams() 完全对齐：
 *   - 报牌局：报牌者 SOLO，其余 OPPONENT（报牌是公开信息）
 *   - 同一人打出♠3和♠A → SOLO，其余全是 OPPONENT
 *   - 不同人各打一张   → 两人 SPADE_A3，其余 OPPONENT
 *   - 仅打出♠3 或 ♠A  → 该人 SPADE_A3，其余 UNKNOWN
 *   - 都未打出         → 全部 UNKNOWN
 */
export function buildPlayedCardsAndTeams(
  actionHistory: RawAction[],
  seatOrder: string[],
  declarerId?: string | null,
): {
  playedCardsByPlayer: Record<string, Card[]>;
  observedTeams: Record<string, TeamSide>;
  playedOutCards: Card[];
} {
  const playedCardsByPlayer: Record<string, Card[]> = {};
  const playedOutCards: Card[] = [];

  for (const pid of seatOrder) {
    playedCardsByPlayer[pid] = [];
  }

  let spade3Player: string | null = null;
  let spadeAPlayer: string | null = null;

  for (const a of actionHistory) {
    if (a.action === 'play' && a.cards) {
      playedOutCards.push(...a.cards);
      if (playedCardsByPlayer[a.playerId]) {
        playedCardsByPlayer[a.playerId]!.push(...a.cards);
      }
      for (const c of a.cards) {
        if (isSpadeThree(c)) spade3Player = a.playerId;
        if (isSpadeAce(c)) spadeAPlayer = a.playerId;
      }
    }
  }

  const observedTeams: Record<string, TeamSide> = {};

  if (declarerId) {
    for (const pid of seatOrder) {
      observedTeams[pid] = pid === declarerId ? TeamSide.Solo : TeamSide.Opponent;
    }
  } else if (spade3Player !== null && spadeAPlayer !== null) {
    if (spade3Player === spadeAPlayer) {
      for (const pid of seatOrder) {
        observedTeams[pid] = pid === spade3Player ? TeamSide.Solo : TeamSide.Opponent;
      }
    } else {
      for (const pid of seatOrder) {
        if (pid === spade3Player || pid === spadeAPlayer) {
          observedTeams[pid] = TeamSide.SpadeA3;
        } else {
          observedTeams[pid] = TeamSide.Opponent;
        }
      }
    }
  } else if (spade3Player !== null) {
    for (const pid of seatOrder) {
      observedTeams[pid] = pid === spade3Player ? TeamSide.SpadeA3 : TeamSide.Unknown;
    }
  } else if (spadeAPlayer !== null) {
    for (const pid of seatOrder) {
      observedTeams[pid] = pid === spadeAPlayer ? TeamSide.SpadeA3 : TeamSide.Unknown;
    }
  } else {
    for (const pid of seatOrder) {
      observedTeams[pid] = TeamSide.Unknown;
    }
  }

  return { playedCardsByPlayer, observedTeams, playedOutCards };
}

/**
 * 从完整历史构建 recentHistory（截取最近 maxLen 条）
 */
export function buildRecentHistory(
  actionHistory: RawAction[],
  playerNames: Record<string, string>,
  maxLen: number = 40,
): ActionRecord[] {
  const start = actionHistory.length > maxLen ? actionHistory.length - maxLen : 0;
  const result: ActionRecord[] = [];
  for (let i = start; i < actionHistory.length; i++) {
    const a = actionHistory[i]!;
    result.push({
      playerId: a.playerId,
      playerName: playerNames[a.playerId] ?? a.playerId.substring(0, 8),
      action: a.action,
      cards: a.cards,
      handType: a.handType,
    });
  }
  return result;
}
