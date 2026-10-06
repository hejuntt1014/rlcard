import { Rank, STRAIGHT_RANK_VALUE, type GameConfig } from '@a3/shared';
import { type Hand, HandType } from './Hand.js';

export interface StraightRange {
  minRank: Rank;
  maxRank: Rank;
}

type StraightRuleConfig = Pick<GameConfig, 'straightStartRank' | 'straightEndRank'>;

/** 将房间里的顺子起止选项转换成统一的牌点范围。 */
export function straightRangeFromConfig(config: StraightRuleConfig): StraightRange {
  return {
    minRank: config.straightStartRank === '4' ? Rank.Four : Rank.Three,
    maxRank: config.straightEndRank === 'K' ? Rank.King : Rank.Ace,
  };
}

/**
 * 房间起止规则只约束顺子家族；其它五张牌型不受影响。
 * 例如 3-K 禁止 10-J-Q-K-A，4-A 禁止 3-4-5-6-7。
 */
export function isHandWithinStraightRange(hand: Hand, range: StraightRange): boolean {
  if (hand.type !== HandType.Straight && hand.type !== HandType.StraightFlush) {
    return true;
  }

  const minAllowed = STRAIGHT_RANK_VALUE[range.minRank];
  const maxAllowed = STRAIGHT_RANK_VALUE[range.maxRank];
  const values = hand.cards.map((card) => STRAIGHT_RANK_VALUE[card.rank]);
  return Math.min(...values) >= minAllowed && Math.max(...values) <= maxAllowed;
}

export function formatStraightRange(range: StraightRange): string {
  return `${range.minRank}-${range.maxRank}`;
}
