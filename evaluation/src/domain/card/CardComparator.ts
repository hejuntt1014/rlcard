import { type Card, Suit, Rank } from '@a3/shared';
import { RANK_VALUE, SUIT_VALUE } from '@a3/shared';

/**
 * 比较两张牌的大小。
 * 先比点数，点数相同比花色。
 * @returns 正数表示 a > b，负数表示 a < b，0 表示相等
 */
export function compareCards(a: Card, b: Card): number {
  const rankDiff = RANK_VALUE[a.rank] - RANK_VALUE[b.rank];
  if (rankDiff !== 0) return rankDiff;
  return SUIT_VALUE[a.suit] - SUIT_VALUE[b.suit];
}

/**
 * 获取一组牌中最大的那张（用于比较对子、三张等）
 * 对子比较规则: 有黑桃的对子更大
 */
export function maxCard(cards: Card[]): Card {
  return cards.reduce((max, card) => (compareCards(card, max) > 0 ? card : max));
}

/**
 * 检查两张牌是否同点数
 */
export function sameRank(a: Card, b: Card): boolean {
  return a.rank === b.rank;
}

/**
 * 检查两张牌是否同花色
 */
export function sameSuit(a: Card, b: Card): boolean {
  return a.suit === b.suit;
}
