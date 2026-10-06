import type { Card } from '@a3/shared';
import { RANK_VALUE, SUIT_VALUE, STRAIGHT_RANK_VALUE } from '@a3/shared';
import { type Hand, HandType, FIVE_CARD_PRIORITY } from './Hand.js';
import { compareCards } from '../card/index.js';

/**
 * 比较两个牌型的大小。
 * 规则:
 * - 只有相同张数的牌可以比较
 * - 五张牌型之间: 同花顺 > 四带一 > 三带对 > 同花 > 顺子 可以互压
 * - 同牌型比主要牌的大小
 * @returns 正数表示 a > b，负数表示 a < b，0 表示相等
 */
export function compareHands(a: Hand, b: Hand): number {
  if (a.size !== b.size) {
    throw new Error(`Cannot compare hands of different sizes: ${a.size} vs ${b.size}`);
  }

  if (a.size === 5) {
    return compareFiveCardHands(a, b);
  }

  if (a.type !== b.type) {
    throw new Error(`Cannot compare non-five-card hands of different types: ${a.type} vs ${b.type}`);
  }

  return comparePrimaryCards(a, b);
}

function compareFiveCardHands(a: Hand, b: Hand): number {
  const aPriority = FIVE_CARD_PRIORITY[a.type];
  const bPriority = FIVE_CARD_PRIORITY[b.type];

  if (aPriority === undefined || bPriority === undefined) {
    throw new Error(`Unknown five-card hand type: ${a.type} or ${b.type}`);
  }

  if (aPriority !== bPriority) {
    return aPriority - bPriority;
  }

  return comparePrimaryCards(a, b);
}

function comparePrimaryCards(a: Hand, b: Hand): number {
  if (isStraightType(a.type) && isStraightType(b.type)) {
    return compareStraightPrimary(a.primaryCard, b.primaryCard);
  }
  // Flush / StraightFlush: suit of the flush takes priority.
  // ♠ flush beats any ♥/♣/♦ flush regardless of card ranks;
  // only when the suit is the same do we compare the highest card.
  if (a.type === HandType.Flush || a.type === HandType.StraightFlush) {
    return compareFlushPrimary(a, b);
  }
  return compareCards(a.primaryCard, b.primaryCard);
}

function isStraightType(type: HandType): boolean {
  return type === HandType.Straight || type === HandType.StraightFlush;
}

function compareFlushPrimary(a: Hand, b: Hand): number {
  // All cards in a flush share the same suit, so any card's suit = the flush's suit.
  const suitDiff = SUIT_VALUE[a.primaryCard.suit] - SUIT_VALUE[b.primaryCard.suit];
  if (suitDiff !== 0) return suitDiff;
  // Same suit: compare by highest card. StraightFlush uses straight rank order.
  if (a.type === HandType.StraightFlush) {
    return compareStraightPrimary(a.primaryCard, b.primaryCard);
  }
  return RANK_VALUE[a.primaryCard.rank] - RANK_VALUE[b.primaryCard.rank];
}

function compareStraightPrimary(a: Card, b: Card): number {
  const rankDiff = STRAIGHT_RANK_VALUE[a.rank] - STRAIGHT_RANK_VALUE[b.rank];
  if (rankDiff !== 0) return rankDiff;
  return SUIT_VALUE[a.suit] - SUIT_VALUE[b.suit];
}

/**
 * 判断 hand 能否压过 lastHand
 */
export function canBeat(hand: Hand, lastHand: Hand): boolean {
  if (hand.size !== lastHand.size) return false;

  try {
    return compareHands(hand, lastHand) > 0;
  } catch {
    return false;
  }
}
