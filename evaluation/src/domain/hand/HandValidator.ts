import type { Card } from '@a3/shared';
import { type Hand } from './Hand.js';
import { detectHand } from './HandDetector.js';
import { canBeat } from './HandComparator.js';
import { containsDiamondFour } from '../card/index.js';
import {
  formatStraightRange,
  isHandWithinStraightRange,
  type StraightRange,
} from './StraightRange.js';

export interface ValidationResult {
  valid: boolean;
  hand: Hand | null;
  error: string | null;
}

/**
 * 验证出牌是否合法
 * @param cards 要出的牌
 * @param playerCards 玩家手中所有牌
 * @param lastPlay 上家出的牌 (null 表示自由出牌)
 * @param isFirstTurn 是否为本局第一手
 * @param straightRange 当前房间允许的顺子起止范围；省略时保持标准 3-A 行为
 */
export function validatePlay(
  cards: Card[],
  playerCards: Card[],
  lastPlay: Hand | null,
  isFirstTurn: boolean,
  straightRange?: StraightRange,
): ValidationResult {
  if (cards.length === 0) {
    return { valid: false, hand: null, error: '必须选择至少一张牌' };
  }

  if (!allCardsInHand(cards, playerCards)) {
    return { valid: false, hand: null, error: '选择的牌不在手中' };
  }

  const hand = detectHand(cards);
  if (!hand) {
    return { valid: false, hand: null, error: '不是合法的牌型' };
  }

  if (straightRange && !isHandWithinStraightRange(hand, straightRange)) {
    return {
      valid: false,
      hand: null,
      error: `当前房间顺子范围为${formatStraightRange(straightRange)}`,
    };
  }

  if (isFirstTurn && !containsDiamondFour(cards)) {
    return { valid: false, hand: null, error: '第一手必须包含方块4' };
  }

  if (lastPlay && !canBeat(hand, lastPlay)) {
    return { valid: false, hand: null, error: '打不过上家的牌' };
  }

  return { valid: true, hand, error: null };
}

function allCardsInHand(cards: Card[], playerCards: Card[]): boolean {
  const handSet = new Set(playerCards.map(c => `${c.suit}_${c.rank}`));
  const usedSet = new Set<string>();

  for (const card of cards) {
    const key = `${card.suit}_${card.rank}`;
    if (!handSet.has(key) || usedSet.has(key)) return false;
    usedSet.add(key);
  }
  return true;
}
