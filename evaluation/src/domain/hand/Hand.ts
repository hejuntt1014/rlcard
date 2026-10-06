import type { Card } from '@a3/shared';

export enum HandType {
  Single = 'single',
  Pair = 'pair',
  Triple = 'triple',
  Straight = 'straight',
  Flush = 'flush',
  FullHouse = 'full_house',
  FourWithOne = 'four_with_one',
  StraightFlush = 'straight_flush',
}

/**
 * 五张牌型优先级 (同花顺 > 四带一 > 三带对 > 同花 > 顺子)
 */
export const FIVE_CARD_PRIORITY: Record<string, number> = {
  [HandType.Straight]: 0,
  [HandType.Flush]: 1,
  [HandType.FullHouse]: 2,
  [HandType.FourWithOne]: 3,
  [HandType.StraightFlush]: 4,
};

export interface Hand {
  type: HandType;
  cards: Card[];
  /** 用于比较的主要牌 (如三带对取三张中最大的) */
  primaryCard: Card;
  /** 牌的数量 */
  size: number;
}
