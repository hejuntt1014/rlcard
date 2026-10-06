import { Rank, Suit } from '../types/card.js';

/** 牌点大小值映射 (3最大=12, 4最小=0) */
export const RANK_VALUE: Record<Rank, number> = {
  [Rank.Four]: 0,
  [Rank.Five]: 1,
  [Rank.Six]: 2,
  [Rank.Seven]: 3,
  [Rank.Eight]: 4,
  [Rank.Nine]: 5,
  [Rank.Ten]: 6,
  [Rank.Jack]: 7,
  [Rank.Queen]: 8,
  [Rank.King]: 9,
  [Rank.Ace]: 10,
  [Rank.Two]: 11,
  [Rank.Three]: 12,
};

/** 花色大小值映射 (黑桃最大=3, 方块最小=0) */
export const SUIT_VALUE: Record<Suit, number> = {
  [Suit.Diamond]: 0,
  [Suit.Club]: 1,
  [Suit.Heart]: 2,
  [Suit.Spade]: 3,
};

/**
 * 顺子中的牌点值映射 (A最大=12, 3算小=1, 4最小=0)
 * 注意: 在顺子里 3 算小牌，不再是最大
 */
export const STRAIGHT_RANK_VALUE: Record<Rank, number> = {
  [Rank.Three]: 1,
  [Rank.Four]: 2,
  [Rank.Five]: 3,
  [Rank.Six]: 4,
  [Rank.Seven]: 5,
  [Rank.Eight]: 6,
  [Rank.Nine]: 7,
  [Rank.Ten]: 8,
  [Rank.Jack]: 9,
  [Rank.Queen]: 10,
  [Rank.King]: 11,
  [Rank.Ace]: 12,
  [Rank.Two]: 0,
};

export const GAME_CONSTANTS = {
  TOTAL_CARDS: 52,
  PLAYERS_COUNT: 4,
  CARDS_PER_PLAYER: 13,
  STRAIGHT_LENGTH: 5,
  MAX_PASS_TO_NEW_ROUND: 3,
  MAX_AI_RETRY: 3,
} as const;
