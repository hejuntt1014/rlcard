export enum Suit {
  Diamond = 'diamond',
  Club = 'club',
  Heart = 'heart',
  Spade = 'spade',
}

export enum Rank {
  Four = '4',
  Five = '5',
  Six = '6',
  Seven = '7',
  Eight = '8',
  Nine = '9',
  Ten = '10',
  Jack = 'J',
  Queen = 'Q',
  King = 'K',
  Ace = 'A',
  Two = '2',
  Three = '3',
}

export interface Card {
  suit: Suit;
  rank: Rank;
}

export type CardId = string;

export function cardId(card: Card): CardId {
  return `${card.suit}_${card.rank}`;
}

export function cardFromId(id: CardId): Card {
  const [suit, rank] = id.split('_') as [Suit, Rank];
  return { suit, rank };
}

/**
 * 牌面图片编号映射 (基于已有 cards/ 目录):
 * 1-13: 方块 A-K
 * 14-26: 梅花 A-K
 * 27-39: 红桃 A-K
 * 40-52: 黑桃 A-K
 * 55: 牌背
 */
const SUIT_IMAGE_OFFSET: Record<Suit, number> = {
  [Suit.Diamond]: 0,
  [Suit.Club]: 13,
  [Suit.Heart]: 26,
  [Suit.Spade]: 39,
};

const RANK_IMAGE_INDEX: Record<Rank, number> = {
  [Rank.Ace]: 1,
  [Rank.Two]: 2,
  [Rank.Three]: 3,
  [Rank.Four]: 4,
  [Rank.Five]: 5,
  [Rank.Six]: 6,
  [Rank.Seven]: 7,
  [Rank.Eight]: 8,
  [Rank.Nine]: 9,
  [Rank.Ten]: 10,
  [Rank.Jack]: 11,
  [Rank.Queen]: 12,
  [Rank.King]: 13,
};

export function cardImageIndex(card: Card): number {
  return SUIT_IMAGE_OFFSET[card.suit] + RANK_IMAGE_INDEX[card.rank];
}

export const CARD_BACK_IMAGE_INDEX = 55;
