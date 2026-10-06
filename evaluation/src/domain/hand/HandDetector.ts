import type { Card } from '@a3/shared';
import { RANK_VALUE, SUIT_VALUE, STRAIGHT_RANK_VALUE, GAME_CONSTANTS } from '@a3/shared';
import { Rank } from '@a3/shared';
import { type Hand, HandType } from './Hand.js';
import { compareCards, sameRank, sameSuit } from '../card/index.js';

/**
 * 检测给定的牌是否构成合法牌型，返回 Hand 或 null
 */
export function detectHand(cards: Card[]): Hand | null {
  const n = cards.length;
  if (n === 0) return null;

  if (n === 1) return detectSingle(cards);
  if (n === 2) return detectPair(cards);
  if (n === 3) return detectTriple(cards);
  if (n === 5) return detectFiveCard(cards);

  return null;
}

function detectSingle(cards: Card[]): Hand {
  return {
    type: HandType.Single,
    cards,
    primaryCard: cards[0]!,
    size: 1,
  };
}

function detectPair(cards: Card[]): Hand | null {
  if (!sameRank(cards[0]!, cards[1]!)) return null;
  const primary = compareCards(cards[0]!, cards[1]!) >= 0 ? cards[0]! : cards[1]!;
  return {
    type: HandType.Pair,
    cards,
    primaryCard: primary,
    size: 2,
  };
}

function detectTriple(cards: Card[]): Hand | null {
  if (!sameRank(cards[0]!, cards[1]!) || !sameRank(cards[1]!, cards[2]!)) return null;
  const sorted = [...cards].sort((a, b) => compareCards(b, a));
  return {
    type: HandType.Triple,
    cards,
    primaryCard: sorted[0]!,
    size: 3,
  };
}

function detectFiveCard(cards: Card[]): Hand | null {
  const sf = detectStraightFlush(cards);
  if (sf) return sf;

  const fwo = detectFourWithOne(cards);
  if (fwo) return fwo;

  const fh = detectFullHouse(cards);
  if (fh) return fh;

  const fl = detectFlush(cards);
  if (fl) return fl;

  const st = detectStraight(cards);
  if (st) return st;

  return null;
}

function detectStraight(cards: Card[]): Hand | null {
  const sorted = [...cards].sort(
    (a, b) => STRAIGHT_RANK_VALUE[a.rank] - STRAIGHT_RANK_VALUE[b.rank]
  );

  for (let i = 1; i < sorted.length; i++) {
    if (sorted[i]!.rank === sorted[i - 1]!.rank) return null;
    const diff = STRAIGHT_RANK_VALUE[sorted[i]!.rank] - STRAIGHT_RANK_VALUE[sorted[i - 1]!.rank];
    if (diff !== 1) return null;
  }

  if (sorted.some(c => c.rank === Rank.Two)) return null;

  const highCard = sorted.reduce((max, c) =>
    STRAIGHT_RANK_VALUE[c.rank] > STRAIGHT_RANK_VALUE[max.rank] ? c : max
  );
  const bestSuitCard = sorted
    .filter(c => STRAIGHT_RANK_VALUE[c.rank] === STRAIGHT_RANK_VALUE[highCard.rank])
    .sort((a, b) => SUIT_VALUE[b.suit] - SUIT_VALUE[a.suit])[0]!;

  return {
    type: HandType.Straight,
    cards,
    primaryCard: bestSuitCard,
    size: 5,
  };
}

function detectFlush(cards: Card[]): Hand | null {
  const firstSuit = cards[0]!.suit;
  if (!cards.every(c => c.suit === firstSuit)) return null;

  if (detectStraight(cards)) return null;

  const sorted = [...cards].sort((a, b) => compareCards(b, a));
  return {
    type: HandType.Flush,
    cards,
    primaryCard: sorted[0]!,
    size: 5,
  };
}

function detectFullHouse(cards: Card[]): Hand | null {
  const rankGroups = groupByRank(cards);
  const entries = Object.entries(rankGroups);

  let tripleRank: Rank | null = null;
  let pairRank: Rank | null = null;

  for (const [rank, group] of entries) {
    if (group.length === 3) tripleRank = rank as Rank;
    else if (group.length === 2) pairRank = rank as Rank;
  }

  if (!tripleRank || !pairRank) return null;

  const tripleCards = rankGroups[tripleRank]!;
  const primary = tripleCards.sort((a, b) => compareCards(b, a))[0]!;

  return {
    type: HandType.FullHouse,
    cards,
    primaryCard: primary,
    size: 5,
  };
}

function detectFourWithOne(cards: Card[]): Hand | null {
  const rankGroups = groupByRank(cards);
  const entries = Object.entries(rankGroups);

  let quadRank: Rank | null = null;
  let singleExists = false;

  for (const [rank, group] of entries) {
    if (group.length === 4) quadRank = rank as Rank;
    else if (group.length === 1) singleExists = true;
  }

  if (!quadRank || !singleExists) return null;

  const quadCards = rankGroups[quadRank]!;
  const primary = quadCards.sort((a, b) => compareCards(b, a))[0]!;

  return {
    type: HandType.FourWithOne,
    cards,
    primaryCard: primary,
    size: 5,
  };
}

function detectStraightFlush(cards: Card[]): Hand | null {
  const firstSuit = cards[0]!.suit;
  if (!cards.every(c => c.suit === firstSuit)) return null;

  const straight = detectStraight(cards);
  if (!straight) return null;

  return {
    type: HandType.StraightFlush,
    cards,
    primaryCard: straight.primaryCard,
    size: 5,
  };
}

function groupByRank(cards: Card[]): Record<string, Card[]> {
  const groups: Record<string, Card[]> = {};
  for (const card of cards) {
    if (!groups[card.rank]) groups[card.rank] = [];
    groups[card.rank]!.push(card);
  }
  return groups;
}
