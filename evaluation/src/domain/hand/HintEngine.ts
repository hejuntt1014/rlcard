import type { Card } from '@a3/shared';
import { RANK_VALUE, SUIT_VALUE, STRAIGHT_RANK_VALUE, Rank } from '@a3/shared';
import { type Hand, HandType, FIVE_CARD_PRIORITY } from './Hand.js';
import { detectHand } from './HandDetector.js';
import { canBeat, compareHands } from './HandComparator.js';
import { compareCards } from '../card/index.js';
import { isHandWithinStraightRange, type StraightRange } from './StraightRange.js';

/**
 * 出牌提示引擎
 * 枚举手牌中所有能压过 lastPlay 的合法组合
 * 如果 lastPlay 为 null (自由出牌)，返回所有合法牌型组合
 */
export function getPlayableHands(myCards: Card[], lastPlay: Hand | null, straightRange?: StraightRange): Hand[] {
  let hands: Hand[];
  if (lastPlay === null) {
    hands = getAllPossibleHands(myCards);
  } else {
    hands = findBeatingHands(myCards, lastPlay);
  }

  if (straightRange) {
    hands = hands.filter((hand) => isHandWithinStraightRange(hand, straightRange));
  }

  return hands;
}

export function hasPlayableHand(myCards: Card[], lastPlay: Hand | null): boolean {
  if (lastPlay === null) {
    return myCards.length > 0;
  }

  const size = lastPlay.size;
  if (size === 1) {
    return hasBeatingSingles(myCards, lastPlay);
  }
  if (size === 2) {
    return hasBeatingPairs(myCards, lastPlay);
  }
  if (size === 3) {
    return hasBeatingTriples(myCards, lastPlay);
  }
  if (size === 5) {
    return hasBeatingFiveCards(myCards, lastPlay);
  }

  return false;
}

function findBeatingHands(myCards: Card[], lastPlay: Hand): Hand[] {
  const results: Hand[] = [];
  const size = lastPlay.size;

  if (size === 1) {
    findBeatingSingles(myCards, lastPlay, results);
  } else if (size === 2) {
    findBeatingPairs(myCards, lastPlay, results);
  } else if (size === 3) {
    findBeatingTriples(myCards, lastPlay, results);
  } else if (size === 5) {
    findBeatingFiveCards(myCards, lastPlay, results);
  }

  results.sort((a, b) => compareHands(a, b));
  return results;
}

function findBeatingSingles(cards: Card[], lastPlay: Hand, results: Hand[]): void {
  for (const card of cards) {
    const hand = detectHand([card]);
    if (hand && canBeat(hand, lastPlay)) {
      results.push(hand);
    }
  }
}

function hasBeatingSingles(cards: Card[], lastPlay: Hand): boolean {
  for (const card of cards) {
    if (canBeat({ type: HandType.Single, cards: [card], primaryCard: card, size: 1 }, lastPlay)) {
      return true;
    }
  }
  return false;
}

function findBeatingPairs(cards: Card[], lastPlay: Hand, results: Hand[]): void {
  const groups = groupByRank(cards);
  for (const group of Object.values(groups)) {
    if (group.length < 2) continue;
    const combos = combinations(group, 2);
    for (const combo of combos) {
      const hand = detectHand(combo);
      if (hand && canBeat(hand, lastPlay)) {
        results.push(hand);
      }
    }
  }
}

function hasBeatingPairs(cards: Card[], lastPlay: Hand): boolean {
  const groups = groupByRank(cards);
  for (const group of Object.values(groups)) {
    if (group.length < 2) continue;
    const pair = strongestCards(group, 2);
    if (pair.length < 2) continue;
    const hand = makePairHand(pair);
    if (canBeat(hand, lastPlay)) {
      return true;
    }
  }
  return false;
}

function findBeatingTriples(cards: Card[], lastPlay: Hand, results: Hand[]): void {
  const groups = groupByRank(cards);
  for (const group of Object.values(groups)) {
    if (group.length < 3) continue;
    const combos = combinations(group, 3);
    for (const combo of combos) {
      const hand = detectHand(combo);
      if (hand && canBeat(hand, lastPlay)) {
        results.push(hand);
      }
    }
  }
}

function hasBeatingTriples(cards: Card[], lastPlay: Hand): boolean {
  const groups = groupByRank(cards);
  for (const group of Object.values(groups)) {
    if (group.length < 3) continue;
    const triple = strongestCards(group, 3);
    if (triple.length < 3) continue;
    const hand = makeTripleHand(triple);
    if (canBeat(hand, lastPlay)) {
      return true;
    }
  }
  return false;
}

function findBeatingFiveCards(cards: Card[], lastPlay: Hand, results: Hand[]): void {
  const lastPriority = FIVE_CARD_PRIORITY[lastPlay.type] ?? -1;

  if (lastPriority <= (FIVE_CARD_PRIORITY[HandType.Straight] ?? 0)) {
    findStraights(cards, results, lastPlay);
  }
  if (lastPriority <= (FIVE_CARD_PRIORITY[HandType.Flush] ?? 1)) {
    findFlushes(cards, results, lastPlay);
  }
  if (lastPriority <= (FIVE_CARD_PRIORITY[HandType.FullHouse] ?? 2)) {
    findFullHouses(cards, results, lastPlay);
  }
  if (lastPriority <= (FIVE_CARD_PRIORITY[HandType.FourWithOne] ?? 3)) {
    findFourWithOnes(cards, results, lastPlay);
  }
  if (lastPriority <= (FIVE_CARD_PRIORITY[HandType.StraightFlush] ?? 4)) {
    findStraightFlushes(cards, results, lastPlay);
  }

  dedup(results);
}

function hasBeatingFiveCards(cards: Card[], lastPlay: Hand): boolean {
  const lastPriority = FIVE_CARD_PRIORITY[lastPlay.type] ?? -1;

  if (lastPriority <= (FIVE_CARD_PRIORITY[HandType.Straight] ?? 0) && hasStraights(cards, lastPlay)) {
    return true;
  }
  if (lastPriority <= (FIVE_CARD_PRIORITY[HandType.Flush] ?? 1) && hasFlushes(cards, lastPlay)) {
    return true;
  }
  if (lastPriority <= (FIVE_CARD_PRIORITY[HandType.FullHouse] ?? 2) && hasFullHouses(cards, lastPlay)) {
    return true;
  }
  if (lastPriority <= (FIVE_CARD_PRIORITY[HandType.FourWithOne] ?? 3) && hasFourWithOnes(cards, lastPlay)) {
    return true;
  }
  if (lastPriority <= (FIVE_CARD_PRIORITY[HandType.StraightFlush] ?? 4) && hasStraightFlushes(cards, lastPlay)) {
    return true;
  }

  return false;
}

function findStraights(cards: Card[], results: Hand[], lastPlay: Hand | null): void {
  const groups = groupByRankForStraight(cards);
  const rankKeys = Object.keys(groups).sort(
    (a, b) => STRAIGHT_RANK_VALUE[a as Rank] - STRAIGHT_RANK_VALUE[b as Rank]
  );

  for (let i = 0; i <= rankKeys.length - 5; i++) {
    const window = rankKeys.slice(i, i + 5);
    if (!isConsecutiveStraight(window)) continue;

    const cardOptions = window.map(r => groups[r]!);
    const combos = cartesianProduct(cardOptions);

    for (const combo of combos) {
      const hand = detectHand(combo);
      if (hand && (!lastPlay || canBeat(hand, lastPlay))) {
        results.push(hand);
      }
    }
  }
}

function hasStraights(cards: Card[], lastPlay: Hand): boolean {
  const groups = groupByRankForStraight(cards);
  const rankKeys = Object.keys(groups).sort(
    (a, b) => STRAIGHT_RANK_VALUE[a as Rank] - STRAIGHT_RANK_VALUE[b as Rank]
  );

  for (let i = 0; i <= rankKeys.length - 5; i++) {
    const window = rankKeys.slice(i, i + 5);
    if (!isConsecutiveStraight(window)) continue;

    const cardOptions = window.map(r => groups[r]!);
    const combos = cartesianProduct(cardOptions);
    for (const combo of combos) {
      const hand = detectHand(combo);
      if (hand && canBeat(hand, lastPlay)) {
        return true;
      }
    }
  }

  return false;
}

function findFlushes(cards: Card[], results: Hand[], lastPlay: Hand | null): void {
  const suitGroups = groupBySuit(cards);
  for (const group of Object.values(suitGroups)) {
    if (group.length < 5) continue;
    const combos = combinations(group, 5);
    for (const combo of combos) {
      const hand = detectHand(combo);
      if (hand && hand.type === HandType.Flush && (!lastPlay || canBeat(hand, lastPlay))) {
        results.push(hand);
      }
    }
  }
}

function hasFlushes(cards: Card[], lastPlay: Hand): boolean {
  const suitGroups = groupBySuit(cards);
  for (const group of Object.values(suitGroups)) {
    if (group.length < 5) continue;
    const combos = combinations(group, 5);
    for (const combo of combos) {
      const hand = detectHand(combo);
      if (hand && hand.type === HandType.Flush && canBeat(hand, lastPlay)) {
        return true;
      }
    }
  }
  return false;
}

function findFullHouses(cards: Card[], results: Hand[], lastPlay: Hand | null): void {
  const groups = groupByRank(cards);
  const entries = Object.entries(groups);

  for (const [tripleRank, tripleGroup] of entries) {
    if (tripleGroup.length < 3) continue;

    const tripleCombos = combinations(tripleGroup, 3);

    for (const [pairRank, pairGroup] of entries) {
      if (pairRank === tripleRank || pairGroup.length < 2) continue;

      const pairCombos = combinations(pairGroup, 2);

      for (const triple of tripleCombos) {
        for (const pair of pairCombos) {
          const combo = [...triple, ...pair];
          const hand = detectHand(combo);
          if (hand && (!lastPlay || canBeat(hand, lastPlay))) {
            results.push(hand);
          }
        }
      }
    }
  }
}

function hasFullHouses(cards: Card[], lastPlay: Hand): boolean {
  const groups = groupByRank(cards);
  const entries = Object.entries(groups);

  for (const [tripleRank, tripleGroup] of entries) {
    if (tripleGroup.length < 3) continue;

    const tripleCombos = combinations(tripleGroup, 3);
    for (const [pairRank, pairGroup] of entries) {
      if (pairRank === tripleRank || pairGroup.length < 2) continue;

      const pairCombos = combinations(pairGroup, 2);
      for (const triple of tripleCombos) {
        for (const pair of pairCombos) {
          const hand = detectHand([...triple, ...pair]);
          if (hand && canBeat(hand, lastPlay)) {
            return true;
          }
        }
      }
    }
  }

  return false;
}

function findFourWithOnes(cards: Card[], results: Hand[], lastPlay: Hand | null): void {
  const groups = groupByRank(cards);

  for (const [quadRank, quadGroup] of Object.entries(groups)) {
    if (quadGroup.length < 4) continue;

    for (const card of cards) {
      if (card.rank === quadRank) continue;
      const combo = [...quadGroup.slice(0, 4), card];
      const hand = detectHand(combo);
      if (hand && (!lastPlay || canBeat(hand, lastPlay))) {
        results.push(hand);
      }
    }
  }
}

function hasFourWithOnes(cards: Card[], lastPlay: Hand): boolean {
  const groups = groupByRank(cards);

  for (const [quadRank, quadGroup] of Object.entries(groups)) {
    if (quadGroup.length < 4) continue;

    for (const card of cards) {
      if (card.rank === quadRank) continue;
      const hand = detectHand([...quadGroup.slice(0, 4), card]);
      if (hand && canBeat(hand, lastPlay)) {
        return true;
      }
    }
  }

  return false;
}

function findStraightFlushes(cards: Card[], results: Hand[], lastPlay: Hand | null): void {
  const suitGroups = groupBySuit(cards);
  for (const group of Object.values(suitGroups)) {
    if (group.length < 5) continue;
    findStraights(group, results, lastPlay);
  }
}

function hasStraightFlushes(cards: Card[], lastPlay: Hand): boolean {
  const suitGroups = groupBySuit(cards);
  for (const group of Object.values(suitGroups)) {
    if (group.length < 5) continue;
    const groupsByRank = groupByRankForStraight(group);
    const rankKeys = Object.keys(groupsByRank).sort(
      (a, b) => STRAIGHT_RANK_VALUE[a as Rank] - STRAIGHT_RANK_VALUE[b as Rank]
    );

    for (let i = 0; i <= rankKeys.length - 5; i++) {
      const window = rankKeys.slice(i, i + 5);
      if (!isConsecutiveStraight(window)) continue;

      const combo = window.map((rank) => groupsByRank[rank]![0]!);
      const hand = detectHand(combo);
      if (hand && hand.type === HandType.StraightFlush && canBeat(hand, lastPlay)) {
        return true;
      }
    }
  }
  return false;
}

function getAllPossibleHands(cards: Card[]): Hand[] {
  const results: Hand[] = [];

  for (const card of cards) {
    results.push({ type: HandType.Single, cards: [card], primaryCard: card, size: 1 });
  }

  const groups = groupByRank(cards);
  for (const group of Object.values(groups)) {
    if (group.length >= 2) {
      const combos = combinations(group, 2);
      for (const combo of combos) {
        const hand = detectHand(combo);
        if (hand) results.push(hand);
      }
    }
    if (group.length >= 3) {
      const combos = combinations(group, 3);
      for (const combo of combos) {
        const hand = detectHand(combo);
        if (hand) results.push(hand);
      }
    }
  }

  findStraights(cards, results, null);
  findFlushes(cards, results, null);
  findFullHouses(cards, results, null);
  findFourWithOnes(cards, results, null);

  dedup(results);
  results.sort((a, b) => {
    if (a.size !== b.size) return a.size - b.size;
    try { return compareHands(a, b); } catch { return 0; }
  });

  return results;
}

// --- Utility functions ---

function groupByRank(cards: Card[]): Record<string, Card[]> {
  const groups: Record<string, Card[]> = {};
  for (const card of cards) {
    if (!groups[card.rank]) groups[card.rank] = [];
    groups[card.rank]!.push(card);
  }
  return groups;
}

function groupByRankForStraight(cards: Card[]): Record<string, Card[]> {
  return groupByRank(cards.filter(c => c.rank !== '2'));
}

function groupBySuit(cards: Card[]): Record<string, Card[]> {
  const groups: Record<string, Card[]> = {};
  for (const card of cards) {
    if (!groups[card.suit]) groups[card.suit] = [];
    groups[card.suit]!.push(card);
  }
  return groups;
}

function isConsecutiveStraight(ranks: string[]): boolean {
  for (let i = 1; i < ranks.length; i++) {
    const diff = STRAIGHT_RANK_VALUE[ranks[i]! as Rank] - STRAIGHT_RANK_VALUE[ranks[i - 1]! as Rank];
    if (diff !== 1) return false;
  }
  return true;
}

function combinations<T>(arr: T[], k: number): T[][] {
  if (k === 0) return [[]];
  if (arr.length < k) return [];

  const results: T[][] = [];
  for (let i = 0; i <= arr.length - k; i++) {
    const rest = combinations(arr.slice(i + 1), k - 1);
    for (const combo of rest) {
      results.push([arr[i]!, ...combo]);
    }
  }
  return results;
}

function cartesianProduct<T>(arrays: T[][]): T[][] {
  if (arrays.length === 0) return [[]];
  const [first, ...rest] = arrays;
  const restProduct = cartesianProduct(rest);
  const results: T[][] = [];
  for (const item of first!) {
    for (const combo of restProduct) {
      results.push([item, ...combo]);
    }
  }
  return results;
}

function dedup(hands: Hand[]): void {
  const seen = new Set<string>();
  let writeIdx = 0;
  for (let i = 0; i < hands.length; i++) {
    const key = hands[i]!.cards
      .map(c => `${c.suit}_${c.rank}`)
      .sort()
      .join('|');
    if (!seen.has(key)) {
      seen.add(key);
      hands[writeIdx] = hands[i]!;
      writeIdx++;
    }
  }
  hands.length = writeIdx;
}

function strongestCards(cards: Card[], count: number): Card[] {
  return [...cards].sort((a, b) => compareCards(b, a)).slice(0, count);
}

function makePairHand(cards: Card[]): Hand {
  const primaryCard = compareCards(cards[0]!, cards[1]!) >= 0 ? cards[0]! : cards[1]!;
  return { type: HandType.Pair, cards, primaryCard, size: 2 };
}

function makeTripleHand(cards: Card[]): Hand {
  const sorted = [...cards].sort((a, b) => compareCards(b, a));
  return { type: HandType.Triple, cards, primaryCard: sorted[0]!, size: 3 };
}
