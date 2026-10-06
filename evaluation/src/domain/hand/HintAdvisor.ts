import type { Card } from '@a3/shared';
import { containsDiamondFour, isDiamondFour } from '../card/index.js';
import { compareCards } from '../card/index.js';
import { type Hand, HandType, FIVE_CARD_PRIORITY } from './Hand.js';
import { compareHands } from './HandComparator.js';
import { getPlayableHands } from './HintEngine.js';
import type { StraightRange } from './StraightRange.js';

const FREE_PLAY_TYPE_WEIGHT: Record<HandType, number> = {
  [HandType.Single]: 0,
  [HandType.Pair]: 8,
  [HandType.Triple]: 14,
  [HandType.Straight]: 22,
  [HandType.Flush]: 30,
  [HandType.FullHouse]: 36,
  [HandType.FourWithOne]: 44,
  [HandType.StraightFlush]: 52,
};

interface HandAnalysis {
  splitPenalty: number;
  structurePenalty: number;
  identityPenalty: number;
}

interface HintContext {
  myCards: Card[];
  lastPlay: Hand | null;
  isFirstTurn: boolean;
  structureWeightByCard: Map<string, number>;
}

/**
 * 面向“提示按钮”的建议排序。
 * 目标不是替玩家做博弈，而是按更顺手、更少拆牌的顺序轮询。
 */
export function getSuggestedHands(
  myCards: Card[],
  lastPlay: Hand | null,
  isFirstTurn: boolean,
  straightRange?: StraightRange,
): Hand[] {
  const playable = getPlayableHands(myCards, lastPlay, straightRange)
    .filter((hand) => !isFirstTurn || containsDiamondFour(hand.cards));

  if (playable.length <= 1) {
    return playable;
  }

  const context: HintContext = {
    myCards,
    lastPlay,
    isFirstTurn,
    structureWeightByCard: buildStructureWeightByCard(myCards, straightRange),
  };
  const sorted = [...playable].sort((a, b) => compareSuggestedHands(a, b, context));
  return dedupeEquivalentHints(sorted);
}

function compareSuggestedHands(a: Hand, b: Hand, context: HintContext): number {
  const analysisA = analyzeHand(context.myCards, context.structureWeightByCard, a);
  const analysisB = analyzeHand(context.myCards, context.structureWeightByCard, b);

  const comparisonSteps = [
    analysisA.splitPenalty - analysisB.splitPenalty,
    compareFirstTurnPreference(a, b, context.isFirstTurn),
    compareFollowPreference(a, b, context.lastPlay),
    analysisA.structurePenalty - analysisB.structurePenalty,
    analysisA.identityPenalty - analysisB.identityPenalty,
    compareFreePlayTypePreference(a, b, context.lastPlay),
    safeCompareHands(a, b),
    compareHandCards(a, b),
  ];

  return comparisonSteps.find((value) => value !== 0) ?? 0;
}

function analyzeHand(allCards: Card[], structureWeightByCard: Map<string, number>, hand: Hand): HandAnalysis {
  const originalByRank = groupCardsByRank(allCards);
  const usedByRank = groupCardsByRank(hand.cards);
  const splitPenalty = calculateSplitPenalty(originalByRank, usedByRank);
  const structurePenalty = calculateStructurePenalty(structureWeightByCard, hand);
  const identityPenalty = calculateIdentityPenalty(hand);

  return { splitPenalty, structurePenalty, identityPenalty };
}

function calculateSplitPenalty(
  originalByRank: Map<string, Card[]>,
  usedByRank: Map<string, Card[]>,
): number {
  let penalty = 0;

  for (const [rank, usedCards] of usedByRank.entries()) {
    const originalCount = originalByRank.get(rank)?.length ?? 0;
    const usedCount = usedCards.length;

    if (usedCount === 0 || usedCount === originalCount) {
      continue;
    }

    switch (originalCount) {
      case 2:
        penalty += 30;
        break;
      case 3:
        penalty += usedCount === 1 ? 60 : 45;
        break;
      case 4:
        penalty += usedCount === 3 ? 55 : usedCount === 2 ? 70 : 80;
        break;
      default:
        break;
    }
  }

  return penalty;
}

function calculateStructurePenalty(structureWeightByCard: Map<string, number>, hand: Hand): number {
  if (hand.size === 5) {
    return 0;
  }

  return hand.cards.reduce((total, card) => total + (structureWeightByCard.get(cardKey(card)) ?? 0), 0);
}

function calculateIdentityPenalty(hand: Hand): number {
  let penalty = 0;
  for (const card of hand.cards) {
    if (card.suit === 'spade' && card.rank === '3') {
      penalty += 4;
    }
    if (card.suit === 'spade' && card.rank === 'A') {
      penalty += 3;
    }
  }
  return penalty;
}

function compareFirstTurnPreference(a: Hand, b: Hand, isFirstTurn: boolean): number {
  if (!isFirstTurn) {
    return 0;
  }

  const aUsesOnlyDiamondFour = a.cards.length === 1 && isDiamondFour(a.cards[0]!);
  const bUsesOnlyDiamondFour = b.cards.length === 1 && isDiamondFour(b.cards[0]!);

  if (aUsesOnlyDiamondFour !== bUsesOnlyDiamondFour) {
    return aUsesOnlyDiamondFour ? 1 : -1;
  }

  return 0;
}

function compareFollowPreference(a: Hand, b: Hand, lastPlay: Hand | null): number {
  if (!lastPlay || lastPlay.size !== 5) {
    return 0;
  }

  const aCrossType = a.type !== lastPlay.type;
  const bCrossType = b.type !== lastPlay.type;

  if (aCrossType !== bCrossType) {
    return aCrossType ? 1 : -1;
  }

  return 0;
}

function compareFreePlayTypePreference(a: Hand, b: Hand, lastPlay: Hand | null): number {
  if (lastPlay !== null) {
    return 0;
  }

  return (FREE_PLAY_TYPE_WEIGHT[a.type] ?? 0) - (FREE_PLAY_TYPE_WEIGHT[b.type] ?? 0);
}

function buildStructureWeightByCard(allCards: Card[], straightRange?: StraightRange): Map<string, number> {
  const structureWeightByCard = new Map<string, number>();
  const allPossibleHands = getPlayableHands(allCards, null, straightRange);

  for (const candidate of allPossibleHands) {
    if (candidate.size !== 5) {
      continue;
    }

    const weight = 4 + ((FIVE_CARD_PRIORITY[candidate.type] ?? 0) * 2);
    for (const card of candidate.cards) {
      const key = cardKey(card);
      structureWeightByCard.set(key, Math.max(structureWeightByCard.get(key) ?? 0, weight));
    }
  }

  return structureWeightByCard;
}

function dedupeEquivalentHints(hands: Hand[]): Hand[] {
  const result: Hand[] = [];
  const seen = new Set<string>();

  for (const hand of hands) {
    const key = getHintDedupKey(hand);
    if (seen.has(key)) {
      continue;
    }
    seen.add(key);
    result.push(hand);
  }

  return result;
}

function getHintDedupKey(hand: Hand): string {
  if (hand.size <= 3) {
    return `${hand.type}:${hand.primaryCard.rank}`;
  }

  if (hand.type === HandType.Straight || hand.type === HandType.StraightFlush) {
    const ranks = [...hand.cards].map((card) => card.rank).sort().join(',');
    return `${hand.type}:${ranks}`;
  }

  return `${hand.type}:${hand.cards.map(cardKey).sort().join('|')}`;
}

function safeCompareHands(a: Hand, b: Hand): number {
  try {
    return compareHands(a, b);
  } catch {
    if (a.size !== b.size) {
      return a.size - b.size;
    }
    return 0;
  }
}

function compareHandCards(a: Hand, b: Hand): number {
  const sortedA = [...a.cards].sort(compareCards);
  const sortedB = [...b.cards].sort(compareCards);
  const length = Math.min(sortedA.length, sortedB.length);

  for (let i = 0; i < length; i++) {
    const diff = compareCards(sortedA[i]!, sortedB[i]!);
    if (diff !== 0) {
      return diff;
    }
  }

  return sortedA.length - sortedB.length;
}

function groupCardsByRank(cards: Card[]): Map<string, Card[]> {
  const grouped = new Map<string, Card[]>();
  for (const card of cards) {
    const group = grouped.get(card.rank) ?? [];
    group.push(card);
    grouped.set(card.rank, group);
  }
  return grouped;
}

function cardKey(card: Card): string {
  return `${card.suit}_${card.rank}`;
}
