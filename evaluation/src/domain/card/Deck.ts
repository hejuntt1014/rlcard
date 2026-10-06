import { type Card, Suit, Rank, GAME_CONSTANTS } from '@a3/shared';
import { createCard } from './Card.js';

function createFullDeck(): Card[] {
  const suits = [Suit.Diamond, Suit.Club, Suit.Heart, Suit.Spade];
  const ranks = [
    Rank.Ace, Rank.Two, Rank.Three, Rank.Four, Rank.Five,
    Rank.Six, Rank.Seven, Rank.Eight, Rank.Nine, Rank.Ten,
    Rank.Jack, Rank.Queen, Rank.King,
  ];
  const deck: Card[] = [];
  for (const suit of suits) {
    for (const rank of ranks) {
      deck.push(createCard(suit, rank));
    }
  }
  return deck;
}

/**
 * Fisher-Yates 洗牌算法
 */
function shuffle<T>(array: T[]): T[] {
  const result = [...array];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [result[i]!, result[j]!] = [result[j]!, result[i]!];
  }
  return result;
}

export interface DealResult {
  hands: [Card[], Card[], Card[], Card[]];
}

/**
 * 创建一副52张牌并洗牌发给4位玩家，每人13张
 */
export function dealCards(): DealResult {
  const deck = shuffle(createFullDeck());

  if (deck.length !== GAME_CONSTANTS.TOTAL_CARDS) {
    throw new Error(`Deck size mismatch: expected ${GAME_CONSTANTS.TOTAL_CARDS}, got ${deck.length}`);
  }

  const perPlayer = GAME_CONSTANTS.CARDS_PER_PLAYER;
  const hands: [Card[], Card[], Card[], Card[]] = [
    deck.slice(0, perPlayer),
    deck.slice(perPlayer, perPlayer * 2),
    deck.slice(perPlayer * 2, perPlayer * 3),
    deck.slice(perPlayer * 3, perPlayer * 4),
  ];

  return { hands };
}
