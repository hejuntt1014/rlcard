import { type Card, Suit, Rank, cardId } from '@a3/shared';
import { RANK_VALUE, SUIT_VALUE } from '@a3/shared';

export function createCard(suit: Suit, rank: Rank): Card {
  return { suit, rank };
}

export function getCardValue(card: Card): number {
  return RANK_VALUE[card.rank] * 4 + SUIT_VALUE[card.suit];
}

export function isSameCard(a: Card, b: Card): boolean {
  return a.suit === b.suit && a.rank === b.rank;
}

export function isSpadeThree(card: Card): boolean {
  return card.suit === Suit.Spade && card.rank === Rank.Three;
}

export function isSpadeAce(card: Card): boolean {
  return card.suit === Suit.Spade && card.rank === Rank.Ace;
}

export function isDiamondFour(card: Card): boolean {
  return card.suit === Suit.Diamond && card.rank === Rank.Four;
}

export function containsSpadeThree(cards: Card[]): boolean {
  return cards.some(isSpadeThree);
}

export function containsSpadeAce(cards: Card[]): boolean {
  return cards.some(isSpadeAce);
}

export function containsDiamondFour(cards: Card[]): boolean {
  return cards.some(isDiamondFour);
}

export function sortCards(cards: Card[]): Card[] {
  return [...cards].sort((a, b) => getCardValue(b) - getCardValue(a));
}

export function cardsToIds(cards: Card[]): string[] {
  return cards.map(cardId);
}
