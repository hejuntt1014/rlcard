import type { Card, AIEngineType } from '@a3/shared';
import { TeamSide, RevealedCard } from '@a3/shared';

export interface Player {
  id: string;
  nickname: string;
  avatar: string;
  seatIndex: number;
  cards: Card[];
  team: TeamSide;
  isTeamRevealed: boolean;
  revealedAs: RevealedCard | null;
  rank: number | null;
  isOnline: boolean;
  isTrustee: boolean;
  isAI: boolean;
  aiEngine: AIEngineType | null;
}

export function createPlayer(
  id: string,
  nickname: string,
  avatar: string,
  seatIndex: number,
  isAI: boolean = false,
  aiEngine: AIEngineType | null = null,
): Player {
  return {
    id,
    nickname,
    avatar,
    seatIndex,
    cards: [],
    team: TeamSide.Unknown,
    isTeamRevealed: false,
    revealedAs: null,
    rank: null,
    isOnline: true,
    isTrustee: false,
    isAI,
    aiEngine,
  };
}

export function removeCardsFromPlayer(player: Player, cardsToRemove: Card[]): void {
  const removeSet = new Set(cardsToRemove.map(c => `${c.suit}_${c.rank}`));
  player.cards = player.cards.filter(c => !removeSet.has(`${c.suit}_${c.rank}`));
}

export function hasCards(player: Player): boolean {
  return player.cards.length > 0;
}
