import type { Card } from '@a3/shared';
import { Suit, Rank, TeamSide, RevealedCard } from '@a3/shared';
import type { Player } from './Player.js';

/**
 * 根据黑桃3和黑桃A的分布确定队伍
 */
export function assignTeams(players: Player[]): {
  isSolo: boolean;
  soloPlayerId: string | null;
  spadeA3PlayerIds: [string, string] | null;
} {
  let spade3Holder: string | null = null;
  let spadeAHolder: string | null = null;

  for (const player of players) {
    for (const card of player.cards) {
      if (card.suit === Suit.Spade && card.rank === Rank.Three) {
        spade3Holder = player.id;
      }
      if (card.suit === Suit.Spade && card.rank === Rank.Ace) {
        spadeAHolder = player.id;
      }
    }
  }

  if (!spade3Holder || !spadeAHolder) {
    throw new Error('Cannot find Spade 3 or Spade A in any player hand');
  }

  const isSolo = spade3Holder === spadeAHolder;

  if (isSolo) {
    const soloPlayer = players.find(p => p.id === spade3Holder)!;
    soloPlayer.team = TeamSide.Solo;

    for (const player of players) {
      if (player.id !== spade3Holder) {
        player.team = TeamSide.Opponent;
      }
    }

    return { isSolo: true, soloPlayerId: spade3Holder, spadeA3PlayerIds: null };
  }

  for (const player of players) {
    if (player.id === spade3Holder || player.id === spadeAHolder) {
      player.team = TeamSide.SpadeA3;
    } else {
      player.team = TeamSide.Opponent;
    }
  }

  return { isSolo: false, soloPlayerId: null, spadeA3PlayerIds: [spade3Holder, spadeAHolder] };
}

/**
 * 检查是否同队
 */
export function isSameTeam(a: Player, b: Player): boolean {
  if (a.team === TeamSide.Unknown || b.team === TeamSide.Unknown) return false;
  return a.team === b.team;
}

/**
 * 检测出牌是否暴露队伍身份 (打出黑桃3或黑桃A)
 */
export function checkTeamReveal(cards: Card[]): RevealedCard | null {
  for (const card of cards) {
    if (card.suit === Suit.Spade && card.rank === Rank.Three) return RevealedCard.Spade3;
    if (card.suit === Suit.Spade && card.rank === Rank.Ace)   return RevealedCard.SpadeA;
  }
  return null;
}
