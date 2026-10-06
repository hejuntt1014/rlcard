import type { Card } from '@a3/shared';
import type { BotBrain, BotContext, BotDecision } from '../BotBrain.js';
import { getPlayableHands, HandType } from '../../hand/index.js';
import { sortCards, getCardValue } from '../../card/index.js';

export abstract class BaseRuleBrain implements BotBrain {
  abstract readonly name: string;
  abstract readonly displayName: string;
  abstract readonly engineId: `rule_${string}`;

  abstract decide(ctx: BotContext): Promise<BotDecision>;

  protected playFirstTurn(cards: Card[]): BotDecision {
    const d4 = cards.find((c) => c.suit === 'diamond' && c.rank === '4');
    if (!d4) {
      return { action: 'play', cards: [sortCards(cards).pop()!] };
    }

    const pairs = this.findPairsContaining(cards, d4);
    if (pairs) return { action: 'play', cards: pairs };

    return { action: 'play', cards: [d4] };
  }

  protected playAggressively(sorted: Card[]): BotDecision {
    if (sorted.length === 1) {
      return { action: 'play', cards: [sorted[0]!] };
    }

    if (sorted.length === 2 && sorted[0]!.rank === sorted[1]!.rank) {
      return { action: 'play', cards: sorted };
    }

    if (
      sorted.length === 3 &&
      sorted[0]!.rank === sorted[1]!.rank &&
      sorted[1]!.rank === sorted[2]!.rank
    ) {
      return { action: 'play', cards: sorted };
    }

    return { action: 'play', cards: [sorted[sorted.length - 1]!] };
  }

  protected playWeakest(sorted: Card[]): BotDecision {
    const groups = this.groupByRank(sorted);
    const entries = Object.entries(groups).sort(
      ([, a], [, b]) => getCardValue(a[0]!) - getCardValue(b[0]!)
    );

    for (const [, group] of entries) {
      if (group.length === 1) {
        return { action: 'play', cards: [group[0]!] };
      }
    }

    for (const [, group] of entries) {
      if (group.length === 2) {
        return { action: 'play', cards: group };
      }
    }

    return { action: 'play', cards: [sorted[sorted.length - 1]!] };
  }

  protected playStrongest(sorted: Card[]): BotDecision {
    const groups = this.groupByRank(sorted);
    const entries = Object.entries(groups).sort(
      ([, a], [, b]) => getCardValue(b[0]!) - getCardValue(a[0]!)
    );

    for (const [, group] of entries) {
      if (group.length >= 3) {
        return { action: 'play', cards: group.slice(0, 3) };
      }
    }

    for (const [, group] of entries) {
      if (group.length === 2) {
        return { action: 'play', cards: group };
      }
    }

    return { action: 'play', cards: [sorted[0]!] };
  }

  protected getConservativeFollowUp(ctx: BotContext): BotDecision {
    const { myCards, lastPlay } = ctx;
    const playable = getPlayableHands(myCards, lastPlay);
    if (playable.length === 0) {
      return { action: 'pass', cards: [] };
    }

    // 保留四带一/同花顺（最强牌型），优先用普通牌型跟牌
    const nonPremium = playable.filter(
      (h) => h.type !== HandType.FourWithOne && h.type !== HandType.StraightFlush
    );
    const candidates = nonPremium.length > 0 ? nonPremium : playable;

    let best = candidates[0]!;
    let bestPenalty = this.calculateBreakPenalty(myCards, best.cards);
    for (const hand of candidates.slice(1)) {
      const penalty = this.calculateBreakPenalty(myCards, hand.cards);
      if (penalty < bestPenalty) {
        best = hand;
        bestPenalty = penalty;
      }
    }

    return { action: 'play', cards: best.cards };
  }

  protected shouldPressureOpponent(ctx: BotContext): boolean {
    const { playerCardCounts, teammateId, myPlayerId } = ctx;

    for (const [id, count] of Object.entries(playerCardCounts)) {
      if (id === myPlayerId || id === teammateId) continue;
      if (count <= 3) return true;
    }

    return false;
  }

  protected isTeammateLastPlay(ctx: BotContext): boolean {
    if (!ctx.teammateId || !ctx.lastPlayPlayerId || !ctx.lastPlay) return false;
    return ctx.lastPlayPlayerId === ctx.teammateId;
  }

  protected findPairsContaining(cards: Card[], target: Card): Card[] | null {
    const partner = cards.find((c) => c.rank === target.rank && c.suit !== target.suit);
    if (partner) return [target, partner];
    return null;
  }

  protected groupByRank(cards: Card[]): Record<string, Card[]> {
    const groups: Record<string, Card[]> = {};
    for (const card of cards) {
      if (!groups[card.rank]) groups[card.rank] = [];
      groups[card.rank]!.push(card);
    }
    return groups;
  }

  private calculateBreakPenalty(allCards: Card[], selectedCards: Card[]): number {
    const selectedKeys = new Set(selectedCards.map((card) => `${card.suit}_${card.rank}`));
    const groups = this.groupByRank(allCards);
    let penalty = 0;

    for (const [rank, group] of Object.entries(groups)) {
      const usedCount = group.filter((card) => selectedKeys.has(`${card.suit}_${card.rank}`)).length;
      if (usedCount === 0 || usedCount === group.length) continue;

      if (group.length === 2) penalty += 30;
      else if (group.length === 3) penalty += 60;
      else if (group.length === 4) penalty += 90;

      if (selectedCards.length === 1 && rank === selectedCards[0]!.rank) {
        penalty += 10;
      }
    }

    return penalty;
  }
}
