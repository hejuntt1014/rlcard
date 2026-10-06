function cardStr(card: any): string {
  const suitMap: Record<string, string> = { spade: '♠', heart: '♥', club: '♣', diamond: '♦' };
  return `${suitMap[card.suit] ?? card.suit}${card.rank}`;
}

export function cardsStr(cards: any[]): string {
  if (!cards || cards.length === 0) return '[]';
  return cards.map(cardStr).join(' ');
}

export function playerTag(id: string, nickname?: string): string {
  if (id.startsWith('bot_')) {
    const parts = id.split('_');
    return `[仿真-${parts[1] ?? 'seat?'}]`;
  }
  if (id.startsWith('ai_')) {
    const parts = id.split('_');
    return `[AI-${parts[1] ?? 'seat?'}]`;
  }
  return nickname ?? id.substring(0, 12);
}

export function formatDebugHandsBlock(
  title: string,
  players: Array<{
    id: string;
    nickname?: string;
    seatIndex?: number;
    engineId?: string;
    team?: string;
    observedTeam?: string;
    count?: number;
    hand?: any[];
  }>,
): string {
  const lines = players
    .sort((a, b) => (a.seatIndex ?? 99) - (b.seatIndex ?? 99))
    .map((p) => {
      const seat = `座${p.seatIndex ?? '?'}`.padEnd(4, ' ');
      const tag = playerTag(p.id, p.nickname).padEnd(12, ' ');
      const engine = (p.engineId ?? '-').padEnd(14, ' ');
      const team = String(p.team ?? 'unknown').padEnd(10, ' ');
      const observed = String(p.observedTeam ?? 'unknown').padEnd(10, ' ');
      const count = String(p.count ?? (p.hand?.length ?? 0)).padStart(2, ' ');
      return `${seat} ${tag} eng=${engine} team=${team} obs=${observed} rem=${count} | ${cardsStr(p.hand ?? [])}`;
    });
  return `[DebugGame] ${title}\n  ${lines.join('\n  ')}`;
}
