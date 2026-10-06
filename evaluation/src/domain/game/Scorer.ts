import { TeamSide } from '@a3/shared';
import type { Player } from './Player.js';
import { isSameTeam } from './Team.js';

export interface ScoreResult {
  scores: Record<string, number>;
  details: string;
}

/**
 * 根据名次和队伍关系计算结算分数
 *
 * 普通模式 (2v2):
 *   同伴一二名: 各+2倍底注
 *   同伴一三名: 各+1倍底注
 *   同伴一四名: 平手
 *
 * 独食模式 (1v3):
 *   独食者第一名: +6倍, 其他各-2倍
 *   独食者第二名: +3倍, 其他各-1倍
 *   独食者第三名: -3倍, 其他各+1倍
 *   独食者第四名: -6倍, 其他各+2倍
 *
 * 报牌模式 (declared 1v3):
 *   报牌者第一名: +4*3倍=+12倍, 其他各-4倍
 *   报牌者非第一名: -4*3倍=-12倍, 其他各+4倍
 */
export function calculateScores(
  players: Player[],
  rankings: string[],
  stakes: number,
  isSolo: boolean,
  soloPlayerId: string | null,
  isDeclared: boolean = false,
  declarerId: string | null = null,
): ScoreResult {
  const scores: Record<string, number> = {};
  for (const p of players) scores[p.id] = 0;

  if (isDeclared && declarerId) {
    return calculateDeclaredScores(players, rankings, stakes, declarerId, scores);
  }

  if (isSolo && soloPlayerId) {
    return calculateSoloScores(players, rankings, stakes, soloPlayerId, scores);
  }

  return calculateTeamScores(players, rankings, stakes, scores);
}

/**
 * 报牌结算：赢得第一名则 +4×底注×3，否则 -4×底注×3
 */
function calculateDeclaredScores(
  players: Player[],
  rankings: string[],
  stakes: number,
  declarerId: string,
  scores: Record<string, number>,
): ScoreResult {
  const declaredRank = rankings.indexOf(declarerId) + 1;
  const declaredWon = declaredRank === 1;

  const declaredScore = declaredWon ? 4 * stakes * 3 : -4 * stakes * 3;
  const otherScore = declaredWon ? -4 * stakes : 4 * stakes;

  scores[declarerId] = declaredScore;
  for (const p of players) {
    if (p.id !== declarerId) {
      scores[p.id] = otherScore;
    }
  }

  return {
    scores,
    details: declaredWon
      ? `报牌者第一名 (+${4 * 3}倍)`
      : `报牌者未获第一名 (-${4 * 3}倍)`,
  };
}

function calculateSoloScores(
  players: Player[],
  rankings: string[],
  stakes: number,
  soloPlayerId: string,
  scores: Record<string, number>,
): ScoreResult {
  const soloRank = rankings.indexOf(soloPlayerId) + 1;

  let soloMultiplier: number;
  let otherMultiplier: number;
  let details: string;

  switch (soloRank) {
    case 1:
      soloMultiplier = 6;
      otherMultiplier = -2;
      details = '独食者第一名';
      break;
    case 2:
      soloMultiplier = 3;
      otherMultiplier = -1;
      details = '独食者第二名';
      break;
    case 3:
      soloMultiplier = -3;
      otherMultiplier = 1;
      details = '独食者第三名';
      break;
    case 4:
      soloMultiplier = -6;
      otherMultiplier = 2;
      details = '独食者第四名';
      break;
    default:
      soloMultiplier = 0;
      otherMultiplier = 0;
      details = '未知名次';
  }

  scores[soloPlayerId] = soloMultiplier * stakes;
  for (const p of players) {
    if (p.id !== soloPlayerId) {
      scores[p.id] = otherMultiplier * stakes;
    }
  }

  return { scores, details };
}

function calculateTeamScores(
  players: Player[],
  rankings: string[],
  stakes: number,
  scores: Record<string, number>,
): ScoreResult {
  const firstPlayer = players.find(p => p.id === rankings[0])!;
  const secondPlayer = players.find(p => p.id === rankings[1])!;

  const firstAndSecondSameTeam = isSameTeam(firstPlayer, secondPlayer);

  if (firstAndSecondSameTeam) {
    for (const p of players) {
      scores[p.id] = isSameTeam(p, firstPlayer) ? 2 * stakes : -2 * stakes;
    }
    return { scores, details: '同伴一二名 (+2倍)' };
  }

  const thirdPlayer = players.find(p => p.id === rankings[2])!;
  const firstAndThirdSameTeam = isSameTeam(firstPlayer, thirdPlayer);

  if (firstAndThirdSameTeam) {
    for (const p of players) {
      scores[p.id] = isSameTeam(p, firstPlayer) ? 1 * stakes : -1 * stakes;
    }
    return { scores, details: '同伴一三名 (+1倍)' };
  }

  return { scores, details: '同伴一四名 (平手)' };
}
