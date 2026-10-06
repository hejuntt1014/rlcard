export enum RoomStatus {
  Waiting = 'waiting',
  ReadyCheck = 'ready_check',
  Playing = 'playing',
  Settling = 'settling',
  Dissolved = 'dissolved',
}

/** 引擎类型：LLM 固定引擎 + 任意规则引擎（支持服务端动态注册） */
export type AIEngineType = 'llm' | `rule_${string}`;

/** 判断一个字符串是否符合 AIEngineType 的格式 */
export function isValidAIEngineType(value: string): value is AIEngineType {
  return value === 'llm' || value.startsWith('rule_');
}

/** 引擎信息（由后端动态注册，通过 room:get_ai_engines 获取） */
export interface AIEngineInfo {
  id: AIEngineType;
  displayName: string;
}

export type RoomRounds = 10 | 20 | 30;
export type CardCountMode = 'live' | 'two' | 'one' | 'hidden';
export type SeatMode = 'random' | 'fixed';
export type ScoringMode = 'score' | 'diamond';

export interface RoomOptions {
  rounds: RoomRounds;
  aaRoom: boolean;
  cardCountMode: CardCountMode;
  seatMode: SeatMode;
  scoringMode: ScoringMode;
}

/** A player's accumulated result within one multi-round room series. */
export interface PlayerSeriesStats {
  playerId: string;
  nickname: string;
  avatar: string;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  /** Win ratio in the inclusive range 0..1. */
  winRate: number;
  /** Rounds in which this player belonged to the two-player Spade A/3 side. */
  doubleLandlordGames: number;
  /** Rounds in which this player declared. */
  declareGames: number;
  totalScore: number;
}

/** Final aggregate for a 10/20/30-round room series. */
export interface RoomSeriesSettlement {
  roomId: string;
  roomName: string;
  rounds: RoomRounds;
  completedRounds: number;
  /** Determines whether totalScore is displayed as points or diamonds. */
  scoringMode: ScoringMode;
  players: PlayerSeriesStats[];
}

export interface DissolveVoteState {
  roomId: string;
  initiatorId: string;
  agreePlayerIds: string[];
  rejectPlayerIds: string[];
  expiresAt: number;
  requiredAgreeCount: number;
  requiredHumanAgreeCount: number;
}

export interface DissolveVoteResult {
  roomId: string;
  approved: boolean;
  message: string;
}

export interface SeatInfo {
  index: number;
  playerId: string | null;
  nickname: string | null;
  avatar: string | null;
  isReady: boolean;
  isAI: boolean;
  aiEngine: AIEngineType | null;
  isOnline: boolean;
}

export interface RoomInfo {
  roomId: string;
  roomName: string;
  ownerId: string;
  status: RoomStatus;
  seats: SeatInfo[];
  stakes: number;
  configName: string;
  spectatorCount: number;
  createdAt: number;
  /** 叫牌规则：true=必须同时有♠3和♠A才能叫牌 */
  declareRequiresBothSpades: boolean;
  /** 顺子最低起点 */
  straightStartRank: '3' | '4';
  /** 顺子最高终点 */
  straightEndRank: 'K' | 'A';
  /** 好友房总局数及当前局数 */
  rounds: RoomRounds;
  currentRound: number;
  /** 创建 AA 房时收取 5 钻房费 */
  aaRoom: boolean;
  /** 对手剩余牌数的显示方式 */
  cardCountMode: CardCountMode;
  /** 下一局是否随机打乱座位 */
  seatMode: SeatMode;
  /** 仅记录积分，或同步兑付钻石 */
  scoringMode: ScoringMode;
}
