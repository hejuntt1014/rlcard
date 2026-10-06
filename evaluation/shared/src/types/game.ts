import type { Card } from './card.js';
import type { AIEngineType } from './room.js';

export enum GamePhase {
  Waiting = 'waiting',
  ReadyCheck = 'ready_check',
  Dealing = 'dealing',
  Declaring = 'declaring',
  Playing = 'playing',
  Settling = 'settling',
  Finished = 'finished',
}

export enum SeatStatus {
  Empty = 'empty',
  Occupied = 'occupied',
  Ready = 'ready',
  Disconnected = 'disconnected',
}

export enum TeamSide {
  SpadeA3 = 'spade_a3',
  Opponent = 'opponent',
  Solo = 'solo',
  Unknown = 'unknown',
}

/** 打出哪张身份牌触发了队伍暴露 */
export enum RevealedCard {
  Spade3 = 'spade3',
  SpadeA = 'spadeA',
}

export interface PlayerState {
  id: string;
  nickname: string;
  avatar: string;
  seatIndex: number;
  cardCount: number;
  cards?: Card[];
  team: TeamSide;
  isTeamRevealed: boolean;
  /** 打出♠3 → Spade3，打出♠A → SpadeA，对手队/未暴露 → null */
  revealedAs: RevealedCard | null;
  rank: number | null;
  isOnline: boolean;
  isTrustee: boolean;
  isAI: boolean;
  aiEngine?: AIEngineType | null;
}

export interface LastPlay {
  playerId: string;
  cards: Card[];
  handType: string;
}

export interface PlayerAction {
  playerId: string;
  action: 'play' | 'pass' | 'none' | 'declare' | 'pass_declare';
  cards: Card[];
  handType: string;
  timestamp: number;
}

export type RLGuessLabel = 'A3' | 'OPP' | 'SOLO';

export interface RLInferenceGuess {
  label: '下家' | '对家' | '上家';
  a3: number;
  opponent: number;
  solo: number;
  mostLikely: RLGuessLabel;
}

export interface RLInferenceDebug {
  actionLabel: string;
  qValue: number | null;
  summary: string;
  guesses: RLInferenceGuess[];
}

export interface GameDebugState {
  enabled: boolean;
  observerPlayerId: string;
  rlByPlayer: Record<string, RLInferenceDebug>;
}

export interface GameState {
  gameId: string;
  roomId: string;
  phase: GamePhase;
  players: PlayerState[];
  currentTurnPlayerId: string | null;
  turnDeadline: number | null;
  lastPlay: LastPlay | null;
  passCount: number;
  rankings: string[];
  scores: Record<string, number> | null;
  round: number;
  playerActions: Record<string, PlayerAction>;
  /** 报牌阶段截止时间（毫秒时间戳），非报牌阶段为 null */
  declareDeadline: number | null;
  /** 报牌玩家 ID，无人报牌为 null */
  declarerId: string | null;
  /** 仅调试模式下存在：全员手牌 + RL 推断等调试信息 */
  debug?: GameDebugState;
}

export type GameEvent =
  | { type: 'GAME_STARTED'; gameId: string; players: PlayerState[] }
  | { type: 'CARDS_DEALT'; playerId: string; cards: Card[] }
  | { type: 'DECLARE_PHASE_STARTED'; deadline: number }
  | { type: 'PLAYER_DECLARED'; playerId: string }
  | { type: 'PLAYER_PASS_DECLARED'; playerId: string }
  | { type: 'DECLARE_PHASE_ENDED'; declarerId: string | null }
  | { type: 'CARD_PLAYED'; playerId: string; cards: Card[]; handType: string }
  | { type: 'PLAYER_PASSED'; playerId: string }
  | { type: 'TURN_CHANGED'; nextPlayerId: string; turnDeadline: number }
  | { type: 'NEW_ROUND'; playerId: string }
  | { type: 'PLAYER_FINISHED'; playerId: string; rank: number }
  | { type: 'TEAM_REVEALED'; playerId: string; team: TeamSide; revealedAs: RevealedCard }
  | { type: 'GAME_ENDED'; rankings: string[]; scores: Record<string, number> }
  | { type: 'PLAYER_DISCONNECTED'; playerId: string }
  | { type: 'PLAYER_RECONNECTED'; playerId: string }
  | { type: 'TRUSTEE_ENABLED'; playerId: string }
  | { type: 'TRUSTEE_DISABLED'; playerId: string }
  | { type: 'PLAY_REJECTED'; playerId: string; reason: string };
