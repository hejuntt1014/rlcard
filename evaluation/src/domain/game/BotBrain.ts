import type { Card, AIEngineType, TeamSide, RLInferenceDebug } from '@a3/shared';
import type { Hand } from '../hand/index.js';

export interface BotDecision {
  action: 'play' | 'pass' | 'declare' | 'pass_declare';
  cards: Card[];
  debug?: {
    rlInference?: RLInferenceDebug;
    /** engine produced a legal fallback after internal ONNX/inference failure */
    onnxFallbackReason?: string;
  };
}

export interface ActionRecord {
  playerId: string;
  playerName: string;
  action: 'play' | 'pass' | 'declare' | 'pass_declare';
  cards: Card[];
  handType: string;
}

export interface GodModeInfo {
  /** 所有玩家的实际手牌 */
  allPlayersCards: Record<string, Card[]>;
  /** 所有玩家的实际队伍（即使未暴露） */
  allTeams: Record<string, TeamSide>;
  /** 我的实际队伍 */
  myTeam: TeamSide;
  /** 我的实际队友 ID（即使未暴露；solo 模式为 null） */
  actualTeammateId: string | null;
  /** 实际队友的手牌 */
  teammateCards: Card[] | null;
}

export interface BotContext {
  myCards: Card[];
  lastPlay: Hand | null;
  lastPlayPlayerId: string | null;
  isFirstTurn: boolean;
  isNewRound: boolean;
  myPlayerId: string;
  myPlayerName: string;
  teammateId: string | null;
  teammateName: string | null;
  teammateRevealed: boolean;
  playerCardCounts: Record<string, number>;
  playerNames: Record<string, string>;
  rankings: string[];
  /** 当前轮的出牌记录（从上一个自由出牌者开始到现在） */
  currentRoundActions: ActionRecord[];
  /** 最近 N 轮的出牌历史 */
  recentHistory: ActionRecord[];
  /** 已经被打出的所有牌 */
  playedOutCards: Card[];
  /** 各玩家已打出的牌（按玩家分组，从完整历史构建） */
  playedCardsByPlayer: Record<string, Card[]>;
  /**
   * 各玩家的可观测队伍（根据♠3/♠A是否被打出来确定）。
   * 与 Python 训练环境中的 observed teams 完全对应。
   */
  observedTeams: Record<string, TeamSide>;
  /** 出牌顺序（座位索引，逆时针） */
  seatOrder: string[];
  /** God Mode 信息，仅规则 AI 可用 */
  godMode?: GodModeInfo;
  /** 是否处于报牌阶段（若为 true，AI 应返回 declare 或 pass_declare） */
  isDeclaringPhase?: boolean;
  /** 是否已经有人报牌并进入 declarer 模式（公开信息） */
  isDeclared?: boolean;
  /** 已报牌玩家 ID（公开信息；无则为 null） */
  declarerId?: string | null;
  /**
   * 当前轮已连续 pass 的次数（与 Python GameState.pass_count 对应）。
   * 由游戏引擎精确维护，不需要从历史重新计算。
   */
  passCount?: number;
  /** 调试模式：允许 AI 输出高噪声调试信息（如身份推断概率） */
  debugMode?: boolean;
  /** 规则变体配置 */
  ruleConfig?: {
    /** 可选显式规则模式；未提供时由 declareRequiresBothSpades 推导 */
    declareRuleMode?: 'none' | 'free' | 'both_spades';
    declareRequiresBothSpades: boolean;
    straightStartRank: '3' | '4';
    straightEndRank: 'K' | 'A';
  };
}

export interface BotBrain {
  readonly name: string;
  /** 用于前端展示的用户友好名称 */
  readonly displayName: string;
  readonly engineId: AIEngineType;
  decide(ctx: BotContext): Promise<BotDecision>;
}
