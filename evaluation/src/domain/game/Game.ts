import type { Card, GameEvent, GameConfig, GameState, PlayerState, LastPlay, PlayerAction, AIEngineType } from '@a3/shared';
import { GamePhase, TeamSide, Suit, Rank } from '@a3/shared';
import { type Player, createPlayer, removeCardsFromPlayer, hasCards } from './Player.js';
import { assignTeams, checkTeamReveal } from './Team.js';
import { TurnManager } from './TurnManager.js';
import { calculateScores, type ScoreResult } from './Scorer.js';
import { dealCards } from '../card/index.js';
import { containsDiamondFour, sortCards, isDiamondFour } from '../card/index.js';
import {
  detectHand,
  validatePlay,
  hasPlayableHand,
  straightRangeFromConfig,
  type Hand,
} from '../hand/index.js';

/** 报牌阶段等待时间（毫秒） */
export const DECLARE_PHASE_DURATION_MS = 5000;

export class Game {
  readonly gameId: string;
  readonly roomId: string;
  readonly config: GameConfig;
  readonly stakes: number;

  private phase: GamePhase = GamePhase.Waiting;
  private players: Player[] = [];
  private turnManager: TurnManager | null = null;
  private lastPlayHand: Hand | null = null;
  private lastPlay: LastPlay | null = null;
  private rankings: string[] = [];
  private rankCounter: number = 0;
  private isFirstTurn: boolean = true;
  private isSolo: boolean = false;
  private soloPlayerId: string | null = null;
  private scoreResult: ScoreResult | null = null;
  private playerActions: Record<string, PlayerAction> = {};
  private actionHistory: PlayerAction[] = [];
  /** 当前回合的服务器绝对截止时间；快照必须复用它，不能在每次同步时重置。 */
  private turnDeadline: number | null = null;

  // ─── 报牌阶段 ──────────────────────────────────────────────────────────────
  private declaringPhase: boolean = false;
  private declarerId: string | null = null;
  private isDeclared: boolean = false;
  /** 已做过报牌决定（declare 或 pass_declare）的玩家集合 */
  private declaredDecisionSet: Set<string> = new Set();
  private declareDeadline: number | null = null;

  constructor(gameId: string, roomId: string, config: GameConfig, stakes: number) {
    this.gameId = gameId;
    this.roomId = roomId;
    this.config = config;
    this.stakes = stakes;
  }

  getPhase(): GamePhase {
    return this.phase;
  }

  getPlayers(): Player[] {
    return this.players;
  }

  getRankings(): string[] {
    return this.rankings;
  }

  getCurrentPlayerId(): string | null {
    return this.turnManager?.currentPlayerId ?? null;
  }

  getLastPlay(): LastPlay | null {
    return this.lastPlay;
  }

  getLastPlayHand(): Hand | null {
    return this.lastPlayHand;
  }

  getPlayerActions(): Record<string, PlayerAction> {
    return this.playerActions;
  }

  setPlayerOnline(playerId: string, isOnline: boolean): GameEvent[] {
    const player = this.findPlayer(playerId);
    if (player.isOnline === isOnline) return [];
    player.isOnline = isOnline;
    return [{ type: isOnline ? 'PLAYER_RECONNECTED' : 'PLAYER_DISCONNECTED', playerId }];
  }

  setTrustee(playerId: string, enable: boolean): GameEvent[] {
    const player = this.findPlayer(playerId);
    if (player.isTrustee === enable) return [];
    player.isTrustee = enable;
    return [{ type: enable ? 'TRUSTEE_ENABLED' : 'TRUSTEE_DISABLED', playerId }];
  }

  isDeclaringPhase(): boolean {
    return this.declaringPhase;
  }

  /**
   * 初始化游戏: 设置玩家、发牌、确定队伍，进入报牌阶段
   */
  start(
    playerInfos: Array<{ id: string; nickname: string; avatar: string; seatIndex: number; isAI: boolean; aiEngine: AIEngineType | null }>,
    preDealtHands?: [Card[], Card[], Card[], Card[]],
  ): GameEvent[] {
    if (this.phase !== GamePhase.Waiting) {
      throw new Error(`Cannot start game in phase: ${this.phase}`);
    }

    this.players = playerInfos.map(p =>
      createPlayer(p.id, p.nickname, p.avatar, p.seatIndex, p.isAI, p.aiEngine)
    );

    const { hands } = preDealtHands ? { hands: preDealtHands } : dealCards();
    for (let i = 0; i < this.players.length; i++) {
      this.players[i]!.cards = sortCards(hands[i]!);
    }

    const teamInfo = assignTeams(this.players);
    this.isSolo = teamInfo.isSolo;
    this.soloPlayerId = teamInfo.soloPlayerId;

    const startPlayerIndex = this.findDiamondFourHolder();
    this.turnManager = new TurnManager(this.players, startPlayerIndex);

    // 进入报牌阶段（而非直接 Playing）
    this.phase = GamePhase.Declaring;
    this.declaringPhase = true;
    this.isFirstTurn = true;
    this.declareDeadline = Date.now() + DECLARE_PHASE_DURATION_MS;

    const events: GameEvent[] = [
      { type: 'GAME_STARTED', gameId: this.gameId, players: this.getPublicPlayerStates() },
    ];

    for (const player of this.players) {
      events.push({ type: 'CARDS_DEALT', playerId: player.id, cards: player.cards });
    }

    events.push({ type: 'DECLARE_PHASE_STARTED', deadline: this.declareDeadline });

    return events;
  }

  /**
   * 玩家报牌：成为独食者，使用 4× 倍率结算
   */
  declare(playerId: string): GameEvent[] {
    if (!this.declaringPhase) {
      return [{ type: 'PLAY_REJECTED', playerId, reason: '现在不是报牌阶段' }];
    }
    if (this.declarerId !== null) {
      return [{ type: 'PLAY_REJECTED', playerId, reason: '已有玩家报牌' }];
    }
    const declarePlayer = this.players.find(p => p.id === playerId);
    if (!declarePlayer) {
      return [{ type: 'PLAY_REJECTED', playerId, reason: '玩家不存在' }];
    }

    if (this.config.declareRequiresBothSpades) {
      const hasSpade3 = declarePlayer.cards.some(c => c.suit === Suit.Spade && c.rank === Rank.Three);
      const hasSpadeA = declarePlayer.cards.some(c => c.suit === Suit.Spade && c.rank === Rank.Ace);
      if (!hasSpade3 || !hasSpadeA) {
        return [{ type: 'PLAY_REJECTED', playerId, reason: '必须同时持有♠3和♠A才能叫牌' }];
      }
    }

    this.declarerId = playerId;
    this.isDeclared = true;
    this.declaredDecisionSet.add(playerId);

    // 报牌者成为独食玩家（覆盖原有队伍分配），报牌是公开信息，所有人身份确定
    this.isSolo = true;
    this.soloPlayerId = playerId;
    for (const p of this.players) {
      p.team = p.id === playerId ? TeamSide.Solo : TeamSide.Opponent;
      p.isTeamRevealed = true;
    }

    this.playerActions[playerId] = {
      playerId,
      action: 'declare',
      cards: [],
      handType: '',
      timestamp: Date.now(),
    };

    const events: GameEvent[] = [
      { type: 'PLAYER_DECLARED', playerId },
    ];
    events.push(...this.endDeclarationPhase());
    return events;
  }

  /**
   * 玩家选择不报牌；全部 4 人都 pass 则直接进入游戏
   */
  passDeclare(playerId: string): GameEvent[] {
    if (!this.declaringPhase) {
      return [{ type: 'PLAY_REJECTED', playerId, reason: '现在不是报牌阶段' }];
    }
    if (this.declarerId !== null) {
      // 已有人报牌，忽略
      return [];
    }
    if (this.declaredDecisionSet.has(playerId)) {
      return [];
    }

    this.declaredDecisionSet.add(playerId);

    this.playerActions[playerId] = {
      playerId,
      action: 'pass_declare',
      cards: [],
      handType: '',
      timestamp: Date.now(),
    };

    const events: GameEvent[] = [{ type: 'PLAYER_PASS_DECLARED', playerId }];

    // 全部 4 人都 pass → 结束报牌阶段
    if (this.declaredDecisionSet.size >= this.players.length) {
      events.push(...this.endDeclarationPhase());
    }

    return events;
  }

  /**
   * 结束报牌阶段，发出 DECLARE_PHASE_ENDED + TURN_CHANGED
   * （由 declare()、passDeclare() 全员 pass、或服务层计时器超时调用）
   */
  endDeclarationPhase(): GameEvent[] {
    if (!this.declaringPhase) return [];

    this.declaringPhase = false;
    this.phase = GamePhase.Playing;
    this.declareDeadline = null;
    this.turnDeadline = Date.now() + this.config.turnTimeout;

    const events: GameEvent[] = [
      { type: 'DECLARE_PHASE_ENDED', declarerId: this.declarerId },
      {
        type: 'TURN_CHANGED',
        nextPlayerId: this.turnManager!.currentPlayerId,
        turnDeadline: this.turnDeadline,
      },
    ];

    return events;
  }

  /**
   * 玩家出牌
   */
  play(playerId: string, cards: Card[]): GameEvent[] {
    this.assertPlaying();

    if (this.turnManager!.currentPlayerId !== playerId) {
      return [{ type: 'PLAY_REJECTED', playerId, reason: '不是你的回合' }];
    }

    const player = this.findPlayer(playerId);
    const lastHand = this.turnManager!.isNewRound ? null : this.lastPlayHand;

    const validation = validatePlay(
      cards,
      player.cards,
      lastHand,
      this.isFirstTurn,
      straightRangeFromConfig(this.config),
    );
    if (!validation.valid) {
      return [{ type: 'PLAY_REJECTED', playerId, reason: validation.error! }];
    }

    const hand = validation.hand!;
    const events: GameEvent[] = [];

    removeCardsFromPlayer(player, cards);

    this.playerActions[playerId] = {
      playerId,
      action: 'play',
      cards,
      handType: hand.type,
      timestamp: Date.now(),
    };
    this.actionHistory.push(this.playerActions[playerId]!);

    events.push({
      type: 'CARD_PLAYED',
      playerId,
      cards,
      handType: hand.type,
    });

    const reveal = checkTeamReveal(cards);
    if (reveal && !player.isTeamRevealed) {
      player.isTeamRevealed = true;
      player.revealedAs = reveal;
      events.push({ type: 'TEAM_REVEALED', playerId, team: player.team, revealedAs: reveal });
    }

    this.lastPlayHand = hand;
    this.lastPlay = { playerId, cards, handType: hand.type };
    this.isFirstTurn = false;

    if (!hasCards(player)) {
      this.rankCounter++;
      player.rank = this.rankCounter;
      this.rankings.push(playerId);
      events.push({ type: 'PLAYER_FINISHED', playerId, rank: this.rankCounter });

      if (this.isGameOver()) {
        this.finishRemainingPlayers();
        events.push(...this.endGame());
        return events;
      }

      const { nextPlayerId, isNewRound } = this.turnManager!.advanceAfterFinish();
      if (isNewRound) {
        this.lastPlayHand = null;
        this.lastPlay = null;
        this.playerActions = {};
        events.push({ type: 'NEW_ROUND', playerId: nextPlayerId });
      }
      this.turnDeadline = Date.now() + this.config.turnTimeout;
      events.push({
        type: 'TURN_CHANGED',
        nextPlayerId,
        turnDeadline: this.turnDeadline,
      });
    } else {
      const { nextPlayerId, isNewRound } = this.turnManager!.advanceAfterPlay();
      if (isNewRound) {
        this.lastPlayHand = null;
        this.lastPlay = null;
        this.playerActions = {};
        events.push({ type: 'NEW_ROUND', playerId: nextPlayerId });
      }
      this.turnDeadline = Date.now() + this.config.turnTimeout;
      events.push({
        type: 'TURN_CHANGED',
        nextPlayerId,
        turnDeadline: this.turnDeadline,
      });
    }

    return events;
  }

  /**
   * 玩家 pass
   */
  pass(playerId: string): GameEvent[] {
    this.assertPlaying();

    if (this.turnManager!.currentPlayerId !== playerId) {
      return [{ type: 'PLAY_REJECTED', playerId, reason: '不是你的回合' }];
    }

    if (this.turnManager!.isNewRound) {
      return [{ type: 'PLAY_REJECTED', playerId, reason: '自由出牌轮不能Pass' }];
    }

    // 只剩1张牌且能压过上家时，必须出牌，不能Pass
    const player = this.findPlayer(playerId);
    if (player.cards.length === 1 && this.lastPlayHand &&
        hasPlayableHand(player.cards, this.lastPlayHand)) {
      return [{ type: 'PLAY_REJECTED', playerId, reason: '只剩最后一张牌且能赢，必须出牌' }];
    }

    this.playerActions[playerId] = {
      playerId,
      action: 'pass',
      cards: [],
      handType: '',
      timestamp: Date.now(),
    };
    this.actionHistory.push(this.playerActions[playerId]!);

    const events: GameEvent[] = [{ type: 'PLAYER_PASSED', playerId }];

    const { nextPlayerId, isNewRound } = this.turnManager!.advanceAfterPass();

    if (isNewRound) {
      this.lastPlayHand = null;
      this.lastPlay = null;
      this.playerActions = {};
      events.push({ type: 'NEW_ROUND', playerId: nextPlayerId });
    }

    this.turnDeadline = Date.now() + this.config.turnTimeout;
    events.push({
      type: 'TURN_CHANGED',
      nextPlayerId,
      turnDeadline: this.turnDeadline,
    });

    return events;
  }

  /**
   * 获取指定玩家视角的 GameState (信息隔离)
   */
  getStateForPlayer(viewerId: string): GameState {
    return {
      gameId: this.gameId,
      roomId: this.roomId,
      phase: this.phase,
      players: this.players.map(p => this.toPlayerState(p, viewerId)),
      currentTurnPlayerId: this.turnManager?.currentPlayerId ?? null,
      turnDeadline: this.turnDeadline,
      lastPlay: this.lastPlay,
      passCount: this.turnManager?.passCount ?? 0,
      rankings: this.rankings,
      scores: this.scoreResult?.scores ?? null,
      round: 0,
      playerActions: { ...this.playerActions },
      declareDeadline: this.declareDeadline,
      declarerId: this.declarerId,
    };
  }

  /**
   * 获取完整状态 (用于回放记录，不做信息隔离)
   */
  getFullState(): GameState {
    return {
      gameId: this.gameId,
      roomId: this.roomId,
      phase: this.phase,
      players: this.players.map(p => ({
        id: p.id,
        nickname: p.nickname,
        avatar: p.avatar,
        seatIndex: p.seatIndex,
        cardCount: p.cards.length,
        cards: p.cards,
        team: p.team,
        isTeamRevealed: p.isTeamRevealed,
        revealedAs: p.revealedAs,
        rank: p.rank,
        isOnline: p.isOnline,
        isTrustee: p.isTrustee,
        isAI: p.isAI,
        aiEngine: p.aiEngine,
      })),
      currentTurnPlayerId: this.turnManager?.currentPlayerId ?? null,
      turnDeadline: this.turnDeadline,
      lastPlay: this.lastPlay,
      passCount: this.turnManager?.passCount ?? 0,
      rankings: this.rankings,
      scores: this.scoreResult?.scores ?? null,
      round: 0,
      playerActions: { ...this.playerActions },
      declareDeadline: this.declareDeadline,
      declarerId: this.declarerId,
    };
  }

  // --- Private helpers ---

  getActionHistory(): PlayerAction[] {
    return this.actionHistory;
  }

  private assertPlaying(): void {
    if (this.phase !== GamePhase.Playing) {
      throw new Error(`Game is not in playing phase: ${this.phase}`);
    }
  }

  private findPlayer(id: string): Player {
    const player = this.players.find(p => p.id === id);
    if (!player) throw new Error(`Player not found: ${id}`);
    return player;
  }

  private findDiamondFourHolder(): number {
    for (let i = 0; i < this.players.length; i++) {
      if (this.players[i]!.cards.some(isDiamondFour)) {
        return i;
      }
    }
    throw new Error('No player holds Diamond 4');
  }

  private isGameOver(): boolean {
    // 报牌局：任何人出完即结束（只判断谁是第1名，报牌者赢=报牌者第1，否则输）
    if (this.isDeclared && this.declarerId) {
      return this.rankings.length >= 1;
    }

    if (this.isSolo) {
      const soloPlayer = this.players.find(p => p.id === this.soloPlayerId)!;
      if (soloPlayer.rank !== null) return true;
      const opponents = this.players.filter(p => p.id !== this.soloPlayerId);
      if (opponents.every(p => p.rank !== null)) return true;
    }

    const activePlayers = this.players.filter(p => p.rank === null);
    if (activePlayers.length <= 1) return true;

    const teams = new Map<TeamSide, Player[]>();
    for (const p of this.players) {
      if (!teams.has(p.team)) teams.set(p.team, []);
      teams.get(p.team)!.push(p);
    }

    for (const [team, members] of teams) {
      if (team === TeamSide.Unknown) continue;
      if (members.every(m => m.rank !== null)) return true;
    }

    return false;
  }

  private finishRemainingPlayers(): void {
    const remaining = this.players
      .filter(p => p.rank === null)
      .sort((a, b) => a.cards.length - b.cards.length);

    for (const p of remaining) {
      this.rankCounter++;
      p.rank = this.rankCounter;
      this.rankings.push(p.id);
    }
  }

  private endGame(): GameEvent[] {
    this.phase = GamePhase.Finished;
    this.turnDeadline = null;
    this.scoreResult = calculateScores(
      this.players,
      this.rankings,
      this.stakes,
      this.isSolo,
      this.soloPlayerId,
      this.isDeclared,
      this.declarerId,
    );

    return [{
      type: 'GAME_ENDED',
      rankings: this.rankings,
      scores: this.scoreResult.scores,
    }];
  }

  private toPlayerState(player: Player, viewerId: string): PlayerState {
    const isMe = player.id === viewerId;
    const gameFinished = this.phase === GamePhase.Finished;
    return {
      id: player.id,
      nickname: player.nickname,
      avatar: player.avatar,
      seatIndex: player.seatIndex,
      cardCount: player.cards.length,
      cards: (isMe || gameFinished) ? player.cards : undefined,
      team: (player.isTeamRevealed || gameFinished) ? player.team : TeamSide.Unknown,
      isTeamRevealed: player.isTeamRevealed || gameFinished,
      revealedAs: player.revealedAs,
      rank: player.rank,
      isOnline: player.isOnline,
      isTrustee: player.isTrustee,
      isAI: player.isAI,
      aiEngine: player.aiEngine,
    };
  }

  private getPublicPlayerStates(): PlayerState[] {
    return this.players.map(p => ({
      id: p.id,
      nickname: p.nickname,
      avatar: p.avatar,
      seatIndex: p.seatIndex,
      cardCount: p.cards.length,
      team: TeamSide.Unknown,
      isTeamRevealed: false,
      revealedAs: null,
      rank: null,
      isOnline: p.isOnline,
      isTrustee: p.isTrustee,
      isAI: p.isAI,
      aiEngine: p.aiEngine,
    }));
  }
}
