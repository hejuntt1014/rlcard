import type { Card, AIEngineType, PlayerAction } from '@a3/shared';
import { GamePhase, TeamSide, ZHANJIANG_STANDARD } from '@a3/shared';
import { Game } from '../domain/game/Game.js';
import { config } from '../config/index.js';
import type { Player } from '../domain/game/Player.js';
import type { BotBrain, BotContext, BotDecision, ActionRecord, GodModeInfo } from '../domain/game/BotBrain.js';
import { detectHand, validatePlay } from '../domain/hand/index.js';
import { sortCards } from '../domain/card/index.js';
import { buildPlayedCardsAndTeams, buildRecentHistory } from '../domain/game/BotContextHelpers.js';
import { cardsStr, formatDebugHandsBlock, playerTag } from '../utils/logger.js';

export interface TurnRecord {
  turn: number;
  playerId: string;
  engineId: AIEngineType;
  action: 'play' | 'pass';
  cards: Card[];
  handType: string;
  cardsRemaining: number;
  decisionSource: 'engine' | 'fallback' | 'force' | 'timeout' | 'error';
  isNewRound: boolean;
  isFirstTurn: boolean;
  elapsedMs: number;
}

export interface DecisionSourceCounts {
  total: number;
  onnxFallback: number;
  fallback: number;
  timeout: number;
  error: number;
  force: number;
}

/** Full context of one invalid play attempt, captured at the moment of fallback */
export interface InvalidPlayRecord {
  turn: number;
  engineId: AIEngineType;
  /** validation error message from validatePlay() */
  validationError: string;
  /** cards the engine tried to play */
  attempted: Card[];
  /** all cards in the player's hand at that moment */
  playerHand: Card[];
  /** the hand the engine needed to beat (null = free play) */
  lastPlayCards: Card[] | null;
  lastPlayHandType: string | null;
  isFirstTurn: boolean;
  isNewRound: boolean;
  /** how many cards remain in other players' hands */
  otherCardCounts: Record<string, number>;
}

export interface GameResult {
  rankings: string[];
  scores: Record<string, number>;
  playerEngines: Record<string, AIEngineType>;
  playerTeams: Record<string, TeamSide>;
  declarerPlayerId: string | null;
  declarerEngineId: AIEngineType | null;
  isSolo: boolean;
  turns: number;
  /** per-engine decision source breakdown */
  decisionStats: Partial<Record<AIEngineType, DecisionSourceCounts>>;
  /** full context of every invalid play attempt (only when captureInvalidPlays=true) */
  invalidPlays?: InvalidPlayRecord[];
  error?: string;
  stuckInfo?: StuckGameInfo;
}

export interface StuckGameInfo {
  reason: string;
  lastTurns: TurnRecord[];
  playerStates: Array<{
    id: string;
    engineId: AIEngineType;
    cards: Card[];
    team: string;
    rank: number | null;
  }>;
  consecutivePasses: number;
  totalTurns: number;
}

export interface EngineAssignment {
  seatIndex: number;
  engineId: AIEngineType;
}

const MAX_TURNS = 300;
const DECIDE_TIMEOUT_MS = 5000;
const MAX_CONSECUTIVE_PASSES = 40;

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timeout after ${ms}ms`)), ms);
    promise.then(
      (v) => { clearTimeout(timer); resolve(v); },
      (e) => { clearTimeout(timer); reject(e); },
    );
  });
}

export class BenchmarkRunner {
  private engines: Map<AIEngineType, BotBrain>;
  private verbose: boolean;
  private debugMode: boolean;
  private rlInferenceDebug: boolean;
  private captureInvalidPlays: boolean;
  private onStuckGame?: (info: StuckGameInfo) => void;

  constructor(
    engines: Map<AIEngineType, BotBrain>,
    verbose = false,
    onStuckGame?: (info: StuckGameInfo) => void,
    captureInvalidPlays = false,
  ) {
    this.engines = engines;
    this.verbose = verbose;
    // Benchmark 默认关闭全量手牌调试，避免评测输出被高噪声日志淹没。
    this.debugMode = verbose && config.gameDebugMode;
    this.rlInferenceDebug = config.rlInferenceDebug;
    this.captureInvalidPlays = captureInvalidPlays;
    this.onStuckGame = onStuckGame;
  }

  async runGame(
    assignments: EngineAssignment[],
    preDealtHands?: [Card[], Card[], Card[], Card[]],
  ): Promise<GameResult> {
    if (assignments.length !== 4) throw new Error('Need exactly 4 players');

    const gameConfig = { ...ZHANJIANG_STANDARD, turnTimeout: 999999 };
    const game = new Game(`bench_${Date.now()}`, 'bench_room', gameConfig, 100);

    const playerInfos = assignments.map((a) => ({
      id: `bot_s${a.seatIndex}_${a.engineId}`,
      nickname: `${a.engineId}[${a.seatIndex}]`,
      avatar: '',
      seatIndex: a.seatIndex,
      isAI: true,
      aiEngine: a.engineId,
    }));

    game.start(playerInfos, preDealtHands);

    // ─── 报牌阶段 ──────────────────────────────────────────────────────────
    const players = game.getPlayers();
    if (game.isDeclaringPhase()) {
      // 按座位顺序让每个玩家做报牌决策
      const sortedForDecl = [...players].sort((a, b) => a.seatIndex - b.seatIndex);
      for (const p of sortedForDecl) {
        if (!game.isDeclaringPhase()) break; // 已有人报牌，停止

        const engineId = p.aiEngine ?? ('rule_default' as AIEngineType);
        const engine = this.engines.get(engineId);

        let decideDeclare: boolean;

        // 支持 doubleNT 报牌决策的引擎（rule_rl 用 ONNX，rule_refai/rule_ultra 用 doubleNT 模拟）
        const ENGINE_USES_DECIDE_FOR_DECLARE = new Set(['rule_rl', 'rule_refai', 'rule_ultra']);

        if (engine && ENGINE_USES_DECIDE_FOR_DECLARE.has(engineId)) {
          try {
            const declCtx = this.buildDeclareContext(p, players);
            if (this.debugMode) {
              this.logDebugHands(`[Declare][${engineId}] ${playerTag(p.id, p.nickname)}`, players, declCtx);
            }
            const decision = await engine.decide(declCtx);
            decideDeclare = decision.action === 'declare';
          } catch (error) {
            if (process.env.A3_EVAL_STRICT === '1') throw error;
            decideDeclare = false;
          }
        } else {
          // 其他引擎使用规则启发（手牌强度评分）
          decideDeclare = shouldDeclare(p.cards);
        }

        if (decideDeclare) {
          game.declare(p.id);
        } else {
          game.passDeclare(p.id);
        }
      }
      // 若仍在报牌阶段（全员 pass 但游戏未自动结束），强制结束
      if (game.isDeclaringPhase()) {
        game.endDeclarationPhase();
      }
    }

    const playerEngines: Record<string, AIEngineType> = {};
    const playerTeams: Record<string, TeamSide> = {};
    for (const p of players) {
      playerEngines[p.id] = p.aiEngine!;
      playerTeams[p.id] = p.team;
    }
    const declarationState = game.getFullState();
    const declarerPlayerId = declarationState.declarerId ?? null;
    const declarerEngineId = declarerPlayerId ? (playerEngines[declarerPlayerId] ?? null) : null;

    // Pre-compute stable data that doesn't change per turn
    const playerNames: Record<string, string> = {};
    const seatOrder: string[] = [];
    const sortedPlayers = [...players].sort((a, b) => a.seatIndex - b.seatIndex);
    for (const p of sortedPlayers) {
      playerNames[p.id] = p.nickname;
      seatOrder.push(p.id);
    }

    let turns = 0;
    let consecutivePasses = 0;
    const actionHistory: Array<{ playerId: string; action: 'play' | 'pass'; cards: Card[]; handType: string }> = [];
    const turnLog: TurnRecord[] = [];
    const playedOutCards: Card[] = [];

    const decisionStats: Record<string, DecisionSourceCounts> = {};
    for (const eid of Object.values(playerEngines)) {
      if (!decisionStats[eid]) {
        decisionStats[eid] = { total: 0, onnxFallback: 0, fallback: 0, timeout: 0, error: 0, force: 0 };
      }
    }
    const invalidPlays: InvalidPlayRecord[] = [];

    while (game.getPhase() === GamePhase.Playing && turns < MAX_TURNS) {
      turns++;

      const currentPlayerId = game.getCurrentPlayerId()!;
      const currentPlayer = players.find((p) => p.id === currentPlayerId)!;

      const engineId = currentPlayer.aiEngine ?? ('rule_default' as AIEngineType);
      const engine = this.engines.get(engineId);
      if (!engine) throw new Error(`Engine not found: ${engineId}`);

      const ctx = this.buildContext(
        game, currentPlayerId, currentPlayer, players,
        actionHistory, playedOutCards, playerNames, seatOrder,
      );
      if (this.debugMode) {
        this.logDebugHands(`[Turn ${turns}][${engineId}] ${playerTag(currentPlayerId, currentPlayer.nickname)}`, players, ctx);
      }

      let decision: BotDecision;
      let decisionSource: TurnRecord['decisionSource'] = 'engine';
      const t0 = performance.now();
      let usedOnnxFallback = false;

      try {
        decision = await withTimeout(engine.decide(ctx), DECIDE_TIMEOUT_MS, `${engineId}.decide`);
        usedOnnxFallback = Boolean(decision.debug?.onnxFallbackReason);
      } catch (err: any) {
        const isTimeout = err.message?.includes('timeout');
        decisionSource = isTimeout ? 'timeout' : 'error';
        if (this.verbose) {
          console.warn(`  [Turn ${turns}] ${engineId} ${isTimeout ? 'TIMEOUT' : 'ERROR'}: ${err.message}`);
        }
        decision =
          ctx.isNewRound || ctx.isFirstTurn
            ? { action: 'play', cards: [sortCards(currentPlayer.cards).pop()!] }
            : { action: 'pass', cards: [] };
      }

      if (decision.action === 'play' && decision.cards.length > 0) {
        const validation = validatePlay(decision.cards, currentPlayer.cards, ctx.lastPlay, ctx.isFirstTurn);
        if (!validation.valid) {
          decisionSource = 'fallback';
          if (this.verbose) {
            console.warn(`  [Turn ${turns}] ${engineId} invalid play [${validation.error}], fallback to rule_default`);
            console.warn(`    Hand(${currentPlayer.cards.length}): ${currentPlayer.cards.map(c => `${c.suit[0]}${c.rank}`).join(' ')}`);
            console.warn(`    Tried: ${decision.cards.map(c => `${c.suit[0]}${c.rank}`).join(' ')}`);
            if (ctx.lastPlay) {
              console.warn(`    LastPlay: ${ctx.lastPlay.cards.map(c => `${c.suit[0]}${c.rank}`).join(' ')} (${ctx.lastPlay.type})`);
            } else {
              console.warn(`    LastPlay: (free)`);
            }
          }
          if (this.captureInvalidPlays) {
            const otherCardCounts: Record<string, number> = {};
            for (const [pid, cnt] of Object.entries(ctx.playerCardCounts)) {
              if (pid !== currentPlayerId) otherCardCounts[playerEngines[pid] ?? pid] = cnt;
            }
            invalidPlays.push({
              turn: turns,
              engineId,
              validationError: validation.error ?? 'unknown',
              attempted: [...decision.cards],
              playerHand: [...currentPlayer.cards],
              lastPlayCards: ctx.lastPlay ? [...ctx.lastPlay.cards] : null,
              lastPlayHandType: ctx.lastPlay?.type ?? null,
              isFirstTurn: ctx.isFirstTurn,
              isNewRound: ctx.isNewRound,
              otherCardCounts,
            });
          }
          const fallback = this.engines.get('rule_default' as AIEngineType);
          if (fallback && engineId !== 'rule_default') {
            try {
              decision = await withTimeout(fallback.decide(ctx), DECIDE_TIMEOUT_MS, 'rule_default.decide');
            } catch {
              decision = ctx.isNewRound || ctx.isFirstTurn
                ? { action: 'play', cards: [sortCards(currentPlayer.cards).pop()!] }
                : { action: 'pass', cards: [] };
              decisionSource = 'force';
            }
          }
        }
      }

      if (decision.action !== 'play' || decision.cards.length === 0) {
        if (ctx.isNewRound || ctx.isFirstTurn) {
          decision = this.forceFreePlay(currentPlayer.cards, ctx.isFirstTurn);
          decisionSource = 'force';
        }
      }

      const elapsed = performance.now() - t0;
      const cardsBeforePlay = currentPlayer.cards.length;

      // accumulate per-engine decision source stats
      {
        const ds = decisionStats[engineId]!;
        ds.total++;
        if (usedOnnxFallback) ds.onnxFallback++;
        if (decisionSource === 'fallback') ds.fallback++;
        else if (decisionSource === 'timeout') ds.timeout++;
        else if (decisionSource === 'error') ds.error++;
        else if (decisionSource === 'force') ds.force++;
      }

      if (decision.action === 'play' && decision.cards.length > 0) {
        consecutivePasses = 0;
        const hand = detectHand(decision.cards);
        const handType = hand?.type ?? '';
        actionHistory.push({
          playerId: currentPlayerId,
          action: 'play',
          cards: decision.cards,
          handType,
        });
        playedOutCards.push(...decision.cards);
        turnLog.push({
          turn: turns,
          playerId: currentPlayerId,
          engineId,
          action: 'play',
          cards: decision.cards,
          handType,
          cardsRemaining: cardsBeforePlay - decision.cards.length,
          decisionSource,
          isNewRound: ctx.isNewRound,
          isFirstTurn: ctx.isFirstTurn,
          elapsedMs: elapsed,
        });
        game.play(currentPlayerId, decision.cards);
      } else {
        consecutivePasses++;
        actionHistory.push({
          playerId: currentPlayerId,
          action: 'pass',
          cards: [],
          handType: '',
        });
        turnLog.push({
          turn: turns,
          playerId: currentPlayerId,
          engineId,
          action: 'pass',
          cards: [],
          handType: '',
          cardsRemaining: cardsBeforePlay,
          decisionSource,
          isNewRound: ctx.isNewRound,
          isFirstTurn: ctx.isFirstTurn,
          elapsedMs: elapsed,
        });
        game.pass(currentPlayerId);
      }

      if (consecutivePasses >= MAX_CONSECUTIVE_PASSES) {
        const stuckInfo = this.buildStuckInfo(
          `${consecutivePasses} consecutive passes`,
          turnLog, game, playerEngines, turns, consecutivePasses,
        );
        this.onStuckGame?.(stuckInfo);
        const finalState = game.getFullState();
        return {
          rankings: game.getRankings(),
          scores: finalState.scores ?? {},
          playerEngines,
          playerTeams,
          declarerPlayerId,
          declarerEngineId,
          isSolo: players.some((p) => p.team === TeamSide.Solo),
          turns,
          decisionStats: decisionStats as Partial<Record<AIEngineType, DecisionSourceCounts>>,
          invalidPlays: invalidPlays.length > 0 ? invalidPlays : undefined,
          error: `Stuck: ${consecutivePasses} consecutive passes`,
          stuckInfo,
        };
      }
    }

    if (turns >= MAX_TURNS && game.getPhase() === GamePhase.Playing) {
      const stuckInfo = this.buildStuckInfo(
        `Max turns (${MAX_TURNS}) exceeded`,
        turnLog, game, playerEngines, turns, consecutivePasses,
      );
      this.onStuckGame?.(stuckInfo);
      const finalState = game.getFullState();
      return {
        rankings: game.getRankings(),
        scores: finalState.scores ?? {},
        playerEngines,
        playerTeams,
        declarerPlayerId,
        declarerEngineId,
        isSolo: players.some((p) => p.team === TeamSide.Solo),
        turns,
        decisionStats: decisionStats as Partial<Record<AIEngineType, DecisionSourceCounts>>,
        invalidPlays: invalidPlays.length > 0 ? invalidPlays : undefined,
        error: `Max turns exceeded (${MAX_TURNS})`,
        stuckInfo,
      };
    }

    const finalState = game.getFullState();
    return {
      rankings: game.getRankings(),
      scores: finalState.scores ?? {},
      playerEngines,
      playerTeams,
      declarerPlayerId,
      declarerEngineId,
      isSolo: players.some((p) => p.team === TeamSide.Solo),
      turns,
      decisionStats: decisionStats as Partial<Record<AIEngineType, DecisionSourceCounts>>,
      invalidPlays: invalidPlays.length > 0 ? invalidPlays : undefined,
    };
  }

  async runDuel(
    engineA: AIEngineType,
    engineB: AIEngineType,
    totalGames: number,
    onProgress?: (completed: number, total: number) => void,
  ): Promise<GameResult[]> {
    const results: GameResult[] = [];
    const patterns: AIEngineType[][] = [
      [engineA, engineB, engineA, engineB],
      [engineB, engineA, engineB, engineA],
    ];

    for (let i = 0; i < totalGames; i++) {
      const pattern = patterns[i % patterns.length]!;
      const assignments: EngineAssignment[] = pattern.map((eng, seat) => ({
        seatIndex: seat,
        engineId: eng,
      }));
      const result = await this.runGame(assignments);
      results.push(result);
      onProgress?.(i + 1, totalGames);
    }
    return results;
  }

  async runFFA(
    engineIds: AIEngineType[],
    totalGames: number,
    onProgress?: (completed: number, total: number) => void,
  ): Promise<GameResult[]> {
    if (engineIds.length !== 4) throw new Error('FFA needs exactly 4 engines');
    const results: GameResult[] = [];

    for (let i = 0; i < totalGames; i++) {
      const rotated = [...engineIds];
      const shift = i % 4;
      for (let s = 0; s < shift; s++) rotated.push(rotated.shift()!);

      const assignments: EngineAssignment[] = rotated.map((eng, seat) => ({
        seatIndex: seat,
        engineId: eng,
      }));
      const result = await this.runGame(assignments);
      results.push(result);
      onProgress?.(i + 1, totalGames);
    }
    return results;
  }

  private buildStuckInfo(
    reason: string,
    turnLog: TurnRecord[],
    game: Game,
    playerEngines: Record<string, AIEngineType>,
    totalTurns: number,
    consecutivePasses: number,
  ): StuckGameInfo {
    const players = game.getPlayers();
    return {
      reason,
      lastTurns: turnLog.slice(-30),
      playerStates: players.map((p) => ({
        id: p.id,
        engineId: playerEngines[p.id]!,
        cards: p.cards || [],
        team: String(p.team),
        rank: p.rank,
      })),
      consecutivePasses,
      totalTurns,
    };
  }

  private buildContext(
    game: Game,
    botId: string,
    botPlayer: Player,
    players: Player[],
    actionHistory: Array<{ playerId: string; action: 'play' | 'pass'; cards: Card[]; handType: string }>,
    playedOutCards: Card[],
    playerNames: Record<string, string>,
    seatOrder: string[],
  ): BotContext {
    const lastPlay = game.getLastPlay();
    const lastHand = game.getLastPlayHand();
    const rankings = game.getRankings();
    const isFirstTurn = rankings.length === 0 && players.every((p) => p.cards.length === 13);
    const isNewRound = !lastPlay;
    const gameState = game.getFullState();
    const passCount = gameState.passCount;

    let teammateId: string | null = null;
    let teammateName: string | null = null;
    for (const p of players) {
      if (p.id === botId) continue;
      if (p.isTeamRevealed && p.team === botPlayer.team && botPlayer.team !== 'unknown') {
        teammateId = p.id;
        teammateName = p.nickname;
        break;
      }
    }

    const playerCardCounts: Record<string, number> = {};
    for (const p of players) {
      playerCardCounts[p.id] = p.cards.length;
    }

    const recentHistory = buildRecentHistory(actionHistory, playerNames);
    const { playedCardsByPlayer, observedTeams } = buildPlayedCardsAndTeams(actionHistory, seatOrder, gameState.declarerId);

    const currentRoundActions: ActionRecord[] = [];
    const actions = game.getPlayerActions();
    for (const pid of seatOrder) {
      const act = actions[pid];
      if (act && act.action !== 'none') {
        currentRoundActions.push({
          playerId: pid,
          playerName: playerNames[pid] ?? pid.substring(0, 8),
          action: act.action as 'play' | 'pass',
          cards: act.cards,
          handType: act.handType,
        });
      }
    }

    const allPlayersCards: Record<string, Card[]> = {};
    const allTeams: Record<string, TeamSide> = {};
    for (const p of players) {
      allPlayersCards[p.id] = p.cards;
      allTeams[p.id] = p.team;
    }

    const myTeam = botPlayer.team;
    let actualTeammateId: string | null = null;
    let teammateCards: Card[] | null = null;
    if (myTeam !== TeamSide.Solo && myTeam !== TeamSide.Unknown) {
      const activeTeammates = players.filter(
        (p) => p.id !== botId && p.team === myTeam && !rankings.includes(p.id),
      );
      if (activeTeammates.length > 0) {
        activeTeammates.sort((a, b) => a.cards.length - b.cards.length);
        actualTeammateId = activeTeammates[0]!.id;
        teammateCards = activeTeammates[0]!.cards;
      } else {
        const finishedMates = players.filter((p) => p.id !== botId && p.team === myTeam);
        if (finishedMates.length > 0) {
          actualTeammateId = finishedMates[0]!.id;
          teammateCards = [];
        }
      }
    }

    const godMode: GodModeInfo = {
      allPlayersCards,
      allTeams,
      myTeam,
      actualTeammateId,
      teammateCards,
    };

    return {
      myCards: botPlayer.cards,
      lastPlay: lastHand,
      lastPlayPlayerId: lastPlay?.playerId ?? null,
      isFirstTurn,
      isNewRound,
      myPlayerId: botId,
      myPlayerName: botPlayer.nickname,
      teammateId,
      teammateName,
      teammateRevealed: !!teammateId,
      playerCardCounts,
      playerNames,
      rankings,
      currentRoundActions,
      recentHistory,
      playedOutCards,
      playedCardsByPlayer,
      observedTeams,
      seatOrder,
      godMode,
      isDeclared: !!gameState.declarerId,
      declarerId: gameState.declarerId ?? null,
      passCount,
      debugMode: this.rlInferenceDebug,
      ruleConfig: {
        declareRuleMode: 'free',
        declareRequiresBothSpades: false,
        straightStartRank: '3',
        straightEndRank: 'A',
      },
    };
  }

  /**
   * 为报牌阶段构造最小化的 BotContext。
   * 报牌时游戏还没开始，大部分字段为空/默认值，只有手牌和基本信息有意义。
   */
  private buildDeclareContext(
    botPlayer: Player,
    players: Player[],
  ): BotContext {
    const seatOrder = [...players].sort((a, b) => a.seatIndex - b.seatIndex).map((p) => p.id);
    const playerCardCounts: Record<string, number> = {};
    const playerNames: Record<string, string> = {};
    const observedTeams: Record<string, TeamSide> = {};
    const playedCardsByPlayer: Record<string, Card[]> = {};
    for (const p of players) {
      playerCardCounts[p.id] = p.cards.length;
      playerNames[p.id] = p.nickname;
      observedTeams[p.id] = TeamSide.Unknown;
      playedCardsByPlayer[p.id] = [];
    }

    const godMode: GodModeInfo = {
      allPlayersCards: Object.fromEntries(players.map((p) => [p.id, p.cards])),
      allTeams: Object.fromEntries(players.map((p) => [p.id, p.team])),
      myTeam: botPlayer.team,
      actualTeammateId: null,
      teammateCards: null,
    };

    return {
      myCards: botPlayer.cards,
      lastPlay: null,
      lastPlayPlayerId: null,
      isFirstTurn: true,
      isNewRound: true,
      myPlayerId: botPlayer.id,
      myPlayerName: botPlayer.nickname,
      teammateId: null,
      teammateName: null,
      teammateRevealed: false,
      playerCardCounts,
      playerNames,
      rankings: [],
      currentRoundActions: [],
      recentHistory: [],
      playedOutCards: [],
      playedCardsByPlayer,
      observedTeams,
      seatOrder,
      godMode,
      isDeclaringPhase: true,
      isDeclared: false,
      declarerId: null,
      debugMode: this.rlInferenceDebug,
      ruleConfig: {
        declareRuleMode: 'free',
        declareRequiresBothSpades: false,
        straightStartRank: '3',
        straightEndRank: 'A',
      },
    };
  }

  private logDebugHands(prefix: string, players: Player[], ctx: BotContext): void {
    console.info(formatDebugHandsBlock(
      prefix,
      players.map((p) => ({
        id: p.id,
        nickname: p.nickname,
        seatIndex: p.seatIndex,
        engineId: p.aiEngine ?? undefined,
        team: String(p.team),
        observedTeam: String(ctx.observedTeams[p.id] ?? TeamSide.Unknown),
        count: p.cards.length,
        hand: p.cards,
      })),
    ));
  }

  private forceFreePlay(cards: Card[], isFirstTurn: boolean): BotDecision {
    if (isFirstTurn) {
      const d4 = cards.find((c) => c.suit === 'diamond' && c.rank === '4');
      if (d4) return { action: 'play', cards: [d4] };
    }
    const sorted = sortCards(cards);
    return { action: 'play', cards: [sorted[sorted.length - 1]!] };
  }
}

/**
 * 判断一手牌是否应该报牌（基于手牌强度的简单规则）。
 * 报牌触发条件：持有高价值牌达到阈值。
 *
 * 评分规则：3→5分, 2→4分, A→3分, K→2分，其余→1分
 * 阈值：总分 ≥ 35（约相当于持有多张高牌）
 */
function shouldDeclare(cards: Card[]): boolean {
  const rankScore: Record<string, number> = {
    '3': 5, '2': 4, 'A': 3, 'K': 2,
  };
  let score = 0;
  for (const c of cards) {
    score += rankScore[c.rank] ?? 1;
  }
  return score >= 35;
}
