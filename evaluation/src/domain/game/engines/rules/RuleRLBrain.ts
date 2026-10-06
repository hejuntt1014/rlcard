/**
 * RuleRLBrain — 强化学习 AI（DMC 训练 + ONNX 推理）V12
 *
 * ============================================================
 * 总体设计
 * ============================================================
 *
 * 本版本采用三输入 ONNX：
 *   - obs_static  [556]
 *   - hist_tokens [24, 88]
 *   - action_feat [111]
 *
 * 模型输出：
 *   - play_q:            1D   — 普通出牌阶段候选动作的 Q 值
 *   - aux_logits:       17D   — relation(9) + s3_owner(4) + sa_owner(4)
 *   - declare_q:         2D   — [pass_declare, declare]
 *
 * ============================================================
 * 关键原则
 * ============================================================
 *
 * 1) legal_actions 是 "当前轮可执行的具体动作集合"（含具体牌面），不是抽象牌型
 * 2) legal_actions 必须 canonicalize → 去重 → 固定排序
 * 3) 房间规则必须同时进入 obs_static、驱动 legal_actions、影响顺子特征
 * 4) afterstate 由代码对每个候选动作 "假执行" 后计算，不是二次推理
 * 5) 关系特征不能提前猜，只允许 unknown → same/opposite
 * 6) 相对位置顺序：slot 0=我, 1=下家, 2=对家, 3=上家
 *
 * ============================================================
 * 1) obs_static (556D)
 * ============================================================
 *
 * [0:52]      own_cards (52D)               — 我的当前手牌 multi-hot
 * [52:104]    current_top_action (52D)       — 当前需要压的牌 (自由出牌=全0)
 * [104:108]   current_top_actor (4D)         — 谁打出的
 * [108:160]   last_nonpass_action (52D)      — 最近一次非pass有效出牌
 * [160:164]   last_nonpass_actor (4D)        — 谁打出的
 * [164:372]   public_played_4p (4×52D=208D)  — 4人已打出的牌
 * [372:428]   remaining_count_4p (4×14D=56D) — 4人剩余牌数 one-hot
 * [428:444]   public_status_4p (4×4D=16D)    — [played_s3, played_sa, declared, finished]
 * [444:447]   game_mode_state (3D)           — [normal, solo, declarer]
 * [447:452]   self_side_state (5D)           — [unknown, a3_side, non_a3, declarer, anti_declarer]
 * [452:461]   other_relation (3×3D=9D)       — [unknown, same_side, opposite_side]
 * [461:513]   unseen_cards (52D)             — 记牌器
 * [513:525]   s3_sa_tracker (12D)            — ♠3/♠A 追踪
 * [525:537]   phase_flags (12D)              — 局面标志
 * [537:544]   room_rules (7D)               — 房间规则
 * [544:556]   threat_flags (12D)             — 终局威胁
 *
 * ============================================================
 * 2) hist_tokens (24 × 88D = 2112D)
 * ============================================================
 *
 * 每个 token = 88D:
 *   [0:4]   actor (4D one-hot)
 *   [4]     valid (1D)
 *   [5]     is_pass (1D)
 *   [6:15]  action_type (9D) — [single,pair,triple,straight,flush,full_house,four_with_one,straight_flush,pass]
 *   [15:28] main_rank (13D one-hot)
 *   [28:32] main_suit (4D one-hot)
 *   [32:36] action_len (4D) — [len1,len2,len3,len5]
 *   [36:88] card_bits (52D)
 *
 * ============================================================
 * 3) action_feat (111D)
 * ============================================================
 *
 * [0:52]    card_bits (52D)
 * [52:61]   action_type (9D one-hot)
 * [61:74]   main_rank (13D)
 * [74:78]   main_suit (4D)
 * [78:82]   action_len (4D)
 * [82:90]   special_flags (8D):
 *             [0] is_pass
 *             [1] contains_d4
 *             [2] contains_s3
 *             [3] contains_sa
 *             [4] reveals_self_identity — 打出♠3/♠A且身份尚未暴露
 *             [5] consumes_last_card   — 打完后0张
 *             [6] leaves_self_with_1   — 打完后1张
 *             [7] forced_nonpass_context — 当前不允许pass（只剩1张可压时）
 * [90:106]  afterstate_summary (16D):
 *             [0]  remaining_count (raw int)
 *             [1]  remaining_eq_1
 *             [2]  remaining_eq_2
 *             [3]  singles_count (raw)
 *             [4]  pairs_count (raw)
 *             [5]  triples_count (raw)
 *             [6]  fivecard_potential (raw)
 *             [7]  min_steps (raw, 启发式近似)
 *             [8]  still_has_s3
 *             [9]  still_has_sa
 *             [10] still_has_rank3_single
 *             [11] still_has_rank2_single
 *             [12] still_has_straight_potential
 *             [13] still_has_flush_potential
 *             [14] still_has_sf_potential
 *             [15] still_has_threepair_or_fourone
 * [106:111] rule_aware_straight (5D):
 *             [0] is_straight_family
 *             [1] touches_room_min
 *             [2] touches_room_max
 *             [3] is_room_max_straight
 *             [4] is_room_min_straight
 *
 * ============================================================
 * V12 已实现
 * ============================================================
 * - Transformer encoder 处理 hist_tokens 序列
 * - 单独 declare head（报牌阶段不再复用 play_q）
 */

import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { Suit, Rank, TeamSide, type Card, type RLInferenceDebug } from "@a3/shared";
import type { BotContext, BotDecision } from "../../BotBrain.js";
import { BaseRuleBrain } from "../BaseRuleBrain.js";
import { getPlayableHands, HandType, type Hand, type StraightRange } from "../../../hand/index.js";
import * as ort from "onnxruntime-node";
import { InferenceBatcher } from '../../../../inferenceBatcher.js';

// ─── Constants ──────────────────────────────────────────────────────────

const SUITS: Suit[] = [Suit.Diamond, Suit.Club, Suit.Heart, Suit.Spade];
const RANKS: Rank[] = [Rank.Four, Rank.Five, Rank.Six, Rank.Seven, Rank.Eight, Rank.Nine, Rank.Ten, Rank.Jack, Rank.Queen, Rank.King, Rank.Ace, Rank.Two, Rank.Three];

const CARD_TO_IDX: Record<string, number> = {};
for (let si = 0; si < SUITS.length; si++) {
  for (let ri = 0; ri < RANKS.length; ri++) {
    CARD_TO_IDX[`${SUITS[si]!}_${RANKS[ri]!}`] = si * 13 + ri;
  }
}

const STRAIGHT_RANK_VAL: Record<string, number> = {
  [Rank.Four]: 2, [Rank.Five]: 3, [Rank.Six]: 4, [Rank.Seven]: 5,
  [Rank.Eight]: 6, [Rank.Nine]: 7, [Rank.Ten]: 8, [Rank.Jack]: 9,
  [Rank.Queen]: 10, [Rank.King]: 11, [Rank.Ace]: 12, [Rank.Two]: 0, [Rank.Three]: 1,
};

const NUM_CARDS = 52;
const NUM_PLAYERS = 4;
const HISTORY_LEN = 24;
const HIST_TOKEN_DIM = 88;
const OBS_STATIC_DIM = 556;
const ACTION_FEAT_DIM = 111;
const STATE_DIM = 2668;  // 训练端仍保持 flat obs；推理端拆成三输入
const AUX_DIM = 17;
const NUM_ACTION_TYPES = 9;

const HAND_TYPE_MAP: Record<string, number> = {
  [HandType.Single]: 0, [HandType.Pair]: 1, [HandType.Triple]: 2,
  [HandType.Straight]: 3, [HandType.Flush]: 4, [HandType.FullHouse]: 5,
  [HandType.FourWithOne]: 6, [HandType.StraightFlush]: 7,
  "pass": 8,
};

const TEAMS_ORDER: string[] = [TeamSide.SpadeA3, TeamSide.Opponent, TeamSide.Solo, TeamSide.Unknown];

// ─── ONNX Model ────────────────────────────────────────────────────────

const MODEL_PATH = process.env.A3_EVAL_MODEL ?? join(dirname(fileURLToPath(import.meta.url)), "a3dizhu_model.onnx");
const COMPACT = process.env.A3_EVAL_LAYOUT !== 'legacy';

let sessionPromise: Promise<ort.InferenceSession> | null = null;

function getSession(): Promise<ort.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["cpu"],
      graphOptimizationLevel: "all",
      ...(process.env.A3_EVAL_LEGACY_THREADS === '1' ? {} : {
        intraOpNumThreads: Number(process.env.A3_EVAL_THREADS ?? 1),
        interOpNumThreads: 1,
        executionMode: 'sequential' as const,
      }),
    }).then((session) => {
      console.info(`[RuleRLBrain V12] ONNX model loaded — inputs: [${session.inputNames}], outputs: [${session.outputNames}]`);
      return session;
    });
  }
  return sessionPromise;
}

export async function initializeModel(): Promise<void> { await getSession(); }
const batcher = new InferenceBatcher(getSession);

function markOnnxFallback(decision: BotDecision, reason: string): BotDecision {
  return {
    ...decision,
    debug: {
      ...decision.debug,
      onnxFallbackReason: reason,
    },
  };
}

// ─── Card Utility ──────────────────────────────────────────────────────

function cardKey(c: Card): string { return `${c.suit}_${c.rank}`; }

function cardIdx(c: Card): number { return CARD_TO_IDX[cardKey(c)] ?? -1; }

function cardsToVec(cards: Card[], out: Float32Array, offset: number): void {
  for (const c of cards) {
    const idx = cardIdx(c);
    if (idx >= 0) out[offset + idx] = 1;
  }
}

function inferMyTeam(myCards: Card[], observedSelf: TeamSide | undefined): string {
  const hasS3 = myCards.some(c => c.suit === Suit.Spade && c.rank === Rank.Three);
  const hasSA = myCards.some(c => c.suit === Suit.Spade && c.rank === Rank.Ace);
  if (hasS3 && hasSA) return TeamSide.Solo;
  if (hasS3 || hasSA) return TeamSide.SpadeA3;
  if (observedSelf && observedSelf !== TeamSide.Unknown) return observedSelf;
  return TeamSide.Opponent;
}

function getDeclareRuleMode(ctx: BotContext): 'none' | 'free' | 'both_spades' {
  const rc = ctx.ruleConfig;
  if (rc?.declareRuleMode) return rc.declareRuleMode;
  return rc?.declareRequiresBothSpades ? 'both_spades' : 'free';
}

function getDeclarerInfo(ctx: BotContext): { isDeclared: boolean; declarerId: string | null } {
  const declarerId = ctx.declarerId ?? null;
  const isDeclared = !!ctx.isDeclared && !!declarerId;
  return { isDeclared, declarerId: isDeclared ? declarerId : null };
}

// ─── V11 Obs Encoding ──────────────────────────────────────────────────

function encodeObs(ctx: BotContext): Float32Array {
  const obs = new Float32Array(STATE_DIM);
  encodeObsStatic(ctx, obs, 0);
  encodeHistTokens(ctx, obs, OBS_STATIC_DIM);
  return obs;
}

function encodeObsStaticVec(ctx: BotContext): Float32Array {
  const out = new Float32Array(OBS_STATIC_DIM);
  encodeObsStatic(ctx, out, 0);
  return out;
}

function encodeHistTokensVec(ctx: BotContext): Float32Array {
  const out = new Float32Array(HISTORY_LEN * HIST_TOKEN_DIM);
  encodeHistTokens(ctx, out, 0);
  return out;
}

function repeatFlat(src: Float32Array, batchSize: number): Float32Array {
  const out = new Float32Array(src.length * batchSize);
  for (let i = 0; i < batchSize; i++) out.set(src, i * src.length);
  return out;
}

function encodeObsStatic(ctx: BotContext, obs: Float32Array, base: number): void {
  let o = base;
  const n = 4;
  const seatOrder = ctx.seatOrder;
  const myIdx = seatOrder.indexOf(ctx.myPlayerId);
  const relOrder = [0, 1, 2, 3].map(i => seatOrder[(myIdx + i) % n]!);

  const s3Key = `${Suit.Spade}_${Rank.Three}`;
  const saKey = `${Suit.Spade}_${Rank.Ace}`;
  const s3Idx = CARD_TO_IDX[s3Key]!;
  const saIdx = CARD_TO_IDX[saKey]!;
  const { isDeclared, declarerId } = getDeclarerInfo(ctx);

  // [0:52] own_cards
  cardsToVec(ctx.myCards, obs, o); o += NUM_CARDS;

  // [52:104] current_top_action
  if (ctx.lastPlay && !ctx.isNewRound && !ctx.isFirstTurn) {
    cardsToVec(ctx.lastPlay.cards, obs, o);
  }
  o += NUM_CARDS;

  // [104:108] current_top_actor
  if (ctx.lastPlayPlayerId && !ctx.isNewRound && !ctx.isFirstTurn) {
    const ri = relOrder.indexOf(ctx.lastPlayPlayerId);
    if (ri >= 0) obs[o + ri] = 1;
  }
  o += n;

  // [108:160] last_nonpass_action
  let lastNonpassCards: Card[] = [];
  let lastNonpassPlayer: string | null = null;
  for (let i = ctx.recentHistory.length - 1; i >= 0; i--) {
    const entry = ctx.recentHistory[i]!;
    if (entry.action === "play" && entry.cards.length > 0) {
      lastNonpassCards = entry.cards;
      lastNonpassPlayer = entry.playerId;
      break;
    }
  }
  cardsToVec(lastNonpassCards, obs, o); o += NUM_CARDS;

  // [160:164] last_nonpass_actor
  if (lastNonpassPlayer) {
    const ri = relOrder.indexOf(lastNonpassPlayer);
    if (ri >= 0) obs[o + ri] = 1;
  }
  o += n;

  // [164:372] public_played_4p (4×52D)
  for (const pid of relOrder) {
    cardsToVec(ctx.playedCardsByPlayer[pid] ?? [], obs, o);
    o += NUM_CARDS;
  }

  // [372:428] remaining_count_4p (4×14D)
  for (const pid of relOrder) {
    const cnt = Math.min(ctx.playerCardCounts[pid] ?? 0, 13);
    obs[o + cnt] = 1;
    o += 14;
  }

  // [428:444] public_status_4p (4×4D: played_s3, played_sa, declared, finished)
  const finishedSet = new Set(ctx.rankings);
  for (const pid of relOrder) {
    const played = ctx.playedCardsByPlayer[pid] ?? [];
    obs[o] = played.some(c => c.suit === Suit.Spade && c.rank === Rank.Three) ? 1 : 0;
    obs[o + 1] = played.some(c => c.suit === Suit.Spade && c.rank === Rank.Ace) ? 1 : 0;
    obs[o + 2] = declarerId === pid ? 1 : 0; // declared
    obs[o + 3] = finishedSet.has(pid) ? 1 : 0;
    o += 4;
  }

  // [444:447] game_mode_state (3D: normal, solo, declarer)
  const myActualTeam = ctx.godMode?.myTeam ?? inferMyTeam(ctx.myCards, ctx.observedTeams[ctx.myPlayerId]);
  const hasSolo = Object.values(ctx.observedTeams).includes(TeamSide.Solo) ||
                  Object.values(ctx.godMode?.allTeams ?? {}).includes(TeamSide.Solo);
  if (isDeclared) {
    obs[o + 2] = 1; // declarer mode
  } else if (hasSolo && myActualTeam === TeamSide.Solo) {
    obs[o + 1] = 1; // solo mode (I'm the solo player)
  } else {
    obs[o + 0] = 1; // normal team mode
  }
  o += 3;

  // [447:452] self_side_state (5D: unknown, a3, non_a3, declarer, anti_declarer)
  if (isDeclared) {
    if (ctx.myPlayerId === declarerId) {
      obs[o + 3] = 1; // declarer
    } else {
      obs[o + 4] = 1; // anti_declarer
    }
  } else if (myActualTeam === TeamSide.SpadeA3 || myActualTeam === TeamSide.Solo) {
    obs[o + 1] = 1; // a3_side
  } else if (myActualTeam === TeamSide.Opponent) {
    obs[o + 2] = 1; // non_a3_side
  } else {
    obs[o + 0] = 1; // unknown
  }
  o += 5;

  // [452:461] other_relation_to_self (3×3D: unknown, same, opposite)
  for (let j = 1; j <= 3; j++) {
    const other = relOrder[j]!;
    const otherObs = ctx.observedTeams[other] ?? TeamSide.Unknown;
    if (isDeclared) {
      if (ctx.myPlayerId === declarerId) {
        obs[o + 2] = 1; // everyone else opposite me
      } else if (other === declarerId) {
        obs[o + 2] = 1; // declarer is opposite
      } else {
        obs[o + 1] = 1; // other anti-declarers are same side
      }
    } else if (otherObs === TeamSide.Unknown) {
      obs[o] = 1; // unknown
    } else if (myActualTeam === otherObs) {
      obs[o + 1] = 1; // same_side
    } else {
      obs[o + 2] = 1; // opposite_side
    }
    o += 3;
  }

  // [461:513] unseen_cards (52D)
  const myHandSet = new Set(ctx.myCards.map(cardIdx));
  const playedSet = new Set(ctx.playedOutCards.map(cardIdx));
  for (let i = 0; i < NUM_CARDS; i++) {
    if (!myHandSet.has(i) && !playedSet.has(i)) obs[o + i] = 1;
  }
  o += NUM_CARDS;

  // [513:525] s3_sa_tracker (12D)
  obs[o++] = myHandSet.has(s3Idx) ? 1 : 0;
  obs[o++] = myHandSet.has(saIdx) ? 1 : 0;
  obs[o++] = playedSet.has(s3Idx) ? 1 : 0;
  obs[o++] = playedSet.has(saIdx) ? 1 : 0;
  // per-player s3
  for (const pid of relOrder) {
    obs[o++] = (ctx.playedCardsByPlayer[pid] ?? []).some(
      c => c.suit === Suit.Spade && c.rank === Rank.Three) ? 1 : 0;
  }
  // per-player sa
  for (const pid of relOrder) {
    obs[o++] = (ctx.playedCardsByPlayer[pid] ?? []).some(
      c => c.suit === Suit.Spade && c.rank === Rank.Ace) ? 1 : 0;
  }

  // [525:537] phase_flags (12D)
  const hasLast = ctx.lastPlay != null && !ctx.isNewRound && !ctx.isFirstTurn;
  const isFree = !hasLast && !ctx.isFirstTurn && !ctx.isDeclaringPhase;
  const mustPlay = ctx.myCards.length === 1 && hasLast;
  obs[o++] = ctx.isDeclaringPhase ? 1 : 0;     // [0] is_declaration_phase
  obs[o++] = ctx.isFirstTurn ? 1 : 0;           // [1] is_first_play_of_game
  obs[o++] = isFree ? 1 : 0;                    // [2] is_free_lead
  obs[o++] = (hasLast && !mustPlay) ? 1 : 0;    // [3] can_pass
  obs[o++] = mustPlay ? 1 : 0;                  // [4] must_play_if_possible
  obs[o++] = hasLast ? 1 : 0;                   // [5] has_current_top
  obs[o++] = (hasLast && ctx.lastPlay!.size === 1) ? 1 : 0;
  obs[o++] = (hasLast && ctx.lastPlay!.size === 2) ? 1 : 0;
  obs[o++] = (hasLast && ctx.lastPlay!.size === 3) ? 1 : 0;
  obs[o++] = (hasLast && ctx.lastPlay!.size === 5) ? 1 : 0;
  const passCount = ctx.passCount ?? 0;
  obs[o++] = passCount === 1 ? 1 : 0;
  obs[o++] = passCount === 2 ? 1 : 0;

  // [537:544] room_rules (7D)
  const rc = ctx.ruleConfig;
  const declareRuleMode = getDeclareRuleMode(ctx);
  obs[o++] = (!rc || rc.straightStartRank === "3") ? 1 : 0;
  obs[o++] = (rc?.straightStartRank === "4") ? 1 : 0;
  obs[o++] = (rc?.straightEndRank === "K") ? 1 : 0;
  obs[o++] = (!rc || rc.straightEndRank === "A") ? 1 : 0;
  obs[o++] = declareRuleMode === "none" ? 1 : 0;
  obs[o++] = declareRuleMode === "free" ? 1 : 0;
  obs[o++] = declareRuleMode === "both_spades" ? 1 : 0;

  // [544:556] threat_flags (12D)
  for (let j = 1; j <= 3; j++) {
    obs[o++] = (ctx.playerCardCounts[relOrder[j]!] ?? 13) === 1 ? 1 : 0;
  }
  for (let j = 1; j <= 3; j++) {
    obs[o++] = (ctx.playerCardCounts[relOrder[j]!] ?? 13) <= 2 ? 1 : 0;
  }
  for (let j = 1; j <= 3; j++) {
    obs[o++] = finishedSet.has(relOrder[j]!) ? 1 : 0;
  }
  let anyEq1 = false, anyLe2 = false;
  for (let j = 1; j <= 3; j++) {
    const cnt = ctx.playerCardCounts[relOrder[j]!] ?? 13;
    if (cnt === 1) anyEq1 = true;
    if (cnt <= 2) anyLe2 = true;
  }
  obs[o++] = anyEq1 ? 1 : 0;
  obs[o++] = anyLe2 ? 1 : 0;
  obs[o++] = ctx.myCards.length === 1 ? 1 : 0;
}

function encodeHistTokens(ctx: BotContext, obs: Float32Array, base: number): void {
  const seatOrder = ctx.seatOrder;
  const myIdx = seatOrder.indexOf(ctx.myPlayerId);
  const relOrder = [0, 1, 2, 3].map(i => seatOrder[(myIdx + i) % 4]!);

  const history = ctx.recentHistory.slice(-HISTORY_LEN);
  let o = base;

  for (const entry of history) {
    // actor (4D)
    const ri = relOrder.indexOf(entry.playerId);
    if (ri >= 0) obs[o + ri] = 1;
    o += 4;

    // valid
    obs[o++] = 1;

    const isPass = entry.action === "pass" || entry.action === "pass_declare";

    // is_pass
    obs[o++] = isPass ? 1 : 0;

    // action_type (9D)
    const typeKey = isPass ? "pass" : (entry.handType || "pass");
    const ati = HAND_TYPE_MAP[typeKey] ?? 8;
    if (ati >= 0 && ati < NUM_ACTION_TYPES) obs[o + ati] = 1;
    o += NUM_ACTION_TYPES;

    // main_rank (13D) — rank index of primary card
    if (!isPass && entry.cards.length > 0) {
      const primaryRank = getPrimaryRank(entry);
      if (primaryRank >= 0) obs[o + primaryRank] = 1;
    }
    o += 13;

    // main_suit (4D)
    if (!isPass && entry.cards.length > 0) {
      const shouldEncodeSuit = typeKey === HandType.Single || typeKey === HandType.Pair ||
        typeKey === HandType.Straight || typeKey === HandType.Flush || typeKey === HandType.StraightFlush;
      if (shouldEncodeSuit) {
        const primarySuit = getPrimarySuit(entry);
        if (primarySuit >= 0) obs[o + primarySuit] = 1;
      }
    }
    o += 4;

    // action_len (4D)
    if (!isPass) {
      const li = handSizeToLenIdx(entry.cards.length);
      if (li >= 0) obs[o + li] = 1;
    }
    o += 4;

    // card_bits (52D)
    if (!isPass) cardsToVec(entry.cards, obs, o);
    o += NUM_CARDS;
  }
  // Remaining tokens are already zero (padding, valid=0)
}

function getPrimaryRank(entry: { cards: Card[]; handType?: string }): number {
  if (entry.cards.length === 0) return -1;
  const isStraight = entry.handType === HandType.Straight || entry.handType === HandType.StraightFlush;
  if (isStraight) {
    let best = -1, bestSrv = -1;
    for (const c of entry.cards) {
      const srv = STRAIGHT_RANK_VAL[c.rank] ?? 0;
      const suitIdx = SUITS.indexOf(c.suit as Suit);
      if (srv > bestSrv || (srv === bestSrv && suitIdx > (SUITS.indexOf(entry.cards[best === -1 ? 0 : best]!.suit as Suit)))) {
        best = entry.cards.indexOf(c);
        bestSrv = srv;
      }
    }
    return best >= 0 ? (RANKS.indexOf(entry.cards[best]!.rank as Rank)) : -1;
  }
  // Normal: highest score card
  let bestScore = -1, bestRank = -1;
  for (const c of entry.cards) {
    const ri = RANKS.indexOf(c.rank as Rank);
    const si = SUITS.indexOf(c.suit as Suit);
    const score = ri * 10 + si;
    if (score > bestScore) { bestScore = score; bestRank = ri; }
  }
  return bestRank;
}

function getPrimarySuit(entry: { cards: Card[] }): number {
  let bestScore = -1, bestSuit = -1;
  for (const c of entry.cards) {
    const ri = RANKS.indexOf(c.rank as Rank);
    const si = SUITS.indexOf(c.suit as Suit);
    const score = ri * 10 + si;
    if (score > bestScore) { bestScore = score; bestSuit = si; }
  }
  return bestSuit;
}

function handSizeToLenIdx(size: number): number {
  if (size === 1) return 0;
  if (size === 2) return 1;
  if (size === 3) return 2;
  if (size === 5) return 3;
  return -1;
}

// ─── V11 Action Feature ────────────────────────────────────────────────

function encodeActionFeature(hand: Hand | null, ctx: BotContext): Float32Array {
  const feat = new Float32Array(ACTION_FEAT_DIM);
  let o = 0;

  // [0:52] card_bits
  if (hand) {
    for (const c of hand.cards) {
      const idx = cardIdx(c);
      if (idx >= 0) feat[idx] = 1;
    }
  }
  o += NUM_CARDS;

  // [52:61] action_type
  const isPass = !hand;
  if (isPass) {
    feat[o + 8] = 1; // pass
  } else if (hand) {
    const ati = HAND_TYPE_MAP[hand.type] ?? -1;
    if (ati >= 0) feat[o + ati] = 1;
  }
  o += NUM_ACTION_TYPES;

  // [61:74] main_rank
  if (hand && hand.primaryCard) {
    const ri = RANKS.indexOf(hand.primaryCard.rank as Rank);
    if (ri >= 0) feat[o + ri] = 1;
  }
  o += 13;

  // [74:78] main_suit
  if (hand && hand.primaryCard) {
    const encodeSuit = hand.type === HandType.Single || hand.type === HandType.Pair ||
      hand.type === HandType.Straight || hand.type === HandType.Flush || hand.type === HandType.StraightFlush;
    if (encodeSuit) {
      const si = SUITS.indexOf(hand.primaryCard.suit as Suit);
      if (si >= 0) feat[o + si] = 1;
    }
  }
  o += 4;

  // [78:82] action_len
  if (hand) {
    const li = handSizeToLenIdx(hand.size);
    if (li >= 0) feat[o + li] = 1;
  }
  o += 4;

  // [82:90] special_flags (8D)
  feat[o] = isPass ? 1 : 0;
  if (hand) {
    feat[o + 1] = hand.cards.some(c => c.suit === Suit.Diamond && c.rank === Rank.Four) ? 1 : 0;
    feat[o + 2] = hand.cards.some(c => c.suit === Suit.Spade && c.rank === Rank.Three) ? 1 : 0;
    feat[o + 3] = hand.cards.some(c => c.suit === Suit.Spade && c.rank === Rank.Ace) ? 1 : 0;
    // reveals_self_identity: only if not already public
    const s3AlreadyPublic = ctx.playedOutCards.some(c => c.suit === Suit.Spade && c.rank === Rank.Three);
    const saAlreadyPublic = ctx.playedOutCards.some(c => c.suit === Suit.Spade && c.rank === Rank.Ace);
    const reveals = (feat[o + 2] === 1 && !s3AlreadyPublic) || (feat[o + 3] === 1 && !saAlreadyPublic);
    feat[o + 4] = reveals ? 1 : 0;

    const handCardSet = new Set(hand.cards.map(cardIdx));
    const remaining = ctx.myCards.filter(c => !handCardSet.has(cardIdx(c)));
    feat[o + 5] = remaining.length === 0 ? 1 : 0;
    feat[o + 6] = remaining.length === 1 ? 1 : 0;

    // forced_nonpass_context: free lead or only 1 card left
    const hasLast = ctx.lastPlay != null && !ctx.isNewRound && !ctx.isFirstTurn;
    const cantPass = !hasLast || (ctx.myCards.length === 1 && hasLast);
    feat[o + 7] = cantPass ? 1 : 0;

    // [90:106] afterstate_summary (16D)
    const as_ = computeAfterstate(remaining, ctx);
    feat[o + 8] = Math.min(as_.remainingCount, 13);
    feat[o + 9] = as_.remainingCount === 1 ? 1 : 0;
    feat[o + 10] = as_.remainingCount === 2 ? 1 : 0;
    feat[o + 11] = Math.min(as_.singles, 13);
    feat[o + 12] = Math.min(as_.pairs, 6);
    feat[o + 13] = Math.min(as_.triples, 4);
    feat[o + 14] = Math.min(as_.fivecardPotential, 10);
    feat[o + 15] = Math.min(as_.minSteps, 13);
    feat[o + 16] = as_.hasS3 ? 1 : 0;
    feat[o + 17] = as_.hasSA ? 1 : 0;
    feat[o + 18] = as_.hasRank3Single ? 1 : 0;
    feat[o + 19] = as_.hasRank2Single ? 1 : 0;
    feat[o + 20] = as_.hasStraightPotential ? 1 : 0;
    feat[o + 21] = as_.hasFlushPotential ? 1 : 0;
    feat[o + 22] = as_.hasSFPotential ? 1 : 0;
    feat[o + 23] = as_.hasThreePairOrFourOne ? 1 : 0;
  }
  o += 8 + 16; // special_flags(8) + afterstate(16)

  // [106:111] rule_aware_straight (5D)
  if (hand && (hand.type === HandType.Straight || hand.type === HandType.StraightFlush)) {
    feat[o] = 1; // is_straight_family
    const srvs = hand.cards.map(c => STRAIGHT_RANK_VAL[c.rank] ?? 0);
    const minSrv = Math.min(...srvs);
    const maxSrv = Math.max(...srvs);
    const rc = ctx.ruleConfig;
    const roomMin = rc?.straightStartRank === "4" ? 2 : 1;
    const roomMax = rc?.straightEndRank === "K" ? 11 : 12;
    feat[o + 1] = minSrv === roomMin ? 1 : 0;
    feat[o + 2] = maxSrv === roomMax ? 1 : 0;
    feat[o + 3] = (maxSrv === roomMax && minSrv === roomMax - 4) ? 1 : 0;
    feat[o + 4] = (minSrv === roomMin && maxSrv === roomMin + 4) ? 1 : 0;
  }

  return feat;
}

interface AfterstateResult {
  remainingCount: number;
  singles: number; pairs: number; triples: number;
  fivecardPotential: number; minSteps: number;
  hasS3: boolean; hasSA: boolean;
  hasRank3Single: boolean; hasRank2Single: boolean;
  hasStraightPotential: boolean; hasFlushPotential: boolean;
  hasSFPotential: boolean; hasThreePairOrFourOne: boolean;
}

const STRAIGHT_SEQS = [
  [12,0,1,2,3],[0,1,2,3,4],[1,2,3,4,5],[2,3,4,5,6],
  [3,4,5,6,7],[4,5,6,7,8],[5,6,7,8,9],[6,7,8,9,10],
];

function computeAfterstate(remaining: Card[], ctx: BotContext): AfterstateResult {
  const rankCount = new Array(13).fill(0) as number[];
  const suitCount = new Array(4).fill(0) as number[];
  for (const c of remaining) {
    const ri = RANKS.indexOf(c.rank as Rank);
    const si = SUITS.indexOf(c.suit as Suit);
    if (ri >= 0) rankCount[ri]++;
    if (si >= 0) suitCount[si]++;
  }

  let singles = 0, pairs = 0, triples = 0, quads = 0;
  for (let r = 0; r < 13; r++) {
    if (rankCount[r] === 1) singles++;
    else if (rankCount[r] === 2) pairs++;
    else if (rankCount[r] === 3) triples++;
    else if (rankCount[r] >= 4) quads++;
  }

  const rc = ctx.ruleConfig;
  const roomMin = rc?.straightStartRank === "4" ? 2 : 1;
  const roomMax = rc?.straightEndRank === "K" ? 11 : 12;

  let hasStraight = false;
  for (const seq of STRAIGHT_SEQS) {
    const vals = seq.map(r => STRAIGHT_RANK_VAL[RANKS[r]!] ?? 0);
    if (Math.min(...vals) < roomMin || Math.max(...vals) > roomMax) continue;
    if (seq.every(r => rankCount[r]! > 0)) { hasStraight = true; break; }
  }

  const hasFlush = suitCount.some(c => c >= 5);
  let hasSF = false;
  if (hasStraight && hasFlush) {
    // Simplified check
    for (let s = 0; s < 4; s++) {
      if (suitCount[s]! < 5) continue;
      const suitRanks = new Set<number>();
      for (const c of remaining) {
        if (SUITS.indexOf(c.suit as Suit) === s) {
          suitRanks.add(RANKS.indexOf(c.rank as Rank));
        }
      }
      for (const seq of STRAIGHT_SEQS) {
        const vals = seq.map(r => STRAIGHT_RANK_VAL[RANKS[r]!] ?? 0);
        if (Math.min(...vals) < roomMin || Math.max(...vals) > roomMax) continue;
        if (seq.every(r => suitRanks.has(r))) { hasSF = true; break; }
      }
      if (hasSF) break;
    }
  }

  let fc = 0;
  if (hasStraight) fc++;
  if (hasFlush) fc++;
  if (hasSF) fc++;
  const hasFullHouse = triples >= 2 || (triples > 0 && pairs > 0);
  const hasThreePairOrFourOne = (COMPACT ? hasFullHouse :
    ((triples > 0 && pairs > 0) || (triples > 0 && singles > 0))) || (quads > 0 && remaining.length >= 5);
  if (hasThreePairOrFourOne) fc++;

  return {
    remainingCount: remaining.length,
    singles, pairs, triples,
    fivecardPotential: fc,
    minSteps: Math.max(singles + pairs + triples + quads, remaining.length > 0 ? 1 : 0),
    hasS3: remaining.some(c => c.suit === Suit.Spade && c.rank === Rank.Three),
    hasSA: remaining.some(c => c.suit === Suit.Spade && c.rank === Rank.Ace),
    hasRank3Single: rankCount[12] === 1, // rank 12 = '3'
    hasRank2Single: rankCount[11] === 1, // rank 11 = '2'
    hasStraightPotential: hasStraight,
    hasFlushPotential: hasFlush,
    hasSFPotential: hasSF,
    hasThreePairOrFourOne,
  };
}

// ─── Inference ─────────────────────────────────────────────────────────

async function selectBestAction(
  session: ort.InferenceSession,
  obsStatic: Float32Array,
  histTokens: Float32Array,
  legalHands: Hand[],
  canPass: boolean,
  ctx: BotContext
): Promise<{ hand: Hand | null; qValue: number; auxLogits?: Float32Array }> {
  const actions: (Hand | null)[] = [...legalHands];
  if (canPass) actions.push(null);
  if (actions.length === 0) return { hand: null, qValue: Number.NEGATIVE_INFINITY };
  if (actions.length === 1) return { hand: actions[0] ?? null, qValue: Number.NaN };

  const batchSize = actions.length;
  const actionData = new Float32Array(batchSize * ACTION_FEAT_DIM);
  for (let i = 0; i < batchSize; i++) {
    actionData.set(encodeActionFeature(actions[i] ?? null, ctx), i * ACTION_FEAT_DIM);
  }

  if (COMPACT) {
    const result = await batcher.infer(obsStatic, histTokens, actionData);
    let best = 0;
    for (let i = 1; i < result.q.length; ++i) if (result.q[i]! > result.q[best]!) best = i;
    return { hand: actions[best] ?? null, qValue: result.q[best]!, auxLogits: result.auxiliary };
  }

  const results = await session.run({
    obs_static: new ort.Tensor("float32", COMPACT ? obsStatic : repeatFlat(obsStatic, batchSize), [COMPACT ? 1 : batchSize, OBS_STATIC_DIM]),
    hist_tokens: new ort.Tensor("float32", COMPACT ? histTokens : repeatFlat(histTokens, batchSize), [COMPACT ? 1 : batchSize, HISTORY_LEN, HIST_TOKEN_DIM]),
    action_feat: new ort.Tensor("float32", actionData, [batchSize, ACTION_FEAT_DIM]),
  });
  const qValues = results["play_q"]!.data as Float32Array;
  const auxValues = results["aux_logits"]?.data as Float32Array | undefined;

  let bestIdx = 0, bestQ = qValues[0]!;
  for (let i = 1; i < qValues.length; i++) {
    if (qValues[i]! > bestQ) { bestQ = qValues[i]!; bestIdx = i; }
  }

  let bestAux: Float32Array | undefined;
  if (auxValues && auxValues.length > 0) {
    const auxDim = COMPACT ? auxValues.length : Math.floor(auxValues.length / batchSize);
    bestAux = COMPACT ? auxValues : auxValues.slice(bestIdx * auxDim, (bestIdx + 1) * auxDim);
  }
  return { hand: actions[bestIdx] ?? null, qValue: bestQ, auxLogits: bestAux };
}

// ─── Aux Decode & Debug ────────────────────────────────────────────────

type GuessLabel = "A3" | "OPP" | "SOLO";
type GuessProb = { label: "下家"|"对家"|"上家"; a3: number; opponent: number; solo: number; mostLikely: GuessLabel };

function softmax3(a: number, b: number, c: number): [number, number, number] {
  const m = Math.max(a, b, c);
  const ea = Math.exp(a - m), eb = Math.exp(b - m), ec = Math.exp(c - m);
  const sum = ea + eb + ec;
  return [ea / sum, eb / sum, ec / sum];
}

function decodeAuxTeamGuess(auxLogits?: Float32Array): GuessProb[] {
  if (!auxLogits || auxLogits.length < 9) return [];
  const labels: GuessProb["label"][] = ["下家", "对家", "上家"];
  return labels.map((label, i) => {
    const base = i * 3;
    const [a3, opponent, solo] = softmax3(auxLogits[base]??0, auxLogits[base+1]??0, auxLogits[base+2]??0);
    const mostLikely: GuessLabel = a3 >= opponent && a3 >= solo ? "A3" : opponent >= solo ? "OPP" : "SOLO";
    return { label, a3, opponent, solo, mostLikely };
  });
}

function formatPct(v: number): string { return `${(v*100).toFixed(1)}%`; }

function buildRLInferenceDebug(actionLabel: string, auxLogits?: Float32Array, qValue?: number): RLInferenceDebug | undefined {
  if (process.env.A3_EVAL_DIAGNOSTICS !== '1') return undefined;
  const guesses = decodeAuxTeamGuess(auxLogits);
  if (guesses.length === 0) return undefined;
  const summary = guesses.map(g => `${g.label}:${g.mostLikely} A3=${formatPct(g.a3)} OPP=${formatPct(g.opponent)} SOLO=${formatPct(g.solo)}`).join(" | ");
  return {
    actionLabel,
    qValue: typeof qValue === "number" && Number.isFinite(qValue) ? qValue : null,
    summary,
    guesses,
  };
}

// ─── Brain ─────────────────────────────────────────────────────────────

function getStraightRange(ctx: BotContext): StraightRange | undefined {
  const rc = ctx.ruleConfig;
  if (!rc) return undefined;
  if (rc.straightStartRank === "3" && rc.straightEndRank === "A") return undefined;
  return {
    minRank: (rc.straightStartRank === "4" ? Rank.Four : Rank.Three) as Rank,
    maxRank: (rc.straightEndRank === "K" ? Rank.King : Rank.Ace) as Rank,
  };
}

export const __test__ = {
  computeAfterstate,
  encodeObsStaticVec,
  encodeHistTokensVec,
  encodeActionFeature,
  getDeclareRuleMode,
  getDeclarerInfo,
};

export default class RuleRLBrain extends BaseRuleBrain {
  readonly name = "RL-DMC";
  readonly displayName = "强化学习AI";
  readonly engineId = "rule_rl" as const;

  async decide(ctx: BotContext): Promise<BotDecision> {
    if (ctx.isDeclaringPhase) return this._decideDeclaration(ctx);
    if (ctx.isFirstTurn) return this._decideFirstTurn(ctx);

    try {
      const session = await getSession();
      const obsStatic = encodeObsStaticVec(ctx);
      const histTokens = encodeHistTokensVec(ctx);
      const lastPlay = ctx.isNewRound ? null : ctx.lastPlay;
      const legalHands = getPlayableHands(ctx.myCards, lastPlay, getStraightRange(ctx));
      if (legalHands.length === 0) return { action: "pass", cards: [] };

      const canPass = !ctx.isNewRound && !(ctx.myCards.length === 1 && legalHands.length > 0);
      const best = await selectBestAction(session, obsStatic, histTokens, legalHands, canPass, ctx);
      const rlInference = buildRLInferenceDebug(
        best.hand ? best.hand.cards.map(cardKey).join("|") : "pass", best.auxLogits, best.qValue);

      if (!best.hand) return { action: "pass", cards: [], debug: rlInference ? { rlInference } : undefined };
      return { action: "play", cards: best.hand.cards, debug: rlInference ? { rlInference } : undefined };
    } catch (err) {
      if (process.env.A3_EVAL_STRICT === '1') throw err;
      console.warn("[RuleRLBrain V12] ONNX inference error, falling back:", err);
      return markOnnxFallback(this.getConservativeFollowUp(ctx), "play_inference_error");
    }
  }

  private async _decideDeclaration(ctx: BotContext): Promise<BotDecision> {
    try {
      const session = await getSession();
      const obsStatic = encodeObsStaticVec(ctx);
      const histTokens = encodeHistTokensVec(ctx);
      const dummyAction = new Float32Array(ACTION_FEAT_DIM);
      if (COMPACT) {
        const result = await batcher.infer(obsStatic, histTokens, dummyAction);
        return { action: result.declaration[1]! > result.declaration[0]! ? 'declare' : 'pass_declare', cards: [] };
      }
      const results = await session.run({
        obs_static: new ort.Tensor("float32", obsStatic, [1, OBS_STATIC_DIM]),
        hist_tokens: new ort.Tensor("float32", histTokens, [1, HISTORY_LEN, HIST_TOKEN_DIM]),
        action_feat: new ort.Tensor("float32", dummyAction, [1, ACTION_FEAT_DIM]),
      });
      const declareQ = results["declare_q"]!.data as Float32Array;

      if ((declareQ[1] ?? -Infinity) > (declareQ[0] ?? -Infinity)) {
        return { action: "declare", cards: [] };
      }
      return { action: "pass_declare", cards: [] };
    } catch (err) {
      if (process.env.A3_EVAL_STRICT === '1') throw err;
      console.warn("[RuleRLBrain V12] Declaration ONNX error:", err);
      return markOnnxFallback({ action: "pass_declare", cards: [] }, "declare_inference_error");
    }
  }

  private async _decideFirstTurn(ctx: BotContext): Promise<BotDecision> {
    try {
      const session = await getSession();
      const obsStatic = encodeObsStaticVec(ctx);
      const histTokens = encodeHistTokensVec(ctx);
      const allHands = getPlayableHands(ctx.myCards, null, getStraightRange(ctx));
      const openingHands = allHands.filter(h =>
        h.cards.some(c => c.suit === Suit.Diamond && c.rank === Rank.Four));
      if (openingHands.length === 0) return this.playFirstTurn(ctx.myCards);

      const best = await selectBestAction(session, obsStatic, histTokens, openingHands, false, ctx);
      const rlInference = buildRLInferenceDebug(
        best.hand ? best.hand.cards.map(cardKey).join("|") : "first_turn_fallback", best.auxLogits, best.qValue);
      if (!best.hand) {
        const fallback = this.playFirstTurn(ctx.myCards);
        return { ...fallback, debug: rlInference ? { rlInference } : undefined };
      }
      return { action: "play", cards: best.hand.cards, debug: rlInference ? { rlInference } : undefined };
    } catch (err) {
      if (process.env.A3_EVAL_STRICT === '1') throw err;
      console.warn("[RuleRLBrain V12] first-turn error:", err);
      return markOnnxFallback(this.playFirstTurn(ctx.myCards), "first_turn_inference_error");
    }
  }
}
