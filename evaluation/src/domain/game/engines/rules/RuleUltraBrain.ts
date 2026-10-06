/**
 * RuleUltraBrain — 完整复刻参考AI出牌逻辑（自包含，无外部转换层依赖）
 *
 * 将参考项目的 logic / packCard / partnerInfo / AI 四大模块全部内联，
 * 所有逻辑自包含在单文件中。内部使用 hex 编码表示牌
 * （方块2=0x01 ... 黑桃A=0x3D），仅在 decide() 入口/出口处做 Card 对象转换。
 *
 * 算法要点：
 * 1. 手牌拆分: 将13张牌拆为 pack[0]=散牌 + pack[1..n]=5张组合
 * 2. 敌友判断: 根据黑桃A/3的归属判定4个座位的敌友关系
 * 3. 出牌决策: 根据牌型、敌友、手数等综合决策最优出牌
 */

import { TeamSide, Suit, Rank, type Card } from '@a3/shared';
import type { BotContext, BotDecision } from '../../BotBrain.js';
import { BaseRuleBrain } from '../BaseRuleBrain.js';
import { HandType, detectHand, type Hand } from '../../../hand/index.js';
import { isSpadeAce, isSpadeThree } from '../../../card/index.js';

// ═══════════════════════════════════════════════════════════════════
//  内部编码系统: Card <-> hex code
//  花色高4位: 方块(Diamond)=0x00, 梅花(Club)=0x10, 红桃(Heart)=0x20, 黑桃(Spade)=0x30
//  牌面低4位: 2=0x01, 3=0x02, 4=0x03, ..., K=0x0C, A=0x0D
// ═══════════════════════════════════════════════════════════════════

const RANK_TO_NIBBLE: Record<string, number> = {
  '2': 0x01, '3': 0x02, '4': 0x03, '5': 0x04, '6': 0x05,
  '7': 0x06, '8': 0x07, '9': 0x08, '10': 0x09,
  'J': 0x0A, 'Q': 0x0B, 'K': 0x0C, 'A': 0x0D,
};
const NIBBLE_TO_RANK: Record<number, Rank> = {
  0x01: Rank.Two, 0x02: Rank.Three, 0x03: Rank.Four, 0x04: Rank.Five,
  0x05: Rank.Six, 0x06: Rank.Seven, 0x07: Rank.Eight, 0x08: Rank.Nine,
  0x09: Rank.Ten, 0x0A: Rank.Jack, 0x0B: Rank.Queen, 0x0C: Rank.King,
  0x0D: Rank.Ace,
};
const SUIT_TO_HIGH: Record<string, number> = {
  [Suit.Diamond]: 0x00, [Suit.Club]: 0x10, [Suit.Heart]: 0x20, [Suit.Spade]: 0x30,
};
const HIGH_TO_SUIT: Suit[] = [Suit.Diamond, Suit.Club, Suit.Heart, Suit.Spade];
const HAND_TYPE_MAP: Record<string, number> = {
  [HandType.Single]: 1, [HandType.Pair]: 2, [HandType.Triple]: 3,
  [HandType.Straight]: 4, [HandType.Flush]: 5, [HandType.FullHouse]: 6,
  [HandType.FourWithOne]: 7, [HandType.StraightFlush]: 8,
};

function toCode(card: Card): number {
  return SUIT_TO_HIGH[card.suit]! + RANK_TO_NIBBLE[card.rank]!;
}

function toCard(code: number): Card {
  return { suit: HIGH_TO_SUIT[(code & 0xF0) / 0x10]!, rank: NIBBLE_TO_RANK[code & 0x0F]! };
}

function toCodeArray(cards: Card[]): number[] {
  return cards.map(toCode);
}

function toCardArray(codes: number[]): Card[] {
  return codes.map(toCard);
}

function toCardType(type: string): number {
  return HAND_TYPE_MAP[type] ?? 0;
}

// ═══════════════════════════════════════════════════════════════════
//  顺子起止配置
//  minRunStart=4 → 最小顺子从逻辑值4开始 (即4-5-6-7-8)
//  maxRunStart=10 → 最大顺子从逻辑值10开始 (即10-J-Q-K-A)
// ═══════════════════════════════════════════════════════════════════

const minRunStart = 4;
const maxRunStart = 10;

// ═══════════════════════════════════════════════════════════════════
//  数据结构定义
// ═══════════════════════════════════════════════════════════════════

interface PackedCard {
  type: number;
  card: number[];
}

interface RefData {
  selfStation: number;
  tableCardData: number[][];
  lastOutCard: { card: number[] | null; cardType: number; station: number };
  iDoubleNTPeople: number;
  i3UpGradePeople: number;
  iAUpGradePeople: number;
  tableUserID: number[];
  A3IsOut: boolean;
  deskPassword: string;
  round: number;
  oneOnOneStation: number | null;
  deskConfig: number[];
}

interface PartnerResult {
  partnerInfo: number[];
  early: boolean;
  lastIsFriend: boolean;
  lastIsEnemy: boolean;
  lastEnemyCard: number[] | null;
  lastEnemyStation: number;
  nextIsEnemy: boolean;
  nextIsFriend: boolean;
  nextIsDoubltNT: boolean;
  nextCard: number[];
  lessPlayer: number;
  myFriendIsRealMan: boolean;
  doubleNTIsEnemy: boolean;
  selfIsDoubleA3: boolean;
}

// ═══════════════════════════════════════════════════════════════════
//  Logic 模块 — 牌值计算、逻辑数组操作、出牌查找
//  核心思想: 每张牌用 "逻辑值"(4~16) 表示大小排序，4最小、3最大
//  逻辑数组 logicArray[v] = 手中逻辑值为v的牌的数量
// ═══════════════════════════════════════════════════════════════════

const logic = {
  /** 牌型编号，用于 switch/case 分发 */
  cardType: {
    自己出牌: 0, 单张: 1, 对子: 2, 三张: 3,
    顺子: 4, 同花: 5, 三带二: 6, 四带一: 7, 同花顺: 8,
  } as const,

  /** 特殊牌的 hex 编码 */
  cardCode: {
    方片A: 0x0D, 方片2: 0x01, 方片3: 0x02, 方片4: 0x03,
    方片5: 0x04, 方片J: 0x0A, 黑桃A: 0x3D, 黑桃3: 0x32,
  } as const,

  /** 逻辑值常量: L4=4(最小) ... LA=14(A) ... L3=16(最大) */
  cardLogic: {
    L4: 4, L5: 5, L6: 6, L7: 7, L8: 8, L9: 9,
    L10: 10, LJ: 11, LQ: 12, LK: 13, LA: 14, L2: 15, L3: 16,
  } as const,

  maxScore: 10,

  getCardString(cards: number[]): string[] {
    const cardsStr: string[] = [];
    cards.forEach(element => {
      let str = '';
      switch (this.getColorValue(element)) {
        case 0: str += '方块'; break;
        case 1: str += '梅花'; break;
        case 2: str += '红桃'; break;
        case 3: str += '黑桃'; break;
        default: str += '错误牌值:' + element; break;
      }
      switch (this.getCardValue(element)) {
        case 1: str += 'A'; break;
        case 11: str += 'J'; break;
        case 12: str += 'Q'; break;
        case 13: str += 'K'; break;
        default: str += this.getCardValue(element); break;
      }
      cardsStr.push(str);
    });
    return cardsStr;
  },

  /** 将牌数组转为 { code: true } 的哈希表，用于 O(1) 查找某张牌是否存在 */
  getAllCardObject(cards: number[]): Record<number, boolean> {
    const obj: Record<number, boolean> = {};
    cards.forEach(element => { obj[element] = true; });
    return obj;
  },

  /**
   * 检查 card 是否是全场最大的牌（没有其他任何人能压过）
   * 遍历所有其他人的手牌，看是否存在同类型且更大的牌
   */
  checkCardIsBigger(card: number[], allCard: number[][]): boolean {
    if (card.length == 0) return false;
    if (card.length < 5) {
      for (let i = 0; i < allCard.length; i++) {
        if (allCard[i] == null || allCard[i]!.length == 0) continue;
        const otherLogicArray = this.getLogicArray(allCard[i]!);
        for (let ii = this.getLogicValue(card[0]!); ii < otherLogicArray.length; ii++) {
          if (ii > this.getLogicValue(card[0]!)) {
            if (otherLogicArray[ii]! >= card.length) return false;
            continue;
          }
          if (otherLogicArray[ii]! >= card.length && ii == this.getLogicValue(card[0]!)) {
            const otherCardObj = this.getAllCardObject(allCard[i]!);
            for (let checkColor = this.getColorValue(card[card.length - 1]!); checkColor < 4; checkColor++) {
              const checkCode = this.getCardCode(checkColor, ii);
              if (otherCardObj[checkCode]) return false;
            }
          }
        }
      }
    }
    return true;
  },

  /** 获取牌面数值: A→1, 2→2, 3→3, 4→4, ..., K→13 */
  getCardValue(card: number): number {
    let value = card & 0x0F;
    if (value == 0x0D) value = 0;
    value = value + 1;
    return value;
  },

  /**
   * 获取逻辑排序值（决定牌的大小顺序）
   * 4=4, 5=5, ..., K=13, A=14, 2=15, 3=16
   * 值越大牌越大，3是最大的
   */
  getLogicValue(card: number): number {
    if (card == 0) return 0;
    let value = this.getCardValue(card);
    if (value <= 3) value += 13;
    return value;
  },

  /** 根据花色索引和逻辑值反推牌的 hex 编码 */
  getCardCode(color: number, logicValue: number): number {
    if (logicValue > 14) logicValue -= 13;
    logicValue -= 1;
    return 0x10 * color + logicValue;
  },

  /** 获取花色索引: 方块=0, 梅花=1, 红桃=2, 黑桃=3 */
  getColorValue(card: number): number {
    return (card & 0xF0) / 0x10;
  },

  /**
   * 生成逻辑值频率数组
   * logicArray[v] = 手中逻辑值为v的牌的数量
   * 例如: 手持两个K → logicArray[13] = 2
   */
  getLogicArray(cards: number[]): number[] {
    const data = new Array(this.getLogicValue(2) + 1);
    for (let i = 0; i < this.getLogicValue(2) + 1; i++) data[i] = 0;
    cards.forEach(element => {
      const lv = this.getLogicValue(element);
      data[lv]++;
    });
    return data;
  },

  /** 生成花色频率数组: colorArray[花色索引] = 该花色的牌数 */
  getColorArray(cards: number[]): number[] {
    const data = new Array(4);
    for (let i = 0; i < 4; i++) data[i] = 0;
    cards.forEach(element => {
      const color = this.getColorValue(element);
      data[color]++;
    });
    return data;
  },

  /** 从 cards 中取出 count 张逻辑值为 outLogicVal 的牌（会修改原数组！） */
  outLogicCard(cards: number[], outLogicVal: number, count?: number): number[] {
    count = !count ? 1 : count;
    let outCount = 0;
    const outCards: number[] = [];
    for (let i = 0; i < cards.length; i++) {
      if (this.getLogicValue(cards[i]!) == outLogicVal) {
        outCards.push(cards[i]!);
        cards.splice(i, 1);
        outCount++;
        i--;
        if (outCount >= count) break;
        continue;
      }
    }
    return outCards;
  },

  /** 从 cards 中取出 count 张花色为 outColor 的牌（会修改原数组！） */
  outColorCard(cards: number[], outColor: number, count?: number): number[] {
    count = !count ? 1 : count;
    let outCount = 0;
    const outCards: number[] = [];
    for (let i = 0; i < cards.length; i++) {
      if (this.getColorValue(cards[i]!) == outColor) {
        outCards.push(cards[i]!);
        cards.splice(i, 1);
        outCount++;
        i--;
        if (outCount >= count) break;
        continue;
      }
    }
    return outCards;
  },

  /** 从 cards 中取出编码为 cardCodeValue 的那张牌（会修改原数组！） */
  outCodeCard(cards: number[], cardCodeValue: number): number[] {
    for (let i = 0; i < cards.length; i++) {
      if (cards[i] == cardCodeValue) {
        const card = cards[i]!;
        cards.splice(i, 1);
        return [card];
      }
    }
    return [];
  },

  /** 从 cards 中批量移除 cardCodeArray 里所有牌（会修改原数组！） */
  outCodeCardArray(cards: number[], cardCodeArray: number[]): void {
    for (let index = 0; index < cardCodeArray.length; index++) {
      const cardCodeValue = cardCodeArray[index]!;
      for (let i = 0; i < cards.length; i++) {
        if (cards[i] == cardCodeValue) {
          cards.splice(i, 1);
          break;
        }
      }
    }
  },

  /**
   * 查找所有出现 checkCount 次的逻辑值
   * 例如 checkCount=2 → 找所有对子; checkCount=4 → 找所有四条
   * include23: 是否包含2和3（大牌通常不参与拆分）
   */
  checkSameCard(card: number[], checkCount: number, logicArray?: number[], include23?: boolean): number[] {
    if (!logicArray) logicArray = this.getLogicArray(card);
    const data: number[] = [];
    for (let i = 0; i < logicArray.length; i++) {
      if (!include23 && (i == this.getLogicValue(this.cardCode.方片2) || i == this.getLogicValue(this.cardCode.方片3))) continue;
      if (logicArray[i] == checkCount) data.push(i);
    }
    return data;
  },

  /**
   * 从小到大查找能压过 lastOutCard 的牌
   * @param minLogic 最小逻辑值限制
   * @param maxLogic 最大逻辑值限制
   * @param brokenPair 是否允许拆对子/三条来出牌
   * @param iAFirst 是否优先出黑桃A
   */
  outLookingCard(cards: number[], pack0LogicArray: number[] | null, minLogic: number | null, maxLogic: number | null, lastOutCard: number[], brokenPair?: boolean, iAFirst?: boolean): number[] {
    if (!pack0LogicArray) pack0LogicArray = this.getLogicArray(cards);
    if (!minLogic) minLogic = -1;
    if (!maxLogic) maxLogic = pack0LogicArray.length;
    minLogic = this.getLogicValue(lastOutCard[0]!) > minLogic ? this.getLogicValue(lastOutCard[0]!) : minLogic;
    const lastOutLength = lastOutCard.length;

    for (let i = minLogic; i < maxLogic; i++) {
      if (pack0LogicArray[i]! < lastOutLength) continue;
      // brokenPair=false 时只找数量恰好匹配的（不拆牌）
      if (!brokenPair && pack0LogicArray[i] != lastOutLength) continue;
      if (i > this.getLogicValue(lastOutCard[0]!)) {
        if (iAFirst && i == this.cardLogic.LA && lastOutLength == 1) {
          if (this.getAllCardObject(cards)[this.cardCode.黑桃A]) return [this.cardCode.黑桃A];
        }
        return this.outLogicCard(cards, i, lastOutLength);
      }
      // 同逻辑值时，比较花色大小
      const cardObj = this.getAllCardObject(cards);
      const lastMaxColor = this.getColorValue(lastOutCard[lastOutCard.length - 1]!);
      for (let color = lastMaxColor + 1; color < 4; color++) {
        if (i == this.cardLogic.LA && lastOutLength == 1) {
          if (this.getAllCardObject(cards)[this.cardCode.黑桃A]) return [this.cardCode.黑桃A];
        }
        const nextColorCard = 0x10 * color + lastOutCard[0]! % 0x10;
        if (cardObj[nextColorCard]) {
          const outCard = this.outCodeCard(cards, nextColorCard);
          if (outCard.length < lastOutLength) {
            outCard.push(...this.outLogicCard(cards, i, lastOutLength - 1));
          }
          return outCard;
        }
      }
      continue;
    }
    return [];
  },

  /** 从大到小查找能压过 lastOutCard 的牌（优先出大牌版本） */
  outLookingCardDESC(cards: number[], pack0LogicArray: number[] | null, minLogic: number | null, maxLogic: number | null, lastOutCard: number[], brokenPair?: boolean, iAFirst?: boolean): number[] {
    if (!pack0LogicArray) pack0LogicArray = this.getLogicArray(cards);
    if (!minLogic) minLogic = -1;
    if (!maxLogic) maxLogic = pack0LogicArray.length;
    minLogic = this.getLogicValue(lastOutCard[0]!) > minLogic ? this.getLogicValue(lastOutCard[0]!) : minLogic;
    const lastOutLength = lastOutCard.length;

    for (let i = maxLogic - 1; i >= minLogic; i--) {
      if (pack0LogicArray[i]! < lastOutLength) continue;
      if (!brokenPair && pack0LogicArray[i] != lastOutLength) continue;
      if (i > this.getLogicValue(lastOutCard[0]!)) {
        if (iAFirst && i == this.cardLogic.LA && lastOutLength == 1) {
          if (this.getAllCardObject(cards)[this.cardCode.黑桃A]) return [this.cardCode.黑桃A];
        }
        return this.outLogicCard(cards, i, lastOutLength);
      }
      const cardObj = this.getAllCardObject(cards);
      const lastMaxColor = this.getColorValue(lastOutCard[lastOutCard.length - 1]!);
      for (let color = lastMaxColor + 1; color < 4; color++) {
        if (i == this.cardLogic.LA && lastOutLength == 1) {
          if (this.getAllCardObject(cards)[this.cardCode.黑桃A]) return [this.cardCode.黑桃A];
        }
        const nextColorCard = 0x10 * color + lastOutCard[0]! % 0x10;
        if (cardObj[nextColorCard]) {
          const outCard = this.outCodeCard(cards, nextColorCard);
          if (outCard.length < lastOutLength) {
            outCard.push(...this.outLogicCard(cards, i, lastOutLength - 1));
          }
          return outCard;
        }
      }
      continue;
    }
    return [];
  },

  /**
   * 计算散牌的"手数"（需要几手才能出完 pack[0] 里的散牌）
   * 同时找出最大的一手牌，判断它是否全场最大
   */
  getHandCardCount(packCard: PackedCard[], pack0LogicArray: number[], _checkNum: number, allCard: number[][], selfStation: number) {
    const result = {
      handCount: 0 as number,
      handCards: [] as number[][],
      mybestCard: [] as number[],
      bestCardIsBigger: false,
    };
    const tmpPack0 = packCard[0]!.card.concat();
    for (let i = 0; i < pack0LogicArray.length; i++) {
      if (pack0LogicArray[i] != 0) {
        const num = pack0LogicArray[i] == 4 ? 3 : pack0LogicArray[i]!;
        const cards = this.outLogicCard(tmpPack0, i, num);
        if (pack0LogicArray[i] == 4) {
          result.handCards.push(this.outLogicCard(tmpPack0, i, 1));
        }
        result.handCards.push(cards);
        result.mybestCard = cards;
      }
    }
    // 总手数 = 散牌手数 + 5张组合数
    result.handCount = result.handCards.length + packCard.length - 1;
    if (result.handCards.length > 0) {
      const tmpAllCard = allCard.concat();
      tmpAllCard[selfStation] = [];
      result.bestCardIsBigger = this.checkCardIsBigger(result.mybestCard, tmpAllCard);
    }
    return result;
  },

  /**
   * 判断是否应该为了压单张而出牌
   * 统计比 singleCard 小的单张数量，如果 < 2 则不该浪费大牌去压
   */
  runOutForSingle(cardObj: Record<number, boolean>, cardLogicArray: number[], singleCard: number): boolean {
    let smallCard = 0;
    for (let i = 0; i < cardLogicArray.length; i++) {
      if (cardLogicArray[i] != 1) continue;
      if (i < this.getLogicValue(singleCard)) smallCard++;
      if (i == this.getLogicValue(singleCard)) {
        let checkCard = singleCard - 0x10;
        while (checkCard > 0) {
          if (cardObj[checkCard]) { smallCard++; break; }
          checkCard -= 0x10;
        }
      }
    }
    return smallCard < 2;
  },

  /**
   * 计算同花/顺子组合的评分
   * 分数越高 → 使用的牌越"不值钱"，越适合组成5张组合
   * 小单张被消耗加分，大牌被消耗扣分
   */
  getColorOrRunScore(cards: number[], packCards: number[] | number[][]): number {
    const allCardLogicArray = this.getLogicArray(cards);
    const flatPack = Array.isArray(packCards[0]) ? (packCards as number[][]).flat() : packCards as number[];
    const allPackLogicArray = this.getLogicArray(flatPack);
    let score = 0;
    for (let i = 0; i < allPackLogicArray.length; i++) {
      if (allPackLogicArray[i] == 0) continue;
      if (i <= this.cardLogic.L10) {
        if (allCardLogicArray[i] == 1) score += 3;
        continue;
      }
      if (i <= this.cardLogic.LK) {
        if (allCardLogicArray[i] == 1) score += 2;
        if (allCardLogicArray[i] == 2) score += 1;
        continue;
      }
      if (i <= this.cardLogic.L2) { score -= 1; continue; }
      if (i == this.cardLogic.L3) { score -= 2; continue; }
    }
    return score;
  },

  /** 检查是否可以组成顺子，返回得分最高的顺子的逻辑值数组(长度5)或空数组 */
  checkRun(cards: number[], logicArray?: number[]): number[] {
    if (cards.length < 5) return [];
    if (!logicArray) logicArray = this.getLogicArray(cards);
    let maxScore = -100;
    let maxScoreRun: number[] = [];
    for (let i = minRunStart; i <= maxRunStart; i++) {
      let faild = false;
      const run: number[] = [];
      for (let ii = i; ii < i + 5; ii++) {
        if (logicArray[ii] == 0) {
          i = ii;
          faild = true;
          break;
        }
        run.push(ii);
      }
      if (!faild) {
        const tempCard = cards.concat();
        const tempRunCard: number[][] = [];
        run.forEach(element => {
          tempRunCard.push(this.outLogicCard(tempCard, element, 1));
        });
        const score = this.getColorOrRunScore(cards, tempRunCard);
        if (score > maxScore) {
          maxScore = score;
          maxScoreRun = run.concat();
        }
      }
    }
    return maxScoreRun;
  },

  /**
   * 比较两手牌大小（仅用于单张和对子）
   * 先比逻辑值，相同则比花色
   */
  compareCard(card0: number[], card1: number[]): boolean {
    if (card0.length != card1.length) return false;
    if (card0.length <= 2) {
      if (this.getLogicValue(card0[0]!) != this.getLogicValue(card1[0]!)) {
        return this.getLogicValue(card0[0]!) > this.getLogicValue(card1[0]!);
      }
      if (card0.length == 1) {
        return this.getColorValue(card0[0]!) > this.getColorValue(card1[0]!);
      } else {
        return Math.max(this.getColorValue(card0[0]!), this.getColorValue(card0[1]!)) >
               Math.max(this.getColorValue(card1[0]!), this.getColorValue(card1[1]!));
      }
    }
    return false;
  },
};

// ═══════════════════════════════════════════════════════════════════
//  Pack 模块 — 手牌拆分与组牌策略
//  核心思想: 将手牌拆分为 pack[0]=散牌 + pack[1..n]=5张组合
//  5张组合优先级: 同花顺 > 四带一 > 三带二 > 同花 > 顺子
//  拆牌目标: 尽量把小散牌塞进5张组合中，减少手数
// ═══════════════════════════════════════════════════════════════════

const pack = {
  /**
   * 核心拆牌函数
   * 输入: 一组牌(hex编码数组)
   * 输出: PackedCard数组, [0]=散牌, [1..n]=各种5张组合
   * 拆牌按优先级依次尝试: 四带一×2→四带一+同花/顺→三带二+同花/顺→
   * 同花×2→三带二×2→顺子+同花→顺子×2→同花顺→单组同花/顺→四条→三条
   * 最后为四条补单张、三条补对子组成三带二
   */
  packCard(cards: number[]): PackedCard[] {
    const packed: PackedCard[] = [];
    let index = 0;
    packed[index] = { type: logic.cardType.单张, card: [] };
    packed[index]!.card = cards.concat();
    index++;

    let lessCardObj: Record<number, boolean> = {};
    let logicArray: number[] = [];
    let colors: number[] = [];
    let colorValue = [0, 0, 0, 0];
    let fourNum: number[] = [];
    let threeNum: number[] = [];

    // 重新分析剩余散牌的状态
    const checkCardInfo = (checkCard: number[]) => {
      lessCardObj = logic.getAllCardObject(checkCard);
      logicArray = logic.getLogicArray(checkCard);
      colors = [];
      colorValue = logic.getColorArray(checkCard);
      for (let i = colorValue.length - 1; i >= 0; i--) {
        if (colorValue[i]! >= 5) colors.push(i);
      }
      fourNum = logic.checkSameCard(checkCard, 4, logicArray);
      threeNum = logic.checkSameCard(checkCard, 3, logicArray);
    };

    checkCardInfo(packed[0]!.card);

    // ─── 两个四带一 ───
    if (packed[0]!.card.length >= 10) {
      if (fourNum.length == 2) {
        for (let ii = 0; ii < fourNum.length; ii++) {
          packed[index] = { type: logic.cardType.四带一, card: [] };
          packed[index]!.card = logic.outLogicCard(packed[0]!.card, fourNum[ii]!, 4);
          index++;
        }
        checkCardInfo(packed[0]!.card);
      }
    }

    // ─── 四带一 + 同花或顺子 ───
    let tempPacked = this.packFourOrThreeWithColorOrRun(4, packed[0]!.card, fourNum, colors);
    if (tempPacked.length != 0) {
      packed[index++] = tempPacked[0]!;
      packed[index++] = tempPacked[1]!;
      checkCardInfo(packed[0]!.card);
    }

    // ─── 三带二 + 同花或顺子 ───
    tempPacked = this.packFourOrThreeWithColorOrRun(3, packed[0]!.card, threeNum, colors);
    if (tempPacked.length != 0) {
      packed[index++] = tempPacked[0]!;
      packed[index++] = tempPacked[1]!;
      checkCardInfo(packed[0]!.card);
    }

    // ─── 两个同花 ───
    if (packed[0]!.card.length >= 10 && colors.length == 2) {
      for (let i = 0; i < colors.length; i++) {
        packed[index] = { type: logic.cardType.同花, card: [] };
        packed[index]!.card = logic.outColorCard(packed[0]!.card, colors[i]!, 5);
        index++;
      }
      checkCardInfo(packed[0]!.card);
    }

    // ─── 两个三带二 ───
    if (packed[0]!.card.length >= 6) {
      if (threeNum.length == 2) {
        for (let ii = 0; ii < threeNum.length; ii++) {
          packed[index] = { type: logic.cardType.三张, card: [] };
          packed[index]!.card = logic.outLogicCard(packed[0]!.card, threeNum[ii]!, 3);
          index++;
        }
        checkCardInfo(packed[0]!.card);
      }
    }

    // ─── 一个同花 + 一个顺子（不冲突地拆分） ───
    if (packed[0]!.card.length >= 10 && colors.length == 1) {
      const sameColor = colors[0]!;
      const lessColor = colorValue[sameColor]! - 5;
      for (let i = minRunStart; packed[0]!.card.length >= 5 && i <= maxRunStart; i++) {
        let faild = false;
        let useColorCount = 0;
        const runUseCard: number[] = [];
        for (let ii = i; ii < i + 5; ii++) {
          if (logicArray[ii] == 0) { i = ii; faild = true; break; }
          let useColor = false;
          let useCard = 0x00;
          for (let checkColor = sameColor; checkColor < (sameColor + 4); checkColor++) {
            const checkCard = logic.getCardCode(checkColor % 4, ii);
            if (lessCardObj[checkCard]) {
              useCard = checkCard;
              useColor = checkColor == sameColor;
            }
          }
          runUseCard.push(useCard);
          if (useColor) {
            useColorCount++;
            if (useColorCount > lessColor) { faild = true; break; }
          }
        }
        if (!faild) {
          packed[index] = { type: logic.cardType.顺子, card: [] };
          for (let ii = 0; ii < runUseCard.length; ii++) {
            packed[index]!.card.push(...logic.outCodeCard(packed[0]!.card, runUseCard[ii]!));
          }
          logicArray = logic.getLogicArray(packed[0]!.card);
          index++;
          packed[index] = { type: logic.cardType.同花, card: [] };
          packed[index]!.card = logic.outColorCard(packed[0]!.card, sameColor, 5);
          index++;
          checkCardInfo(packed[0]!.card);
        }
      }
    }

    // ─── 两个顺子 ───
    if (packed[0]!.card.length >= 10) {
      const tempCard = packed[0]!.card.concat();
      let tempLA = logic.getLogicArray(tempCard);
      const run1 = logic.checkRun(tempCard, tempLA);
      if (run1.length == 5) {
        for (let ii = 0; ii < run1.length; ii++) logic.outLogicCard(tempCard, run1[ii]!);
        tempLA = logic.getLogicArray(tempCard);
        const run2 = logic.checkRun(tempCard, tempLA);
        if (run2.length == 5) {
          packed[index] = { type: logic.cardType.顺子, card: [] };
          for (let ii = 0; ii < run1.length; ii++) packed[index]!.card.push(...logic.outLogicCard(packed[0]!.card, run1[ii]!));
          index++;
          packed[index] = { type: logic.cardType.顺子, card: [] };
          for (let ii = 0; ii < run2.length; ii++) packed[index]!.card.push(...logic.outLogicCard(packed[0]!.card, run2[ii]!));
          index++;
          checkCardInfo(packed[0]!.card);
        }
      }
    }

    // ─── 同花顺 ───
    for (let value = minRunStart; value <= maxRunStart; value++) {
      if (packed[0]!.card.length < 5) break;
      for (let color = 0; color < 4; color++) {
        if (packed[0]!.card.length < 5) break;
        let faild = false;
        for (let ii = 0; ii < 5; ii++) {
          if (!lessCardObj[logic.getCardCode(color, value + ii)]) { faild = true; break; }
        }
        if (!faild) {
          packed[index] = { type: logic.cardType.同花顺, card: [] };
          for (let ii = 0; ii < 5; ii++) {
            packed[index]!.card.push(...logic.outCodeCard(packed[0]!.card, logic.getCardCode(color, value + ii)));
          }
          index++;
          checkCardInfo(packed[0]!.card);
        }
      }
    }

    // ─── 查找单组最优同花或顺子 ───
    {
      let colorMaxScore = 0;
      let colorCards: number[] = [];
      let runMaxScore = 0;
      let runCards: number[] = [];

      if (colors.length > 0) {
        const color = colors[0]!;
        // 优先选单张小牌组成同花，减少散牌手数
        colorCards = this.getColorCard(color, logicArray, lessCardObj, 1, 5, null, logic.cardLogic.L10);
        colorCards.push(...this.getColorCard(color, logicArray, lessCardObj, 1, 5 - colorCards.length, logic.cardLogic.L10, logic.cardLogic.LA));
        colorCards.push(...this.getColorCard(color, logicArray, lessCardObj, 2, 5 - colorCards.length, null, logic.cardLogic.LA));
        colorCards.push(...this.getColorCard(color, logicArray, lessCardObj, 3, 5 - colorCards.length, null, logic.cardLogic.LA));
        colorCards.push(...this.getColorCard(color, logicArray, lessCardObj, null, 5 - colorCards.length, logic.cardLogic.LA, logic.cardLogic.L3));
        colorCards.push(...this.getColorCard(color, logicArray, lessCardObj, null, 5 - colorCards.length, logic.cardLogic.L3, null));
        if (colorCards.length < 5) {
          colorCards = [];
        } else {
          colorMaxScore = logic.getColorOrRunScore(packed[0]!.card, colorCards);
        }
      }

      const runLogic = logic.checkRun(packed[0]!.card, logicArray);
      if (runLogic.length == 5) {
        const tmpCard = packed[0]!.card.concat();
        for (let ii = 0; ii < runLogic.length; ii++) {
          runCards.push(...logic.outLogicCard(tmpCard, runLogic[ii]!));
        }
        runMaxScore = logic.getColorOrRunScore(packed[0]!.card, runCards);
      }

      // 选得分更高的组合（消耗更多小散牌的更优）
      if (runMaxScore > colorMaxScore && runMaxScore >= 5) {
        packed[index] = { type: logic.cardType.顺子, card: runCards };
        index++;
        logic.outCodeCardArray(packed[0]!.card, runCards);
        checkCardInfo(packed[0]!.card);
      }
      if (colorMaxScore > runMaxScore && colorMaxScore >= 5) {
        packed[index] = { type: logic.cardType.同花, card: colorCards };
        index++;
        logic.outCodeCardArray(packed[0]!.card, colorCards);
        checkCardInfo(packed[0]!.card);
      }
    }

    // ─── 四条（不够5张时降为三条） ───
    if (fourNum.length > 0) {
      if (packed[0]!.card.length >= 5) {
        packed[index] = { type: logic.cardType.四带一, card: [] };
        packed[index]!.card = logic.outLogicCard(packed[0]!.card, fourNum[0]!, 4);
        index++;
      } else {
        packed[index] = { type: logic.cardType.三张, card: [] };
        packed[index]!.card = logic.outLogicCard(packed[0]!.card, fourNum[0]!, 3);
        index++;
      }
      checkCardInfo(packed[0]!.card);
    }

    // ─── 三条 ───
    if (threeNum.length > 0) {
      packed[index] = { type: logic.cardType.三张, card: [] };
      packed[index]!.card = logic.outLogicCard(packed[0]!.card, threeNum[0]!, 3);
      index++;
      checkCardInfo(packed[0]!.card);
    }

    // ─── 给四条补单张、给三条补对子组成三带二 ───
    let haveP4 = false;
    packed.forEach(element => {
      if (element.type == logic.cardType.四带一 && element.card.length == 4) {
        haveP4 = false;
        for (let i = 0; i < logicArray.length; i++) {
          if (logicArray[i] == 1) {
            if (i == logic.cardLogic.L4 && lessCardObj[logic.cardCode.方片4]) { haveP4 = true; continue; }
            if (haveP4 && i >= logic.cardLogic.L2) {
              element.card.push(...logic.outCodeCard(packed[0]!.card, logic.cardCode.方片4));
              checkCardInfo(packed[0]!.card);
              break;
            }
            element.card.push(...logic.outLogicCard(packed[0]!.card, i, 1));
            checkCardInfo(packed[0]!.card);
            break;
          }
        }
        if (element.card.length == 4 && haveP4) {
          element.card.push(...logic.outLogicCard(packed[0]!.card, 4, 1));
          checkCardInfo(packed[0]!.card);
        }
        if (element.card.length == 4) {
          for (let i = 0; i < logicArray.length; i++) {
            if (logicArray[i]! > 1) {
              element.card.push(...logic.outLogicCard(packed[0]!.card, i, 1));
              checkCardInfo(packed[0]!.card);
              break;
            }
          }
        }
      }
      if (element.type == logic.cardType.三张 && element.card.length == 3) {
        for (let i = 0; i < logic.cardLogic.L2; i++) {
          if (logicArray[i]! >= 2) {
            if (i == logic.cardLogic.L4 && lessCardObj[logic.cardCode.方片4]) { continue; }
            if (haveP4 && i >= logic.cardLogic.LA) {
              element.card.push(...logic.outLogicCard(packed[0]!.card, 4, 2));
              checkCardInfo(packed[0]!.card);
              break;
            }
            element.type = logic.cardType.三带二;
            element.card.push(...logic.outLogicCard(packed[0]!.card, i, 2));
            checkCardInfo(packed[0]!.card);
            break;
          }
        }
      }
    });

    // ─── 2个三条合并为1个三带二 + 1个单张回散牌 ───
    let threeCount = 0;
    let lastIndex = -1;
    for (let i = 1; i < packed.length; i++) {
      const element = packed[i]!;
      if (element.type == logic.cardType.三张) {
        threeCount++;
        if (threeCount == 2) {
          const pair = logic.outLogicCard(packed[i]!.card, logic.getLogicValue(packed[i]!.card[0]!), 2);
          const single = logic.outLogicCard(packed[i]!.card, logic.getLogicValue(packed[i]!.card[0]!), 1);
          packed.splice(i, 1);
          packed[lastIndex]!.type = logic.cardType.三带二;
          packed[lastIndex]!.card.push(...pair);
          packed[0]!.card.push(...single);
          break;
        }
        lastIndex = i;
      }
    }

    // ─── 顺子最大牌换更大花色（提升顺子战斗力） ───
    packed.forEach(ele => {
      if (ele.type == logic.cardType.顺子) {
        const oldCard = ele.card[ele.card.length - 1]!;
        if (logicArray[logic.getLogicValue(oldCard)]! > 0) {
          for (let i = 3; i > logic.getColorValue(oldCard); i--) {
            const cardCode = logic.getCardCode(i, logic.getLogicValue(oldCard));
            if (lessCardObj[cardCode]) {
              logic.outCodeCard(packed[0]!.card, cardCode);
              ele.card[ele.card.length - 1] = cardCode;
              packed[0]!.card.push(oldCard);
              break;
            }
          }
        }
      }
    });

    return packed.sort(this.comparePack);
  },

  /** 比较两个组合的大小（用于排序和判断能否压过） */
  comparePack(pack1: PackedCard, pack2: PackedCard): number {
    const compType = pack1.type - pack2.type;
    const cardType = pack1.type;
    if (compType == 0) {
      if (cardType == logic.cardType.三带二 || cardType == logic.cardType.四带一) {
        // Bug fix: card[0] is not guaranteed to be the triple/quad when pack comes from
        // an unsorted lastOutCard. Scan cards to find the main group's logic value.
        const mainCount = cardType == logic.cardType.三带二 ? 3 : 4;
        const getMainValue = (cards: number[]): number => {
          const freq = new Map<number, number>();
          for (const c of cards) {
            const v = logic.getLogicValue(c);
            freq.set(v, (freq.get(v) ?? 0) + 1);
          }
          for (const [v, cnt] of freq) {
            if (cnt >= mainCount) return v;
          }
          return logic.getLogicValue(cards[0]!);
        };
        return getMainValue(pack1.card) - getMainValue(pack2.card);
      }
      pack1.card.sort((a, b) => (logic.getLogicValue(a) * 0x10 + logic.getColorValue(a)) - (logic.getLogicValue(b) * 0x10 + logic.getColorValue(b)));
      pack2.card.sort((a, b) => (logic.getLogicValue(a) * 0x10 + logic.getColorValue(a)) - (logic.getLogicValue(b) * 0x10 + logic.getColorValue(b)));
      const pack1BigCard = pack1.card[pack1.card.length - 1]!;
      const pack2BigCard = pack2.card[pack2.card.length - 1]!;
      if (cardType != logic.cardType.同花 && cardType != logic.cardType.同花顺) {
        if (cardType == logic.cardType.顺子) {
          if (logic.getLogicValue(pack1BigCard) == logic.getLogicValue(pack2BigCard)) {
            return logic.getColorValue(pack1BigCard) - logic.getColorValue(pack2BigCard);
          }
        }
        return logic.getLogicValue(pack1.card[pack1.card.length - 1]!) - logic.getLogicValue(pack2.card[pack2.card.length - 1]!);
      }
      // 同花/同花顺: 不同花色时高花色赢 (♠>♥>♣>♦)，同花色才比最大单张点数。
      // card[0] 在同花手牌里与其他牌同花色，所以取花色是安全的。
      const color1 = logic.getColorValue(pack1.card[0]!);
      const color2 = logic.getColorValue(pack2.card[0]!);
      if (color1 !== color2) return color1 - color2;
      return logic.getLogicValue(pack1BigCard) - logic.getLogicValue(pack2BigCard);
    }
    return compType;
  },

  /** 从指定花色中按条件选取牌（用于组同花） */
  getColorCard(color: number, logicArray: number[], lessCardObj: Record<number, boolean>, sameNum: number | null, max: number, minLogicValue: number | null, maxLogicValue: number | null): number[] {
    if (max <= 0) return [];
    if (!maxLogicValue) maxLogicValue = logicArray.length;
    if (!minLogicValue) minLogicValue = logic.cardLogic.L4;
    const colorCards: number[] = [];
    for (let i = minLogicValue; i < maxLogicValue; i++) {
      if (sameNum && logicArray[i] != sameNum) continue;
      if (lessCardObj[logic.getCardCode(color, i)]) {
        colorCards.push(logic.getCardCode(color, i));
        if (colorCards.length >= max) break;
      }
    }
    return colorCards;
  },

  /**
   * 尝试将四条/三条与同花或顺子组合拆分
   * type=4: 四带一 + 同花/顺子
   * type=3: 三带二 + 同花/顺子
   * 选择评分最高的组合方案
   */
  packFourOrThreeWithColorOrRun(type: number, cards: number[], sameNum: number[], colors: number[]): PackedCard[] {
    if (cards.length < 5 + type || sameNum.length == 0) return [];
    const cardType = type == 3 ? logic.cardType.三张 : logic.cardType.四带一;
    let maxScore = -100;
    let maxScorePack: PackedCard[] = [];

    for (let sameIndex = 0; sameIndex < sameNum.length; sameIndex++) {
      const tempCard = cards.concat();
      const threeOrForeCard = logic.outLogicCard(tempCard, sameNum[sameIndex]!, type);

      // 尝试与同花组合
      if (colors.length >= 1) {
        const tempColorArray = logic.getColorArray(tempCard);
        for (let i = 0; i < tempColorArray.length; i++) {
          if (tempColorArray[i]! < 5) continue;
          const checkPairCard = tempCard.concat();
          let hasPair = false;
          if (type == 3) {
            for (let idx = 0; idx < checkPairCard.length - 1; idx++) {
              if (logic.getLogicValue(checkPairCard[0]!) == logic.cardLogic.L2) break;
              if (logic.getLogicValue(checkPairCard[idx]!) == logic.getLogicValue(checkPairCard[idx + 1]!) &&
                  logic.getColorValue(checkPairCard[idx]!) != i && logic.getColorValue(checkPairCard[idx + 1]!) != i) {
                hasPair = true;
              }
            }
            if (!hasPair && tempColorArray[i]! > 5) {
              for (let idx = 0; idx < checkPairCard.length - 1; idx++) {
                if (logic.getLogicValue(checkPairCard[idx]!) == logic.getLogicValue(checkPairCard[idx + 1]!)) {
                  hasPair = true;
                  logic.outCodeCard(checkPairCard, checkPairCard[idx]!);
                  logic.outCodeCard(checkPairCard, checkPairCard[idx]!);
                  break;
                }
              }
            }
          }
          const colorResult = logic.outColorCard(checkPairCard, i, 5);
          let score = logic.getColorOrRunScore(tempCard, colorResult);
          const PackCard: PackedCard[] = [];
          PackCard[0] = { type: cardType, card: threeOrForeCard };
          PackCard[1] = { type: logic.cardType.同花, card: colorResult };
          if (type == 3 && (hasPair || logic.outLookingCard(checkPairCard, null, null, null, [0, 0], true, false).length > 0)) {
            score += 100;
          }
          if (score > maxScore) {
            maxScorePack = PackCard;
          }
        }
      }

      // 尝试与顺子组合
      const run = logic.checkRun(tempCard);
      if (run.length == 5) {
        let score = logic.getColorOrRunScore(tempCard, run);
        const checkPairCard = tempCard.concat();
        const PackCard: PackedCard[] = [];
        PackCard[0] = { type: cardType, card: threeOrForeCard };
        PackCard[1] = { type: logic.cardType.顺子, card: [] };
        for (let i = 0; i < run.length; i++) {
          PackCard[1]!.card.push(...logic.outLogicCard(checkPairCard, run[i]!));
        }
        if (type == 3 && logic.outLookingCard(checkPairCard, null, null, null, [0, 0], true, false).length > 0) {
          score += 100;
        }
        if (score > maxScore) {
          maxScorePack = PackCard;
        }
      }
    }

    if (maxScorePack.length != 0) {
      for (let i = 0; i < maxScorePack.length; i++) {
        logic.outCodeCardArray(cards, maxScorePack[i]!.card);
      }
    }
    return maxScorePack;
  },
};

// ═══════════════════════════════════════════════════════════════════
//  Partner 模块 — 敌友判断系统
//  根据黑桃A/黑桃3的归属判定4个座位的敌友关系
//  friend=1(友方), enemy=-1(敌方), unknow=0(未知)
//  独食(doubleNT): 同时拥有♠A和♠3的玩家，1v3模式
// ═══════════════════════════════════════════════════════════════════

const partner = {
  friend: 1,
  enemy: -1,
  unknow: 0,

  /**
   * 分析所有玩家的敌友关系
   * 核心逻辑:
   * 1. 独食模式: 独食者为敌，其他所有人为友
   * 2. 已知♠A/♠3持有者: 与自己同阵营为友，否则为敌
   * 3. 未知时: 根据出牌历史和剩余牌数推断
   */
  getPartnetInfo(allCard: number[][], selfStation: number, lastStation: number, iDoubleNTPeople: number, i3UpGradePeople: number, iAUpGradePeople: number, lastOutCard: number[], tableUserID: number[], A3IsOut: boolean): PartnerResult {
    const result: PartnerResult = {
      partnerInfo: [0, 0, 0, 0],
      early: true, lastIsFriend: false, lastIsEnemy: false,
      lastEnemyCard: null, lastEnemyStation: 255,
      nextIsEnemy: false, nextIsFriend: false, nextIsDoubltNT: false,
      nextCard: [], lessPlayer: 0, myFriendIsRealMan: false,
      doubleNTIsEnemy: false, selfIsDoubleA3: false,
    };

    let lastIsNM = false;
    if (iDoubleNTPeople == undefined) return result;
    const selfCard = allCard[selfStation]!;
    result.doubleNTIsEnemy = iDoubleNTPeople != 255 && iDoubleNTPeople != selfStation;

    // 判断队友是否是真人（tableUserID > 10000 表示真人）
    if (iDoubleNTPeople != 255) {
      if (tableUserID[iDoubleNTPeople]! > 10000) result.myFriendIsRealMan = true;
    } else {
      let tempA = iAUpGradePeople;
      let temp3 = i3UpGradePeople;
      for (let station = 0; station < allCard.length; station++) {
        for (let i = 0; i < allCard[station]!.length; i++) {
          if (allCard[station]![i] == logic.cardCode.黑桃A) tempA = station;
          if (allCard[station]![i] == logic.cardCode.黑桃3) temp3 = station;
        }
      }
      for (let station = 0; station < 4; station++) {
        if (station != selfStation && tableUserID[station]! > 10000) {
          result.myFriendIsRealMan = (tempA == station || temp3 == station) == (tempA == selfStation || temp3 == selfStation);
          break;
        }
      }
    }

    // 判断是否游戏初期（所有人手牌都 > 2张）
    for (let i = 0; i < allCard.length; i++) {
      if (allCard[i]!.length <= 2) result.early = false;
    }

    if (iDoubleNTPeople == 255) {
      // 非独食模式: 根据♠A/♠3出牌情况判断阵营
      if (!result.early || i3UpGradePeople != 255 || iAUpGradePeople != 255 || A3IsOut) {
        result.early = false;
        for (let station = 0; station < allCard.length; station++) {
          for (let i = 0; i < allCard[station]!.length; i++) {
            if (allCard[station]![i] == logic.cardCode.黑桃A) iAUpGradePeople = station;
            if (allCard[station]![i] == logic.cardCode.黑桃3) i3UpGradePeople = station;
          }
        }
      } else {
        // 游戏初期，检查上一轮是否出了A（可能暴露阵营）
        for (let i = 0; i < lastOutCard.length; i++) {
          if (logic.getLogicValue(lastOutCard[i]!) == logic.cardLogic.LA) { lastIsNM = true; break; }
        }
      }
      // 检查自己手中是否持有♠A/♠3
      for (let i = 0; i < selfCard.length; i++) {
        if (selfCard[i] == logic.cardCode.黑桃A) iAUpGradePeople = selfStation;
        if (selfCard[i] == logic.cardCode.黑桃3) i3UpGradePeople = selfStation;
      }
      if (iAUpGradePeople == selfStation && i3UpGradePeople == selfStation) result.selfIsDoubleA3 = true;
    } else {
      // 独食模式: 重置♠A/♠3归属，判断下家是否是独食者
      iAUpGradePeople = 255;
      i3UpGradePeople = 255;
      const nextIdx = (selfStation + 1) % 4;
      if (nextIdx == iDoubleNTPeople) result.nextIsDoubltNT = true;
    }

    const selfIsA3 = iDoubleNTPeople == selfStation || i3UpGradePeople == selfStation || iAUpGradePeople == selfStation;
    let lessUserCount = 0;
    let less1User = 0;

    // 根据已知信息设置各座位的敌友关系
    for (let i = 0; i < 4; i++) {
      if (i == selfStation) continue;
      if (allCard[i]!.length != 0) { lessUserCount++; less1User = i; }
      if (lastIsNM && i == lastStation) {
        result.partnerInfo[i] = selfIsA3 ? this.enemy : this.friend;
      }
      if (iDoubleNTPeople == i || i3UpGradePeople == i || iAUpGradePeople == i) {
        result.partnerInfo[i] = selfIsA3 ? this.friend : this.enemy;
        continue;
      }
      if ((iDoubleNTPeople != 255 || (i3UpGradePeople != 255 && iAUpGradePeople != 255)) && result.partnerInfo[i] == this.unknow) {
        result.partnerInfo[i] = selfIsA3 ? this.enemy : this.friend;
      }
    }

    // 只剩一个对手时，必为敌人
    if (lessUserCount == 1) result.partnerInfo[less1User] = this.enemy;

    for (let i = 0; i < 4; i++) {
      if (allCard[i]!.length > 0) result.lessPlayer++;
      // 只剩1张牌的未知玩家视为敌人（威胁大）
      if (allCard[i]!.length == 1 && result.partnerInfo[i] == this.unknow) {
        result.partnerInfo[i] = selfIsA3 ? this.enemy : this.friend;
      }
    }

    result.lastIsFriend = lastStation != selfStation && result.partnerInfo[lastStation] == this.friend;
    result.lastIsEnemy = lastOutCard.length > 0 && lastStation != selfStation && result.partnerInfo[lastStation] == this.enemy;

    // 找到下一个有牌的玩家，判断其敌友
    let maybeFriendCount = 0;
    for (let i = 1; i < 4; i++) {
      const chairID = (selfStation + i) % 4;
      if (allCard[chairID]!.length > 0 && result.partnerInfo[chairID] != this.enemy) maybeFriendCount++;
    }
    if (maybeFriendCount >= 1) {
      for (let i = 1; i < 4; i++) {
        const chairID = (selfStation + i) % 4;
        if (allCard[chairID]!.length > 0) {
          result.nextIsEnemy = result.partnerInfo[chairID] == this.enemy;
          result.nextIsFriend = result.partnerInfo[chairID] == this.friend;
          result.nextCard = allCard[chairID]!.concat();
          break;
        }
      }
    }

    // 确定"最后一个敌人"：仅当只有一个敌人有牌时才设置
    for (let i = 0; i < 4; i++) {
      if (result.partnerInfo[i] == this.unknow && allCard[i]!.length != 0 && i != selfStation) {
        result.lastEnemyStation = 255;
        result.lastEnemyCard = null;
        break;
      }
      if (result.partnerInfo[i] == this.enemy && result.lastEnemyCard == null && allCard[i]!.length != 0) {
        result.lastEnemyStation = i;
        result.lastEnemyCard = allCard[i]!;
        continue;
      }
      if (result.partnerInfo[i] == this.enemy && result.lastEnemyCard && allCard[i]!.length != 0) {
        result.lastEnemyStation = 255;
        result.lastEnemyCard = null;
        break;
      }
    }

    return result;
  },
};

// ═══════════════════════════════════════════════════════════════════
//  AI 模块 — 核心出牌决策引擎
//  outCard: 主出牌函数，根据当前局面决定打什么牌
//  doubleNT: 独食模拟（模拟完整对局判断能否独食）
//  oneOnOne: 1v1对决策略（与独食者单挑时的特殊出牌）
// ═══════════════════════════════════════════════════════════════════

const ai = {
  /**
   * 主出牌决策函数
   * 流程:
   * 1. 准备数据: 排序手牌、获取敌友信息、拆分手牌
   * 2. 特殊处理: 三条归入散牌、敌人即将走完时拆牌等
   * 3. 判断敌人/友方即将走完的情况
   * 4. 独食模式特殊处理
   * 5. 按牌型分支决策(自由出牌/单张/对子/三张/五张)
   */
  outCard(data: RefData): number[] {
    data.lastOutCard.card = data.lastOutCard.card == null ? [] : data.lastOutCard.card;
    // 按逻辑值*16+花色排序，确保同点数牌按花色从小到大
    data.tableCardData[data.selfStation]!.sort((a, b) => {
      return (logic.getLogicValue(a) * 0x10 + logic.getColorValue(a) -
        (logic.getLogicValue(b) * 0x10 + logic.getColorValue(b)));
    });

    const selfStation = data.selfStation;
    const allCard: number[][] = [[], [], [], []];
    for (let i = 0; i < 4; i++) {
      if (!data.tableCardData[i]) continue;
      for (let ii = 0; ii < data.tableCardData[i]!.length; ii++) {
        if (data.tableCardData[i]![ii] != 0) allCard[i]!.push(data.tableCardData[i]![ii]!);
      }
    }

    const selfCard = allCard[selfStation]!;
    const lastOutCard = data.lastOutCard.card!;
    const lastStation = lastOutCard.length == 0 ? selfStation : data.lastOutCard.station;
    // eslint-disable-next-line eqeqeq
    const cardType = (lastOutCard as any) == 0 ? logic.cardType.自己出牌 : data.lastOutCard.cardType;
    const tableUserID = data.tableUserID;

    // ─── 获取敌友关系信息 ───
    const partnerInfo = partner.getPartnetInfo(
      allCard, selfStation, lastStation,
      data.iDoubleNTPeople, data.i3UpGradePeople, data.iAUpGradePeople,
      lastOutCard, tableUserID, data.A3IsOut
    );

    // ─── 拆分手牌为 pack[0]=散牌 + pack[1..n]=组合 ───
    let packCard = pack.packCard(selfCard);

    // 特殊情况: 只有散牌+三条，且总共3张，把三条归入散牌
    if (packCard.length == 2 && packCard[1]!.type == logic.cardType.三张 && selfCard.length == 3) {
      packCard.splice(1, 1);
      packCard[0]!.card.push(...selfCard);
    }

    // 散牌为空时，把大三条(≥J)归入散牌（避免浪费大牌组三条）
    if (packCard[0]!.card.length == 0 && cardType != logic.cardType.自己出牌) {
      for (let i = 1; i < packCard.length; i++) {
        const ele = packCard[i]!;
        if (ele.type != logic.cardType.三张) continue;
        if (logic.getLogicValue(ele.card[0]!) >= logic.cardLogic.LJ) {
          packCard[0]!.card.push(...ele.card);
          packCard.splice(i, 1);
          i--;
          continue;
        }
      }
    }

    // ─── 判断是否所有其他人都是敌人 ───
    let allEnemy = true;
    for (let i = 0; i < partnerInfo.partnerInfo.length; i++) {
      if (i == selfStation) continue;
      if (allCard[i]!.length == 0) continue;
      if (partnerInfo.partnerInfo[i] != partner.enemy) { allEnemy = false; break; }
    }

    // 全是敌人且对手出了小牌(1-2张)时，尝试从组合中拆牌来压
    let pack0LogicArray = logic.getLogicArray(packCard[0]!.card);
    if (allEnemy && lastOutCard.length > 0 && lastOutCard.length <= 2 &&
      logic.getLogicValue(lastOutCard[0]!) < logic.cardLogic.LK) {
      let hasBigger = false;
      for (let i = pack0LogicArray.length - 1; i >= 0; i--) {
        if (pack0LogicArray[i]! >= lastOutCard.length) { hasBigger = true; break; }
      }
      if (!hasBigger) {
        if (lastOutCard.length == 1) {
          // 找最大牌的组合拆开
          let maxIndex = 0;
          let maxLogic = 0;
          for (let i = 1; i < packCard.length; i++) {
            let tempMaxLogic = 0;
            for (let ii = 0; ii < packCard[i]!.card.length; ii++) {
              if (logic.getLogicValue(packCard[i]!.card[ii]!) > tempMaxLogic) {
                tempMaxLogic = logic.getLogicValue(packCard[i]!.card[ii]!);
              }
            }
            if (tempMaxLogic > maxLogic) { maxLogic = tempMaxLogic; maxIndex = i; }
          }
          if (maxIndex != 0) {
            packCard[0]!.card.push(...packCard[maxIndex]!.card);
            packCard.splice(maxIndex, 1);
          }
        }
      }
    }

    // ─── 判断敌人/友方是否即将走完（只剩1-2张） ───
    let enemyWillGo = false;
    let friendWillGo = false;
    let enemyWillGoLogic = 0;
    let firendWillGoLogic = 0;

    if ((cardType == logic.cardType.单张 && selfCard.length != 1) ||
      (cardType == logic.cardType.对子 && selfCard.length != 2)) {
      for (let i = 0; i < partnerInfo.partnerInfo.length; i++) {
        if (i == selfStation) continue;
        if (i == lastStation) continue;
        if (allCard[i]!.length == lastOutCard.length) {
          if (logic.compareCard(allCard[i]!, lastOutCard)) {
            if (partnerInfo.partnerInfo[i] == partner.enemy) {
              enemyWillGo = true;
              enemyWillGoLogic = logic.getLogicValue(allCard[i]![0]!);
              if (partnerInfo.doubleNTIsEnemy) break;
            }
            if (partnerInfo.partnerInfo[i] == partner.friend) {
              friendWillGo = true;
              firendWillGoLogic = logic.getLogicValue(allCard[i]![0]!);
            }
          }
        }
      }
      // 友方即将走完且敌人不会走: 出小牌让友方走，或直接放行
      if (!enemyWillGo && friendWillGo) {
        const maxLogic = firendWillGoLogic > logic.cardLogic.LJ ? logic.cardLogic.LJ : firendWillGoLogic;
        const outCard = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, null, maxLogic, lastOutCard, false);
        if (outCard.length > 0) return outCard;
        return [];
      }
    }

    // ─── 判断自己是否能一手走完 ───
    let selfWillGo = false;
    if (lastOutCard.length <= 2) {
      if (logic.outLookingCard(selfCard.concat(), null, null, null, lastOutCard, false, false).length == selfCard.length) {
        selfWillGo = true;
      }
    }
    // 【改进】5张手牌且能组成5张组合压过对手时，也算能一手走完
    if (!selfWillGo && selfCard.length == 5 && lastOutCard.length == 5 && cardType >= logic.cardType.顺子) {
      const tmpSelfPack = pack.packCard(selfCard.concat());
      for (let i = 1; i < tmpSelfPack.length; i++) {
        if (pack.comparePack(tmpSelfPack[i]!, { type: cardType, card: lastOutCard }) > 0) {
          selfWillGo = true;
          break;
        }
      }
    }

    // ─── 独食模式特殊处理 ───
    if (partnerInfo.doubleNTIsEnemy && cardType) {
      if (selfWillGo) return selfCard;
      if (data.oneOnOneStation && data.oneOnOneStation != selfStation) return [];
      // 尝试1v1对决策略
      const oneOnOneCard = this.oneOnOne(allCard[selfStation]!, allCard[data.iDoubleNTPeople]!, lastOutCard, cardType);
      if (oneOnOneCard.length > 0) return oneOnOneCard;
    }

    // 独食模式下散牌无法压过时，考虑全部拆开
    if (partnerInfo.doubleNTIsEnemy && lastOutCard.length <= 2 &&
      logic.outLookingCard(packCard[0]!.card.concat(), pack0LogicArray, null, null, lastOutCard, true, false).length == 0) {
      let ourFrienCanKickHim = false;
      for (let i = 1; i < 4; i++) {
        const station = (selfStation + i) % 4;
        if (station == data.iDoubleNTPeople) break;
        if (logic.outLookingCard(allCard[station]!.concat(), logic.getLogicArray(allCard[station]!), null, null, lastOutCard, true, false).length > 0) {
          ourFrienCanKickHim = true; break;
        }
      }
      let emenyWillGoPair = false;
      if (lastOutCard.length == 2) {
        const enemyCard = allCard[data.iDoubleNTPeople]!.concat();
        const enemyLogicArray = logic.getLogicArray(enemyCard);
        const enemyOutCard = logic.outLookingCard(enemyCard, enemyLogicArray, null, logic.cardLogic.L2, lastOutCard, true, false);
        emenyWillGoPair = enemyOutCard.length > 0;
      }
      // 多种紧急情况下，把所有组合拆回散牌全力压制
      if (enemyWillGo ||
        (lastOutCard.length == 1 && logic.getLogicValue(lastOutCard[0]!) < logic.cardLogic.LK) ||
        emenyWillGoPair ||
        (partnerInfo.lastIsEnemy && !ourFrienCanKickHim)) {
        while (packCard.length > 1) {
          packCard[0]!.card.push(...packCard[1]!.card);
          packCard.splice(1, 1);
        }
      }
    }

    // 最后一个敌人只剩1-2张时，考虑把组合拆开来压
    pack0LogicArray = logic.getLogicArray(packCard[0]!.card);
    if (partnerInfo.lastIsEnemy && lastStation == partnerInfo.lastEnemyStation &&
      partnerInfo.lastEnemyCard && partnerInfo.lastEnemyCard.length <= 2 &&
      (cardType == logic.cardType.单张 || cardType == logic.cardType.对子)) {
      let ourFrienCanKickHim = false;
      for (let i = 1; i < 4; i++) {
        const station = (selfStation + i) % 4;
        if (station == data.iDoubleNTPeople) break;
        if (logic.outLookingCard(allCard[station]!.concat(), logic.getLogicArray(allCard[station]!), null, null, lastOutCard, true, false).length > 0) {
          ourFrienCanKickHim = true; break;
        }
      }
      const tempLogic0Card = packCard[0]!.card.concat();
      const tryOutCard = logic.outLookingCard(tempLogic0Card, pack0LogicArray, null, null, lastOutCard, true);
      if (tryOutCard.length == 0 && !ourFrienCanKickHim) {
        // 散牌压不过且友方也帮不了，全部拆开
        packCard = [];
        packCard[0] = { type: logic.cardType.单张, card: selfCard };
        pack0LogicArray = logic.getLogicArray(packCard[0]!.card);
      }
    }

    // ─── 统计大小牌数量（用于决定是否激进出牌） ───
    let smallCard = 0;
    let bigCard = 0;
    let smallPairCount = 0;
    for (let i = 0; i < pack0LogicArray.length; i++) {
      if (pack0LogicArray[i] == 0) continue;
      if (i < logic.cardLogic.LJ) {
        // 小于J: 单张算小牌，对子单独计
        if (pack0LogicArray[i] == 1) smallCard++;
        else smallPairCount++;
      } else if (i >= logic.cardLogic.LJ && i < logic.cardLogic.LA) {
        // J~K: 对子算大牌，单张算小牌
        if (pack0LogicArray[i]! >= 2) bigCard++;
        else smallCard++;
      } else if (i >= logic.cardLogic.LA) {
        // A以上: 全部算大牌
        bigCard += pack0LogicArray[i]!;
      }
    }
    smallCard = smallCard + smallPairCount;

    // ─── 上家已走完时的特殊处理 ───
    if ((cardType == logic.cardType.单张 && selfCard.length != 1) ||
      (cardType == logic.cardType.对子 && selfCard.length != 2 && !enemyWillGo)) {
      if (allCard[lastStation]!.length == 0) {
        if (!enemyWillGo) {
          const nextRoundStation = lastStation + 1;
          for (let i = 0; i < 3; i++) {
            const chairID = (nextRoundStation + i) % 4;
            if (allCard[chairID]!.length > 0) {
              if (chairID == selfStation || partnerInfo.partnerInfo[chairID] == partner.friend) {
                if (lastOutCard.length == 1) {
                  const outCard = logic.outLookingCard(packCard[0]!.card, null, null, logic.cardLogic.LQ, lastOutCard, false, false);
                  if (outCard.length >= 0) return outCard;
                }
                return [];
              }
              break;
            }
          }
        }
      }
    }

    const cardObj = logic.getAllCardObject(selfCard);
    const pack0CardObj = logic.getAllCardObject(packCard[0]!.card);

    // 友方出的大牌被敌人压过，但我方散牌压不过且下家是敌人: 放弃跟牌
    if (cardType != logic.cardType.自己出牌 && partnerInfo.lastIsFriend &&
      partnerInfo.lastEnemyCard && partnerInfo.lastEnemyCard.length == 1 &&
      !logic.runOutForSingle(pack0CardObj, pack0LogicArray, partnerInfo.lastEnemyCard[0]!) &&
      partnerInfo.nextIsEnemy && !enemyWillGo) {
      return [];
    }

    // ═══════════════════════════════════════════════════════════════
    //  按牌型分支决策
    // ═══════════════════════════════════════════════════════════════

    switch (cardType) {
      // ─── 自由出牌（没有需要跟的牌） ───
      case logic.cardType.自己出牌: {
        // 持有方片4时优先出方片4（游戏规则第一手必出方片4）
        if (cardObj[logic.cardCode.方片4]) {
          for (let i = 1; i < packCard.length; i++) {
            for (let ii = 0; ii < packCard[i]!.card.length; ii++) {
              if (packCard[i]!.card[ii] != logic.cardCode.方片4) continue;
              return packCard[i]!.card;
            }
          }
          return logic.outLogicCard(packCard[0]!.card, 4, pack0LogicArray[4]);
        }

        // 只剩5张: 检查能否组成三带二或四带一直接走
        if (selfCard.length == 5) {
          if (packCard[0]!.card.length == 5) {
            let singleCardLogic = 0, pairCardLogic = 0, threeCardLogic = 0, fourCardLogic = 0;
            for (let i = 0; i < pack0LogicArray.length; i++) {
              switch (pack0LogicArray[i]) {
                case 1: singleCardLogic = i; break;
                case 2: pairCardLogic = i; break;
                case 3: threeCardLogic = i; break;
                case 4: fourCardLogic = i; break;
              }
            }
            if (threeCardLogic != 0 && pairCardLogic != 0) {
              const outCard = logic.outLogicCard(packCard[0]!.card, threeCardLogic, 3);
              return outCard.concat(logic.outLogicCard(packCard[0]!.card, pairCardLogic, 2));
            }
            if (fourCardLogic != 0 && singleCardLogic != 0) {
              const outCard = logic.outLogicCard(packCard[0]!.card, fourCardLogic, 4);
              return outCard.concat(logic.outLogicCard(packCard[0]!.card, singleCardLogic, 1));
            }
          }
        }

        // 检查是否有友方只剩1张且比自己最小牌小: 出小牌让友方接
        if (data.iDoubleNTPeople != 255 && data.iDoubleNTPeople != selfStation) {
          for (let i = 1; i < 4; i++) {
            const station = (selfStation + i) % 4;
            if (station == data.iDoubleNTPeople) break;
            if (allCard[station]!.length == 1) {
              if (partnerInfo.partnerInfo[station] == partner.friend) {
                if (!logic.compareCard([selfCard[0]!], allCard[station]!)) return [selfCard[0]!];
              }
            }
          }
        }

        // ─── 【改进】队友送牌: 出队友能接且全场无人能压过的牌 ───
        // 场景: 队友只剩1-2张且是全场最大牌 → 出比它小的牌让队友接住直接走完
        if (selfCard.length > 2 && !partnerInfo.doubleNTIsEnemy) {
          for (let i = 1; i < 4; i++) {
            const tmStation = (selfStation + i) % 4;
            if (allCard[tmStation]!.length == 0) continue;
            if (partnerInfo.partnerInfo[tmStation] != partner.friend) continue;
            if (allCard[tmStation]!.length > 2) break; // 队友牌多就不需要送
            const tmCards = allCard[tmStation]!;
            // 排除自己和队友，检查队友的牌是否全场最大
            const checkAllCard = allCard.concat();
            checkAllCard[selfStation] = [];
            checkAllCard[tmStation] = [];
            // 队友剩1张单牌: 出比它小的单牌
            if (tmCards.length == 1 && logic.checkCardIsBigger(tmCards, checkAllCard)) {
              const tmLogic = logic.getLogicValue(tmCards[0]!);
              for (let j = 4; j < tmLogic; j++) {
                if (pack0LogicArray[j]! >= 1) return logic.outLogicCard(packCard[0]!.card, j, 1);
              }
            }
            // 队友剩1对: 出比它小的对子
            if (tmCards.length == 2 && logic.getLogicValue(tmCards[0]!) == logic.getLogicValue(tmCards[1]!)) {
              if (logic.checkCardIsBigger(tmCards, checkAllCard)) {
                const tmLogic = logic.getLogicValue(tmCards[0]!);
                for (let j = 4; j < tmLogic; j++) {
                  if (pack0LogicArray[j]! >= 2) return logic.outLogicCard(packCard[0]!.card, j, 2);
                }
              }
            }
            break; // 只看第一个有牌的友方
          }
        }

        // 所有敌人手牌都 < 5张: 优先出5张组合（不给敌人出5张组合的机会）
        let enemyHaveNo5Card = true;
        for (let i = 0; i < 4; i++) {
          if (i == selfStation) continue;
          if (allCard[i]!.length < 5) continue;
          if (partnerInfo.partnerInfo[i] != partner.enemy) continue;
          enemyHaveNo5Card = false; break;
        }
        if (enemyHaveNo5Card && packCard.length > 1) return packCard[1]!.card;
        // 注意: 当敌人有5张牌时，即使我方组合无敌也不主动出
        // 无敌组合留着防守更有价值: 敌人出5张时压他→夺回控制权→出散牌

        // 计算散牌手数
        const handCardInfo = logic.getHandCardCount(packCard, pack0LogicArray, 0, allCard, selfStation);

        // 只剩2手以内: 精细策略
        if (handCardInfo.handCount <= 2) {
          if (handCardInfo.handCards.length == 2) {
            let singleCards: number[] = [];
            let pairCards: number[] = [];
            handCardInfo.handCards.forEach((ele) => {
              if (ele.length == 1) singleCards = ele;
              else if (ele.length == 2) pairCards = ele;
            });
            if (singleCards.length != 0 && pairCards.length != 0) {
              const singleLogic = logic.getLogicValue(singleCards[0]!);
              const pairLogic = logic.getLogicValue(pairCards[0]!);
              const tmpAllCard = allCard.concat();
              tmpAllCard[selfStation] = [];
              if (partnerInfo.lastEnemyCard && partnerInfo.lastEnemyCard.length == 1) return pairCards;
              if (logic.checkCardIsBigger(singleCards, tmpAllCard)) return singleCards;
              if (logic.checkCardIsBigger(pairCards, tmpAllCard)) return pairCards;
              if (singleLogic < logic.cardLogic.L10 && pairLogic > logic.cardLogic.L10) return singleCards;
              if (singleLogic < logic.cardLogic.L10 && pairLogic <= logic.cardLogic.L10) return pairCards;
              if (singleLogic >= logic.cardLogic.L10) return pairCards;
            }
            // 【改进】两手散牌都是同类型(如两个单张): 有一手无敌时先出弱牌试探
            // 例: 单张3(无敌) + 单张7 → 先出7试探，保留3做终结/保险
            if (handCardInfo.handCards.length == 2) {
              const weakHand = handCardInfo.handCards[0]!;
              const strongHand = handCardInfo.handCards[1]!;
              const tmpAllCard2 = allCard.concat();
              tmpAllCard2[selfStation] = [];
              if (logic.checkCardIsBigger(strongHand, tmpAllCard2)) {
                return weakHand; // 先出弱牌，保留无敌牌
              }
            }
          }
          // 散牌1手 + 5张组合1手 = 2手: 有组合无敌时先出散牌
          if (handCardInfo.handCards.length == 1 && packCard.length == 2) {
            const tmpAllCard3 = allCard.concat();
            tmpAllCard3[selfStation] = [];
            const comboUnbeatable = (() => {
              for (let ei = 0; ei < 4; ei++) {
                if (ei == selfStation || allCard[ei]!.length < 5) continue;
                if (partnerInfo.partnerInfo[ei] == partner.friend) continue;
                const ePack = pack.packCard(allCard[ei]!.concat());
                for (let ek = 1; ek < ePack.length; ek++) {
                  if (pack.comparePack(ePack[ek]!, packCard[1]!) > 0) return false;
                }
              }
              return true;
            })();
            if (comboUnbeatable && handCardInfo.handCards[0]!.length > 0) {
              // 组合无敌 → 先出散牌试探，组合留着防守+终结
              return handCardInfo.handCards[0]!;
            }
          }
          if (handCardInfo.bestCardIsBigger && !partnerInfo.nextIsFriend) return handCardInfo.mybestCard;
        }

        // 最后一个敌人只剩1-2张: 针对性出牌
        if (partnerInfo.lastEnemyCard && partnerInfo.lastEnemyCard.length <= 2) {
          if (packCard.length > 1) return packCard[1]!.card;
          if (handCardInfo.handCount == 1) return handCardInfo.handCards[0]!;
          if (partnerInfo.lastEnemyCard.length == 1) {
            // 敌人剩1张: 优先出对子，再出能压过的单张
            for (let i = 0; i < pack0LogicArray.length; i++) {
              if (pack0LogicArray[i] == 2) return logic.outLogicCard(packCard[0]!.card, i, 2);
            }
            const minLogic2 = logic.getLogicValue(partnerInfo.lastEnemyCard[0]!) < logic.cardLogic.LJ
              ? logic.cardLogic.LJ : logic.getLogicValue(partnerInfo.lastEnemyCard[0]!);
            const outCard = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, minLogic2, null, partnerInfo.lastEnemyCard, false, false);
            if (outCard.length != 0) return outCard;
            for (let i = pack0LogicArray.length - 1; i >= 0; i--) {
              if (pack0LogicArray[i] != 0) {
                const cardNum = pack0LogicArray[i]! >= 4 ? 3 : pack0LogicArray[i]!;
                return logic.outLogicCard(packCard[0]!.card, i, cardNum);
              }
            }
          }
          if (partnerInfo.lastEnemyCard.length == 2) {
            // 敌人剩2张(对子): 出单张拆掉敌人对子
            const enemyPairLogic = logic.getLogicValue(partnerInfo.lastEnemyCard[0]!);
            // 优先出比敌人对子大的单张 → 敌人压不住，我们保持控制权
            for (let i = enemyPairLogic + 1; i < pack0LogicArray.length; i++) {
              if (pack0LogicArray[i] == 1) return logic.outLogicCard(packCard[0]!.card, i, 1);
            }
            // 没有更大的单张，出最小单张骚扰(迫使敌人拆对子)
            let bigger = 0;
            for (let i = pack0LogicArray.length - 1; i >= 0; i--) {
              if (pack0LogicArray[i] != 0) { bigger = i; break; }
            }
            for (let i = 0; i < bigger; i++) {
              if (pack0LogicArray[i] == 1) return logic.outLogicCard(packCard[0]!.card, i, 1);
            }
          }
        }

        // 下家是敌人且只剩1-2张: 针对性出牌
        if (partnerInfo.nextIsEnemy && partnerInfo.nextCard.length <= 2) {
          if (packCard.length > 1) return packCard[1]!.card;
          if (handCardInfo.handCount == 1) return handCardInfo.handCards[0]!;
          if (partnerInfo.nextCard.length == 1) {
            let outLogic2 = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, null, logic.cardLogic.LA, [0, 0], false);
            if (outLogic2.length > 0) return outLogic2;
            outLogic2 = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, null, logic.cardLogic.L2, partnerInfo.nextCard, true);
            if (outLogic2.length > 0) return outLogic2;
          }
          if (partnerInfo.nextCard.length == 2) {
            let outLogic2 = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, null, logic.cardLogic.L2, [0], false);
            if (outLogic2.length > 0) return outLogic2;
            outLogic2 = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, null, logic.cardLogic.LA, [0], true);
            if (outLogic2.length > 0) return outLogic2;
          }
        }

        // 检查是否有敌人只剩1张且有友方也只剩1张: 出中间值牌
        let nextEnemyLessSingleCard = 0;
        for (let i = 1; i < 4; i++) {
          const station = (selfStation + i) % 4;
          if (allCard[station]!.length == 1) {
            if (partnerInfo.partnerInfo[station] == partner.enemy) {
              nextEnemyLessSingleCard = allCard[station]![0]!; continue;
            }
            if (partnerInfo.partnerInfo[station] == partner.friend) {
              let outCard = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, nextEnemyLessSingleCard, logic.getLogicValue(allCard[station]![0]!), [logic.cardCode.方片4], false);
              if (outCard.length > 0) return outCard;
              outCard = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, nextEnemyLessSingleCard, logic.getLogicValue(allCard[station]![0]!), [logic.cardCode.方片4], true);
              if (outCard.length > 0) return outCard;
            }
          }
        }

        // 3手牌: 尝试从最大牌往下找合适的出牌
        if (handCardInfo.handCount == 3) {
          if (handCardInfo.bestCardIsBigger) {
            for (let i = logic.getLogicValue(handCardInfo.mybestCard[0]!) - 1; i >= 0; i--) {
              if (pack0LogicArray[i] == handCardInfo.mybestCard.length) {
                return logic.outLogicCard(packCard[0]!.card, i, pack0LogicArray[i]!);
              }
            }
          }
          if (handCardInfo.handCards.length >= 2) {
            const nextBiggerCard = handCardInfo.handCards[handCardInfo.handCards.length - 2]!;
            const tmpAllCard = allCard.concat();
            tmpAllCard[selfStation] = [];
            if (logic.checkCardIsBigger(nextBiggerCard, tmpAllCard)) {
              for (let i = logic.getLogicValue(nextBiggerCard[0]!) - 1; i >= 0; i--) {
                if (pack0LogicArray[i] == nextBiggerCard.length) {
                  return logic.outLogicCard(packCard[0]!.card, i, pack0LogicArray[i]!);
                }
              }
            }
          }
        }

        // 独食模式的自由出牌策略
        if (partnerInfo.doubleNTIsEnemy && packCard.length >= 2) {
          const doubleNTPack = pack.packCard(allCard[data.iDoubleNTPeople]!);
          if (doubleNTPack.length == 1) return packCard[1]!.card;
          if (packCard[0]!.card.length == 0) return packCard[1]!.card;
          if (handCardInfo.handCount <= 2 && (allCard[data.iDoubleNTPeople]! as any) >= 7) return packCard[1]!.card;
          if (packCard.length >= 3) {
            if (pack.comparePack(packCard[packCard.length - 1]!, doubleNTPack[doubleNTPack.length - 1]!) > 0 && (allCard[data.iDoubleNTPeople]! as any) >= 7) return packCard[1]!.card;
            if (pack.comparePack(packCard[1]!, doubleNTPack[doubleNTPack.length - 1]!) > 0) return packCard[1]!.card;
          }
          if (packCard.length == 2) {
            if (pack.comparePack(packCard[1]!, doubleNTPack[doubleNTPack.length - 1]!) > 0) {
              if (handCardInfo.handCount <= 2) return packCard[1]!.card;
            }
          }
        } else {
          // 非独食: 有5张组合就出
          if (packCard.length > 1 && (packCard[0]!.card.length <= 4 || partnerInfo.nextIsDoubltNT)) {
            if (packCard[1]!.type <= logic.cardType.同花) return packCard[1]!.card;
          }
          if (packCard.length > 1 && (packCard[0]!.card.length <= 2 || partnerInfo.nextIsDoubltNT)) {
            return packCard[1]!.card;
          }
        }

        // 下家是独食者: 出能压制的牌
        if (partnerInfo.nextIsDoubltNT) {
          const bestOutNum = allCard[data.iDoubleNTPeople]!.length == 1 ? 2 : 1;
          for (let i = 0; i < logic.cardLogic.LK; i++) {
            if (bestOutNum == 1 && i < logic.cardLogic.LJ) continue;
            if (pack0LogicArray[i] == bestOutNum) return logic.outLogicCard(packCard[0]!.card, i, bestOutNum);
          }
        }

        // 辅助函数: 按范围找单张或对子出
        const outSingleOrPair = (minLogic: number, maxLogic: number, count: number): number[] => {
          for (let i = minLogic; i < maxLogic; i++) {
            if (pack0LogicArray[i]! >= count) {
              const cardNum = pack0LogicArray[i]! >= 4 ? 3 : pack0LogicArray[i]!;
              return logic.outLogicCard(packCard[0]!.card, i, cardNum);
            }
          }
          return [];
        };

        if (partnerInfo.nextIsDoubltNT) {
          let outCard = outSingleOrPair(logic.cardLogic.L4, logic.cardLogic.LA, 2);
          if (outCard.length != 0) return outCard;
          outCard = outSingleOrPair(logic.cardLogic.LK, logic.cardLogic.L2, 1);
          if (outCard.length != 0) return outCard;
          for (let i = logic.cardLogic.LK; i >= 0; i--) {
            if (pack0LogicArray[i]! > 0) {
              const cardNum = pack0LogicArray[i]! >= 4 ? 3 : pack0LogicArray[i]!;
              return logic.outLogicCard(packCard[0]!.card, i, cardNum);
            }
          }
        }

        // 有敌人只剩1张: 先出对子再出单张
        if (nextEnemyLessSingleCard != 0) {
          if (packCard.length > 1) return packCard[1]!.card;
          let outCard = outSingleOrPair(logic.cardLogic.L4, logic.cardLogic.LJ, 2);
          if (outCard.length != 0) return outCard;
          let haveBigSingle = false;
          let BigPairCount = 0;
          for (let i = logic.cardLogic.L10; i < pack0LogicArray.length; i++) {
            if (pack0LogicArray[i] == 1) { haveBigSingle = true; break; }
            if (pack0LogicArray[i] == 2) BigPairCount++;
          }
          let OutBigPair = true;
          if (!haveBigSingle) OutBigPair = BigPairCount <= 0;
          if (OutBigPair) {
            outCard = outSingleOrPair(logic.cardLogic.L4, logic.cardLogic.LA, 2);
            if (outCard.length != 0) return outCard;
          }
          outCard = outSingleOrPair(nextEnemyLessSingleCard, logic.cardLogic.LA, 1);
          if (outCard.length != 0) return outCard;
        }
        // 默认自由出牌: 优先出小对子(对子更难被压，更容易保持控制权)
        // 然后再出单张
        for (let spi = 0; spi < logic.cardLogic.LJ; spi++) {
          if (pack0LogicArray[spi] == 2) return logic.outLogicCard(packCard[0]!.card, spi, 2);
        }
        return outSingleOrPair(0, pack0LogicArray.length, 1);
      }

      // ─── 跟单张 ───
      case logic.cardType.单张: {
        // 友方出的且友方只剩1张: 出小牌让友方走，或放行
        if (partnerInfo.lastIsFriend && allCard[lastStation]!.length == 1 && selfCard.length != 1 && !enemyWillGo) {
          const outCard = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, null, logic.cardLogic.LK, lastOutCard, false, false);
          if (outCard.length > 0) return outCard;
          return [];
        }

        // 最后一个敌人只剩1-2张: 从大到小找能压的牌
        if (partnerInfo.lastEnemyCard && partnerInfo.lastEnemyCard.length <= 2) {
          let bigger = 0;
          for (let i = pack0LogicArray.length - 1; i >= 0; i--) {
            if (pack0LogicArray[i] == 1) { bigger = i; break; }
            if (pack0LogicArray[i] == 2) { bigger = i + 1; break; }
          }
          bigger = partnerInfo.lastEnemyCard.length == 2 ? bigger : pack0LogicArray.length;
          bigger = (partnerInfo.lastIsFriend && allCard[lastStation]!.length > 0 && logic.getLogicValue(lastOutCard[0]!) >= logic.cardLogic.LA) ? 0 : bigger;
          for (let i = bigger - 1; i >= logic.getLogicValue(lastOutCard[0]!); i--) {
            if (pack0LogicArray[i] == 0) continue;
            if (i == logic.getLogicValue(lastOutCard[0]!)) {
              for (let color = logic.getColorValue(lastOutCard[0]!) + 1; color < 4; color++) {
                const nextColorCard = 0x10 * color + (lastOutCard[0]! % 0x10);
                if (pack0CardObj[nextColorCard]) return [nextColorCard];
              }
              continue;
            }
            return logic.outLogicCard(packCard[0]!.card, i, 1);
          }
        }

        // 计算只用散牌还需几手走完，判断是否值得出大牌
        const handCards: number[][] = [];
        let mybestCardSingle: number[] = [];
        const tmpPack0 = packCard[0]!.card.concat();
        for (let i = 0; i < pack0LogicArray.length; i++) {
          if (pack0LogicArray[i] != 0) {
            const num = pack0LogicArray[i] == 4 ? 3 : pack0LogicArray[i]!;
            const cards = logic.outLogicCard(tmpPack0, i, num);
            if (pack0LogicArray[i] == 4) handCards.push(logic.outLogicCard(tmpPack0, i, 1));
            handCards.push(cards);
            if (pack0LogicArray[i] == 1) mybestCardSingle = cards;
          }
        }
        if (handCards.length > 0 && handCards.length + packCard.length - 1 <= 2) {
          if (mybestCardSingle.length == 1 && logic.getLogicValue(mybestCardSingle[0]!) > logic.getLogicValue(lastOutCard[0]!)) {
            const tmpAllCard = allCard.concat();
            tmpAllCard[selfStation] = [];
            if (handCards.length + packCard.length - 1 == 1) return mybestCardSingle;
            if (logic.checkCardIsBigger(mybestCardSingle, tmpAllCard)) {
              if (partnerInfo.lastIsFriend && allCard[lastStation]!.length > 0 && !partnerInfo.doubleNTIsEnemy && !enemyWillGo) return [];
              return mybestCardSingle;
            }
          }
        }

        // 黑桃A出牌优先级控制
        const BlackAFirst = !cardObj[logic.cardCode.黑桃3] && data.i3UpGradePeople != selfStation && data.iDoubleNTPeople == 255;
        const BlackALast = partnerInfo.selfIsDoubleA3 && data.iDoubleNTPeople == 255;

        if (partnerInfo.nextIsDoubltNT) {
          // 下家是独食者: 优先出K~2的大牌压制
          if (logic.getLogicValue(lastOutCard[0]!) < logic.cardLogic.LK) {
            let topUpCard = logic.outLookingCardDESC(packCard[0]!.card, pack0LogicArray, logic.cardLogic.LK, logic.cardLogic.L2, lastOutCard, false, BlackAFirst);
            if (topUpCard.length != 0) return topUpCard;
            topUpCard = logic.outLookingCardDESC(packCard[0]!.card, pack0LogicArray, logic.cardLogic.LK, logic.cardLogic.L2, lastOutCard, true, BlackAFirst);
            if (topUpCard.length != 0) return topUpCard;
            topUpCard = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, logic.cardLogic.L2, null, lastOutCard, true, BlackAFirst);
            if (topUpCard.length != 0) return topUpCard;
            topUpCard = logic.outLookingCardDESC(packCard[0]!.card, pack0LogicArray, null, null, lastOutCard, true, BlackAFirst);
            if (topUpCard.length != 0) return topUpCard;
          }
        } else {
          // 普通模式: 先出小牌(不超过2)
          if (enemyWillGo) {
            const outSmallCard = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, enemyWillGoLogic, logic.cardLogic.L3, lastOutCard, false, BlackAFirst);
            if (outSmallCard.length != 0) return outSmallCard;
          }
          let outSmallCard = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, null, logic.cardLogic.L2, lastOutCard, false, BlackAFirst);
          if (!!partnerInfo.early && BlackALast && outSmallCard.length > 0 && outSmallCard[0] == logic.cardCode.黑桃A) {
            outSmallCard = [];
          }
          if (outSmallCard.length != 0) return outSmallCard;
        }

        // 友方出牌且友方还有牌: 根据牌大小决定是否让给友方
        // 友方出大牌(A以上): 不浪费大牌压友方，pass让友方保持控制
        // 友方出中小牌(< A): 不自动pass，继续评估敌情紧急度
        //   → 如果紧急度高(多敌威胁)，用大牌抢控制权压制敌人
        //   → 如果紧急度低，后续逻辑也不会出牌，效果等于pass
        if (!enemyWillGo && partnerInfo.lastIsFriend && allCard[lastStation]!.length > 0 && selfCard.length > 3) {
          if (logic.getLogicValue(lastOutCard[0]!) >= logic.cardLogic.LA) return [];
        }

        // 计算"敌情紧急度"，决定是否出大牌
        let enemyAdd = partnerInfo.lastIsEnemy ? (allCard[lastStation]!.length > 0 && allCard[lastStation]!.length <= 2 ? 1 : 0) : 0;
        enemyAdd = enemyWillGo ? enemyAdd + 1 : enemyAdd;
        enemyAdd = (partnerInfo.lastIsEnemy && partnerInfo.doubleNTIsEnemy) ? enemyAdd + 1 : enemyAdd;
        enemyAdd = (!partnerInfo.early && allEnemy) ? enemyAdd + 1 : enemyAdd;
        enemyAdd = data.iDoubleNTPeople == selfStation ? 2 : enemyAdd;
        enemyAdd = (partnerInfo.early && partnerInfo.selfIsDoubleA3 && data.iAUpGradePeople == 255 && data.i3UpGradePeople == 255) ? enemyAdd - 1 : enemyAdd;
        if (data.iDoubleNTPeople == 255) enemyAdd = partnerInfo.early ? enemyAdd - 1 : enemyAdd;
        enemyAdd = (partnerInfo.nextIsDoubltNT && enemyWillGo) ? enemyAdd + 100 : enemyAdd;

        // 【改进】多个敌人同时快走完(≤3张)时，额外加大紧急度
        {
          let dangerEnemyCount = 0;
          for (let dei = 0; dei < 4; dei++) {
            if (dei == selfStation) continue;
            if (partnerInfo.partnerInfo[dei] != partner.enemy) continue;
            if (allCard[dei]!.length > 0 && allCard[dei]!.length <= 3) dangerEnemyCount++;
          }
          if (dangerEnemyCount >= 2) enemyAdd += 2;
        }

        // 大牌数+紧急度 ≥ 小牌数: 出大牌压制
        if (bigCard + enemyAdd >= smallCard || (partnerInfo.lastEnemyCard && partnerInfo.lastEnemyCard.length != 0)) {
          if (enemyWillGo && !partnerInfo.lastEnemyCard) {
            for (let i = 0; i < 4; i++) {
              if (i != selfStation && partnerInfo.partnerInfo[i] == partner.enemy && allCard[i]!.length == 1 && allCard[i]![0] == logic.cardCode.黑桃3) return [];
            }
          }
          let outBigCard = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, logic.cardLogic.LA, logic.cardLogic.L3, lastOutCard, true, BlackAFirst);
          if (outBigCard.length != 0) return outBigCard;
          const minBrokenPairLogic = partnerInfo.lastEnemyCard ? logic.cardLogic.LJ : logic.cardLogic.LA;
          outBigCard = logic.outLookingCardDESC(packCard[0]!.card, pack0LogicArray, minBrokenPairLogic, logic.cardLogic.L3, lastOutCard, true, BlackAFirst);
          if (outBigCard.length != 0) return outBigCard;

          // 尝试出3（黑桃3是最大的单张）
          let selfOut3 = 0;
          const beginColor = logic.getLogicValue(lastOutCard[0]!) == logic.cardLogic.L3 ? logic.getColorValue(lastOutCard[0]!) + 1 : 0;
          for (let color = beginColor; color < 4; color++) {
            const nextColorCard = 0x10 * color + logic.cardCode.方片3;
            if (pack0CardObj[nextColorCard]) selfOut3 = nextColorCard;
          }
          if (selfOut3 != 0) {
            if (partnerInfo.lastIsEnemy && partnerInfo.doubleNTIsEnemy) {
              for (let i = 1; i < 4; i++) {
                const station = (selfStation + i) % 4;
                if (station == data.iDoubleNTPeople) break;
                const tempCard = allCard[station]!.concat();
                const tempPackCard = pack.packCard(tempCard);
                const friendWillOut = logic.outLookingCard(tempPackCard[0]!.card, null, logic.cardLogic.LA, null, lastOutCard, false, false);
                if (friendWillOut.length > 0) {
                  if (logic.compareCard([selfOut3], friendWillOut)) return [];
                }
              }
            }
            return [selfOut3];
          }
        }

        // 最后尝试拆对子出单张
        const outSmallCard2 = logic.outLookingCardDESC(packCard[0]!.card, pack0LogicArray, null, logic.cardLogic.LA, lastOutCard);
        if (outSmallCard2.length != 0) return outSmallCard2;
        if (selfCard.length == 2 && logic.getLogicValue(selfCard[0]!) > logic.getLogicValue(lastOutCard[0]!)) return [selfCard[0]!];
        break;
      }

      // ─── 跟对子 ───
      case logic.cardType.对子: {
        // 友方出的且友方只剩1-2张: 放行（让友方走完）
        // 【改进】自己也只剩2张(对子)时不放行，自己出走完
        // 【改进】自己只剩3张以内时也不放行，争取自己走完
        if (partnerInfo.lastIsFriend && allCard[lastStation]!.length > 0 && allCard[lastStation]!.length <= 2
            && allCard[selfStation]!.length != 2 && allCard[selfStation]!.length > 3) return [];

        // 最后一个敌人只剩1-2张: 从大到小找对子压
        if (partnerInfo.lastIsEnemy && partnerInfo.lastEnemyCard && partnerInfo.lastEnemyCard.length <= 2) {
          for (let i = pack0LogicArray.length - 1; i > logic.getLogicValue(lastOutCard[0]!); i--) {
            if (pack0LogicArray[i]! >= 2) return logic.outLogicCard(packCard[0]!.card, i, 2);
          }
        }

        // 计算散牌手数，判断是否值得出大对子
        const handCards2: number[][] = [];
        let mybestCardPair: number[] = [];
        let mybestCardSingle2: number[] = [];
        const tmpPack02 = packCard[0]!.card.concat();
        for (let i = 0; i < pack0LogicArray.length; i++) {
          if (pack0LogicArray[i] != 0) {
            const num = pack0LogicArray[i] == 4 ? 3 : pack0LogicArray[i]!;
            const cards = logic.outLogicCard(tmpPack02, i, num);
            if (pack0LogicArray[i] == 4) handCards2.push(logic.outLogicCard(tmpPack02, i, 1));
            handCards2.push(cards);
            if (pack0LogicArray[i] == 1) mybestCardSingle2 = cards;
            if (pack0LogicArray[i] == 2) mybestCardPair = cards;
          }
        }
        if (handCards2.length > 0 && handCards2.length + packCard.length - 1 <= 2) {
          if (mybestCardPair.length == 2 && logic.getLogicValue(mybestCardPair[0]!) > logic.getLogicValue(lastOutCard[0]!)) {
            const tmpAllCard = allCard.concat();
            tmpAllCard[selfStation] = [];
            if (handCards2.length + packCard.length - 1 == 1) return mybestCardPair;
            if ((logic.checkCardIsBigger(mybestCardSingle2, tmpAllCard), partnerInfo.lastIsFriend && allCard[lastStation]!.length > 0 && !partnerInfo.doubleNTIsEnemy && !enemyWillGo)) return [];
            if (logic.checkCardIsBigger(mybestCardPair, tmpAllCard)) return mybestCardPair;
          }
        }

        // 下家是敌人: 出10~A的对子压制
        if (partnerInfo.nextIsEnemy) {
          const topUpCard = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, logic.cardLogic.L10, logic.cardLogic.LA, lastOutCard);
          if (topUpCard.length != 0) return topUpCard;
        } else {
          // 下家是友方: 出小对子
          const outSmallCard = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, null, logic.cardLogic.LJ, lastOutCard);
          if (outSmallCard.length != 0) return outSmallCard;
        }

        // 对子的敌情紧急度
        let enemyAdd2 = partnerInfo.lastIsEnemy ? (allCard[lastStation]!.length > 0 && allCard[lastStation]!.length <= 2 ? 2 : 1) : 0;
        enemyAdd2 = enemyWillGo ? enemyAdd2 + 1 : enemyAdd2;
        enemyAdd2 = partnerInfo.doubleNTIsEnemy ? enemyAdd2 + 1 : enemyAdd2;
        enemyAdd2 = (partnerInfo.lastIsEnemy && partnerInfo.doubleNTIsEnemy) ? enemyAdd2 + 2 : enemyAdd2;
        enemyAdd2 = (partnerInfo.lastIsFriend && (allCard[lastStation]! as any) > 2) ? enemyAdd2 - 1 : enemyAdd2;
        if (!partnerInfo.doubleNTIsEnemy) enemyAdd2 = partnerInfo.early ? enemyAdd2 - 1 : enemyAdd2;

        let brokenedPair = false;
        if (enemyWillGo || bigCard + enemyAdd2 >= smallCard || (partnerInfo.lastEnemyCard && partnerInfo.lastEnemyCard.length != 0)) {
          brokenedPair = true;
          // 出J~K的大对子
          let outBigCard = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, logic.cardLogic.LJ, logic.cardLogic.LK, lastOutCard);
          if (outBigCard.length != 0) return outBigCard;
          if ((!partnerInfo.lastIsFriend && bigCard + enemyAdd2 >= smallCard + 1) || enemyWillGo) {
            const maxLogic3 = (bigCard + enemyAdd2 >= smallCard + 3) ? null : logic.cardLogic.L3;
            const outPair2 = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, null, maxLogic3, lastOutCard, true);
            if (outPair2.length != 0) return outPair2;
          }
        }

        const topUpCard2 = logic.outLookingCard(packCard[0]!.card, pack0LogicArray, null, logic.cardLogic.LJ, lastOutCard);
        if (topUpCard2.length != 0) return topUpCard2;
        if (partnerInfo.lastIsFriend || !brokenedPair) return [];

        // 尝试从三条中拆出对子
        for (let i = 0; i < packCard.length; i++) {
          if (logic.getLogicValue(packCard[i]!.card[0]!) < logic.cardLogic.LJ) continue;
          if (packCard[i]!.type == logic.cardType.三张 && logic.getLogicValue(packCard[i]!.card[0]!) > logic.getLogicValue(lastOutCard[0]!)) {
            return logic.outLogicCard(packCard[i]!.card, logic.getLogicValue(packCard[i]!.card[0]!), 2);
          }
        }
        if (enemyWillGo || partnerInfo.doubleNTIsEnemy) {
          for (let i = 1; i < packCard.length; i++) {
            if (packCard[i]!.type != logic.cardType.三带二) continue;
            if (packCard[i]!.type == logic.cardType.三带二 && logic.getLogicValue(packCard[i]!.card[3]!) > logic.getLogicValue(lastOutCard[0]!)) {
              return logic.outLogicCard(packCard[i]!.card, logic.getLogicValue(packCard[i]!.card[3]!), 2);
            }
          }
        }
        break;
      }

      // ─── 跟三张 ───
      case logic.cardType.三张: {
        if (partnerInfo.lastIsFriend && allCard[lastStation]!.length > 0 && allCard[lastStation]!.length <= 2) return [];
        for (let i = 0; i < packCard.length; i++) {
          if (packCard[i]!.type == logic.cardType.三张 && logic.getLogicValue(packCard[i]!.card[0]!) > logic.getLogicValue(lastOutCard[0]!)) return packCard[i]!.card;
        }
        if (partnerInfo.lastIsFriend) return [];
        for (let i = 0; i < packCard.length; i++) {
          if (packCard[i]!.type == logic.cardType.三带二 && logic.getLogicValue(packCard[i]!.card[0]!) > logic.getLogicValue(lastOutCard[0]!)) {
            return logic.outLogicCard(packCard[i]!.card, logic.getLogicValue(packCard[i]!.card[0]!), 3);
          }
        }
        break;
      }

      // ─── 跟五张组合 (顺子/同花/三带二/四带一/同花顺) ───
      case logic.cardType.顺子:
      case logic.cardType.同花:
      case logic.cardType.三带二:
      case logic.cardType.四带一:
      case logic.cardType.同花顺: {
        if (selfCard.length < 5) return [];

        const onlyPack = packCard[0]!.card.length == 0;
        // 友方出的且友方还剩1-2张: 放行（除非只剩5张或只有组合牌）
        if (partnerInfo.lastIsFriend && allCard[lastStation]!.length > 0 && allCard[lastStation]!.length <= 2 && allCard[selfStation]!.length != 5 && !onlyPack) return [];

        // 限制最大可出牌型（友方出牌时不需要出太大的）
        let maxType = (partnerInfo.lastIsFriend && allCard[lastStation]!.length > 0) ? logic.cardType.对子 : logic.cardType.同花顺;
        maxType = packCard[0]!.card.length == 0 ? logic.cardType.同花顺 : maxType;
        maxType = allCard[selfStation]!.length == 5 ? logic.cardType.同花顺 : maxType;
        for (let i = 1; i < packCard.length; i++) {
          if (packCard[i]!.type > maxType) continue;
          if (pack.comparePack(packCard[i]!, { type: cardType, card: lastOutCard }) > 0) return packCard[i]!.card;
        }

        // 最后一个敌人只剩1-2张或独食模式: 尝试拆牌组5张
        if (((partnerInfo.lastEnemyCard && partnerInfo.lastEnemyCard.length <= 2) || partnerInfo.doubleNTIsEnemy) && partnerInfo.lastIsEnemy) {
          // 先看友方有没有能压过的组合
          for (let i = 1; i < 4; i++) {
            const testStation = (selfStation + i) % 4;
            if (testStation == data.iDoubleNTPeople) break;
            const allPack = pack.packCard(allCard[testStation]!);
            for (let ii = 1; ii < allPack.length; ii++) {
              if (pack.comparePack(allPack[ii]!, { type: cardType, card: lastOutCard }) > 0) return [];
            }
          }
          // 尝试用四条+单张或三条+对子组出5张
          const allCardLogicArray = logic.getLogicArray(selfCard);
          const fourNum = logic.checkSameCard(selfCard, 4, allCardLogicArray, true);
          const threeNum = logic.checkSameCard(selfCard, 3, allCardLogicArray, true);
          if (fourNum.length > 0) {
            for (let i = 0; i < fourNum.length; i++) {
              let tempCard = selfCard.concat();
              const outCard = logic.outLogicCard(tempCard, fourNum[0]!, 4);
              let tempLA = logic.getLogicArray(tempCard);
              outCard.push(...logic.outLookingCard(tempCard, tempLA, null, null, [0], false, false));
              if ((outCard as any) < 5) {
                outCard.push(...logic.outLookingCard(tempCard, tempLA, null, null, [0], true, false));
                if (outCard.length == 5) {
                  if (pack.comparePack({ type: logic.cardType.四带一, card: outCard }, { type: cardType, card: lastOutCard }) > 0) return outCard;
                }
              }
            }
          }
          if (threeNum.length > 0) {
            let tempCard = selfCard.concat();
            const outCard = logic.outLogicCard(tempCard, threeNum[0]!, 3);
            let tempLA = logic.getLogicArray(tempCard);
            outCard.push(...logic.outLookingCard(tempCard, tempLA, null, null, [0, 0], false, false));
            if ((outCard as any) < 5) {
              outCard.push(...logic.outLookingCard(tempCard, tempLA, null, null, [0, 0], true, false));
            }
            if (outCard.length == 5) {
              if (pack.comparePack({ type: logic.cardType.三带二, card: outCard }, { type: cardType, card: lastOutCard }) > 0) return outCard;
            }
          }
        }
        break;
      }
    }
    return [];
  },

  /**
   * 独食模拟：模拟完整对局判断能否独食成功
   * 通过模拟自己独食后的出牌过程，看能否最先走完
   */
  doubleNT(data: RefData): number {
    if (data.iDoubleNTPeople != 255) return 0;
    const allCard: number[][] = [[], [], [], []];
    for (let i = 0; i < 4; i++) {
      if (data.tableCardData[i] == null) data.tableCardData[i] = [];
      if (!data.tableCardData[i]) continue;
      for (let ii = 0; ii < data.tableCardData[i]!.length; ii++) {
        if (data.tableCardData[i]![ii] != 0) allCard[i]!.push(data.tableCardData[i]![ii]!);
      }
      data.tableCardData[i] = allCard[i]!;
    }
    if (allCard[data.selfStation]!.length != 13) return 0;
    const logicArray = logic.getLogicArray(allCard[data.selfStation]!);
    const cardObj = logic.getAllCardObject(allCard[data.selfStation]!);
    if (logicArray[logic.cardLogic.L3]! < 2 && !cardObj[logic.cardCode.黑桃3]) return 0;

    const simdata: RefData = JSON.parse(JSON.stringify(data));
    simdata.iDoubleNTPeople = data.selfStation;

    while (true) {
      const outCard = this.outCard(simdata);
      let outCardType = 0;
      switch (outCard.length) {
        case 1: outCardType = logic.cardType.单张; break;
        case 2: outCardType = logic.cardType.对子; break;
        case 3: outCardType = logic.cardType.三张; break;
        case 5: {
          if (logic.checkSameCard(outCard, 4, logic.getLogicArray(outCard)).length != 0) { outCardType = logic.cardType.四带一; break; }
          if (logic.checkSameCard(outCard, 3, logic.getLogicArray(outCard)).length != 0) { outCardType = logic.cardType.三带二; break; }
          const run = logic.checkRun(outCard);
          const colorArray = logic.getColorArray(outCard);
          let sameColor = false;
          for (let i = 0; i < colorArray.length; i++) { if (colorArray[i] == 5) { sameColor = true; break; } }
          if (run.length > 0 && sameColor) { outCardType = logic.cardType.同花顺; break; }
          if (sameColor) { outCardType = logic.cardType.同花; break; }
          if (run.length > 0) { outCardType = logic.cardType.顺子; break; }
          break;
        }
      }
      if (outCard.length > 0) {
        simdata.lastOutCard = { card: outCard, cardType: outCardType, station: simdata.selfStation };
        for (let i = 0; i < outCard.length; i++) logic.outCodeCard(simdata.tableCardData[simdata.selfStation]!, outCard[i]!);
        if (simdata.tableCardData[simdata.selfStation]!.length == 0) return simdata.selfStation == data.selfStation ? 1 : 0;
      }
      simdata.selfStation = (simdata.selfStation + 1) % 4;
      if (simdata.selfStation == simdata.lastOutCard.station) simdata.lastOutCard.card = [];
    }
  },

  /**
   * 1v1对决策略（与独食者单挑时的特殊出牌逻辑）
   * 核心思想: 找到一手牌能压过对手且对手无法反压，同时自己剩余手数 ≤ 1
   */
  oneOnOne(player0Card: number[], player1Card: number[], lastOutCard: number[], lastOutType: number): number[] {
    if (lastOutCard.length == 3) return [];
    const player0Pack = pack.packCard(player0Card);
    let result: number[] = [];
    if (lastOutCard.length == 1 || lastOutCard.length == 2) {
      // 从大到小找能压过的牌，且对手压不回来
      result = logic.outLookingCardDESC(player0Pack[0]!.card.concat(), null, null, null, lastOutCard, false, false);
      if (result.length == 0) return [];
      if (logic.outLookingCard(player1Card.concat(), null, null, null, result, false, false).length > 0) return [];
    }
    const player0Pack0LogicArray = logic.getLogicArray(player0Pack[0]!.card);
    const player1Pack = pack.packCard(player1Card);
    if (lastOutCard.length == 5) {
      if (player0Pack.length == 1) return [];
      if (pack.comparePack(player0Pack[player0Pack.length - 1]!, { card: lastOutCard, type: lastOutType }) > 0) {
        result = player0Pack[player0Pack.length - 1]!.card;
      } else {
        return [];
      }
    }
    // 检查对手的最大组合是否能压过我方所有组合
    if (player1Pack.length > 1) {
      for (let i = 1; i < player0Pack.length; i++) {
        if (pack.comparePack(player1Pack[player1Pack.length - 1]!, player0Pack[i]!) > 0) return [];
      }
    }
    // 统计对手能压过多少手散牌
    let otherBigCount = 0;
    for (let i = 0; i < player0Pack0LogicArray.length; i++) {
      if ((player0Pack0LogicArray[i] as any).length == 0) continue;
      const outCard = logic.outLogicCard(player0Pack[0]!.card.concat(), i, player0Pack0LogicArray[i]!);
      const player1TempCard = player1Card.concat();
      if (logic.outLookingCard(player1TempCard, null, null, null, outCard, true, false).length > 0) {
        otherBigCount++;
      } else if (result.length == 0) {
        result = outCard;
      }
    }
    // 只有当对手最多压过1手时才出牌
    if (result.length == 0 || otherBigCount > 1) return [];
    return result;
  },
};

// ═══════════════════════════════════════════════════════════════════
//  ultraRolloutMove — 供 MCTS Rollout 使用的 Ultra AI 出牌接口
//  将 MCTSGameState 的原生 Card 对象格式转为内部 hex 编码 RefData，
//  调用 ai.outCard() 决策后将结果转回 Hand 对象。
//  优势：Rollout 质量远高于随机启发式，每次模拟更接近真实打法。
// ═══════════════════════════════════════════════════════════════════

/**
 * 将 MCTSGameState 的信息转为 RefData 并调用 Ultra AI 决策。
 * 供 MCTS 的 Rollout 阶段使用，无需实例化整个 Brain 类。
 *
 * @param allCards  各玩家当前手牌 (Card 对象数组，索引对应座位)
 * @param currentPlayer 当前行动玩家座位索引
 * @param lastPlay  上一手出牌；null 表示自由出牌（新轮次）
 * @param lastPlayPlayer 谁出的上一手；-1 表示新轮次
 * @param teams     各座位玩家队伍 (TeamSide)
 * @returns 要出的牌（Hand 对象）；null 表示 pass
 */
export function ultraRolloutMove(
  allCards: Card[][],
  currentPlayer: number,
  lastPlay: Hand | null,
  lastPlayPlayer: number,
  teams: TeamSide[],
): Hand | null {
  // 转换为 hex 编码（ai.outCard 会对 selfStation 数组排序，需要副本）
  const tableCardData: number[][] = allCards.map(hand => hand.map(toCode));

  // 独食玩家
  let iDoubleNTPeople = 255;
  for (let i = 0; i < teams.length; i++) {
    if (teams[i] === TeamSide.Solo) { iDoubleNTPeople = i; break; }
  }

  // 扫描当前手牌推断 ♠A/♠3 是否已出
  const spade3Code = 0x32;
  const spadeACode = 0x3D;
  let hasSpade3 = false;
  let hasSpadeA = false;
  for (const hand of tableCardData) {
    for (const code of hand) {
      if (code === spade3Code) hasSpade3 = true;
      if (code === spadeACode) hasSpadeA = true;
    }
  }
  const A3IsOut = !hasSpade3 || !hasSpadeA;

  // 构建 lastOutCard 字段
  let lastOutCardData: { card: number[] | null; cardType: number; station: number };
  if (lastPlay === null || lastPlayPlayer < 0) {
    lastOutCardData = { card: null, cardType: 0, station: currentPlayer };
  } else {
    const codes = lastPlay.cards.map(toCode);
    codes.sort((a, b) =>
      (logic.getLogicValue(a) * 0x10 + logic.getColorValue(a)) -
      (logic.getLogicValue(b) * 0x10 + logic.getColorValue(b))
    );
    lastOutCardData = {
      card: codes,
      cardType: toCardType(lastPlay.type),
      station: lastPlayPlayer,
    };
  }

  const data: RefData = {
    selfStation: currentPlayer,
    tableCardData,
    lastOutCard: lastOutCardData,
    iDoubleNTPeople,
    i3UpGradePeople: 255,  // Rollout 中无需追踪历史，partner 模块会自动扫描手牌
    iAUpGradePeople: 255,
    tableUserID: [1, 2, 3, 4],
    A3IsOut,
    deskPassword: '',
    round: 0,
    oneOnOneStation: null,
    deskConfig: [0, 0, 0, 0, 1, 1, 50],
  };

  const resultCodes = ai.outCard(data);
  if (resultCodes.length === 0) return null;

  const cards = resultCodes.map(toCard);
  return detectHand(cards) ?? null;
}

// ═══════════════════════════════════════════════════════════════════
//  RuleUltraBrain — 桥接 BotContext 与内部 AI 引擎
//  继承 BaseRuleBrain，在 decide() 中将 BotContext 转为 RefData，
//  调用 ai.outCard() 决策，再将结果转回 Card 对象
// ═══════════════════════════════════════════════════════════════════

export default class RuleUltraBrain extends BaseRuleBrain {
  readonly name = 'Ultra';
  readonly displayName = 'Ultra(完整复刻)';
  readonly engineId = 'rule_ultra' as const;

  async decide(ctx: BotContext): Promise<BotDecision> {
    if (!ctx.godMode) return this.getConservativeFollowUp(ctx);
    const data = this.buildRefData(ctx);

    // 报牌阶段：调用 doubleNT 模拟能否独食获胜
    if (ctx.isDeclaringPhase) {
      const result = ai.doubleNT(data);
      return { action: result === 1 ? 'declare' : 'pass_declare', cards: [] };
    }

    const resultCodes = ai.outCard(data);
    if (resultCodes.length === 0) return { action: 'pass', cards: [] };
    return { action: 'play', cards: toCardArray(resultCodes) };
  }

  /**
   * 将 BotContext (游戏框架的上下文) 转为 RefData (AI引擎的数据结构)
   * 包含: 所有玩家手牌、上一手出牌、独食/♠A/♠3信息等
   */
  private buildRefData(ctx: BotContext): RefData {
    const gm = ctx.godMode!;
    const seatOrder = ctx.seatOrder;
    const selfStation = seatOrder.indexOf(ctx.myPlayerId);

    // 转换所有玩家手牌为 hex 编码
    const tableCardData: number[][] = [[], [], [], []];
    for (let i = 0; i < 4; i++) {
      const pid = seatOrder[i]!;
      const cards = pid === ctx.myPlayerId ? ctx.myCards : (gm.allPlayersCards[pid] ?? []);
      tableCardData[i] = toCodeArray(cards);
    }

    // 转换上一手出牌信息
    let lastOutCardData: { card: number[] | null; cardType: number; station: number };
    if (ctx.lastPlay && ctx.lastPlayPlayerId && !ctx.isNewRound && !ctx.isFirstTurn) {
      const codes = toCodeArray(ctx.lastPlay.cards);
      codes.sort((a, b) =>
        (logic.getLogicValue(a) * 0x10 + logic.getColorValue(a)) -
        (logic.getLogicValue(b) * 0x10 + logic.getColorValue(b))
      );
      lastOutCardData = {
        card: codes,
        cardType: toCardType(ctx.lastPlay.type),
        station: seatOrder.indexOf(ctx.lastPlayPlayerId),
      };
    } else {
      lastOutCardData = { card: null, cardType: 0, station: selfStation };
    }

    // 追踪谁打出过♠3/♠A（用于敌友判断）
    let i3UpGradePeople = 255;
    let iAUpGradePeople = 255;
    for (const action of ctx.recentHistory) {
      for (const card of action.cards) {
        if (isSpadeThree(card)) i3UpGradePeople = seatOrder.indexOf(action.playerId);
        if (isSpadeAce(card)) iAUpGradePeople = seatOrder.indexOf(action.playerId);
      }
    }

    // 独食玩家: 从 TeamSide.Solo 读取
    let iDoubleNTPeople = 255;
    for (let i = 0; i < 4; i++) {
      const pid = seatOrder[i]!;
      if (gm.allTeams[pid] === TeamSide.Solo) {
        iDoubleNTPeople = i;
        break;
      }
    }

    const A3IsOut = ctx.playedOutCards.some(isSpadeAce) || ctx.playedOutCards.some(isSpadeThree);

    return {
      selfStation,
      tableCardData,
      lastOutCard: lastOutCardData,
      iDoubleNTPeople,
      i3UpGradePeople,
      iAUpGradePeople,
      tableUserID: [1, 2, 3, 4],
      A3IsOut,
      deskPassword: '',
      round: 0,
      oneOnOneStation: null,
      deskConfig: [0, 0, 0, 0, 1, 1, 50],
    };
  }
}
