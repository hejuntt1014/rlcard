export interface GameConfig {
  turnTimeout: number;
  readyCheckTimeout: number;
  reconnectTimeout: number;
  emptyRoomTimeout: number;
  minStakes: number;
  maxStakes: number;
  defaultStakes: number;
  initialCoins: number;
  dailySignInReward: number;

  allowFlush: boolean;
  allowStraightFlush: boolean;
  allowFourWithOne: boolean;
  threeIsHighest: boolean;
  suitOrder: readonly string[];
  rankOrder: readonly string[];

  /** true=必须同时持有♠3和♠A才能叫牌, false=自由叫牌 */
  declareRequiresBothSpades: boolean;
  /** 顺子最低起点: '3' 或 '4' */
  straightStartRank: '3' | '4';
  /** 顺子最高终点: 'K' 或 'A' */
  straightEndRank: 'K' | 'A';

  name: string;
  description: string;
}

export const ZHANJIANG_STANDARD: GameConfig = {
  turnTimeout: 30_000,
  readyCheckTimeout: 3_000,
  reconnectTimeout: 60_000,
  emptyRoomTimeout: 300_000,
  minStakes: 10,
  maxStakes: 100_000,
  defaultStakes: 100,
  initialCoins: 1000,
  dailySignInReward: 200,

  allowFlush: true,
  allowStraightFlush: true,
  allowFourWithOne: true,
  threeIsHighest: true,
  suitOrder: ['spade', 'heart', 'club', 'diamond'],
  rankOrder: ['3', '2', 'A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4'],

  declareRequiresBothSpades: false,
  straightStartRank: '3',
  straightEndRank: 'A',

  name: '湛江标准版',
  description: '52张牌，3最大4最小，黑桃A3配对，方块4先出',
};
