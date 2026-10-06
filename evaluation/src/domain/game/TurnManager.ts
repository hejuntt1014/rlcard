import type { Player } from './Player.js';

export interface TurnState {
  currentPlayerId: string;
  lastPlayPlayerId: string | null;
  passCount: number;
  isNewRound: boolean;
}

export class TurnManager {
  private players: Player[];
  private currentIndex: number;
  private _passCount: number = 0;
  private _lastPlayPlayerId: string | null = null;
  private _isNewRound: boolean = true;

  constructor(players: Player[], startIndex: number) {
    this.players = players;
    this.currentIndex = startIndex;
  }

  get currentPlayerId(): string {
    return this.players[this.currentIndex]!.id;
  }

  get passCount(): number {
    return this._passCount;
  }

  get isNewRound(): boolean {
    return this._isNewRound;
  }

  get lastPlayPlayerId(): string | null {
    return this._lastPlayPlayerId;
  }

  getState(): TurnState {
    return {
      currentPlayerId: this.currentPlayerId,
      lastPlayPlayerId: this._lastPlayPlayerId,
      passCount: this._passCount,
      isNewRound: this._isNewRound,
    };
  }

  /**
   * 当前玩家出牌后，推进到下一个有牌的玩家
   */
  advanceAfterPlay(): { nextPlayerId: string; isNewRound: boolean } {
    this._lastPlayPlayerId = this.currentPlayerId;
    this._passCount = 0;
    this._isNewRound = false;
    this.moveToNextActivePlayer();
    return { nextPlayerId: this.currentPlayerId, isNewRound: false };
  }

  /**
   * 当前玩家 pass 后，推进到下一个有牌的玩家
   * 如果所有其他活跃玩家都 pass 了，则新一轮由最后出牌的人开始
   */
  advanceAfterPass(): { nextPlayerId: string; isNewRound: boolean } {
    this._passCount++;
    this.moveToNextActivePlayer();

    const activeCount = this.players.filter(p => p.rank === null).length;
    const lastPlayerIsActive = this._lastPlayPlayerId !== null &&
      this.players.some(p => p.id === this._lastPlayPlayerId && p.rank === null);
    const maxPasses = lastPlayerIsActive ? activeCount - 1 : activeCount;

    if (this._passCount >= maxPasses) {
      this._passCount = 0;
      this._isNewRound = true;
      if (this._lastPlayPlayerId) {
        const lastPlayerIdx = this.players.findIndex(p => p.id === this._lastPlayPlayerId);
        if (lastPlayerIdx !== -1) {
          this.currentIndex = lastPlayerIdx;
          if (!this.players[this.currentIndex]!.rank) {
            this._lastPlayPlayerId = null;
            return { nextPlayerId: this.currentPlayerId, isNewRound: true };
          }
          this.moveToNextActivePlayer();
        }
      }
      this._lastPlayPlayerId = null;
      return { nextPlayerId: this.currentPlayerId, isNewRound: true };
    }

    return { nextPlayerId: this.currentPlayerId, isNewRound: false };
  }

  /**
   * 当某玩家出完牌（打出最后的手牌）后，推进到下一个有牌的玩家
   * 出完牌也是一次"出牌"，需要重置 pass 计数
   */
  advanceAfterFinish(): { nextPlayerId: string; isNewRound: boolean } {
    this._lastPlayPlayerId = this.currentPlayerId;
    this._passCount = 0;
    this._isNewRound = false;
    this.moveToNextActivePlayer();
    return { nextPlayerId: this.currentPlayerId, isNewRound: false };
  }

  /**
   * 按 seatIndex 递增顺序移动到下一个还有手牌的玩家
   */
  private moveToNextActivePlayer(): void {
    const totalPlayers = this.players.length;
    for (let i = 1; i <= totalPlayers; i++) {
      const nextIdx = (this.currentIndex + i) % totalPlayers;
      const player = this.players[nextIdx]!;
      if (player.rank === null) {
        this.currentIndex = nextIdx;
        return;
      }
    }
  }
}
