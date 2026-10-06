import type { Card } from './card.js';
import type { GameState, GameEvent } from './game.js';
import type { DissolveVoteResult, DissolveVoteState, RoomInfo, AIEngineType, AIEngineInfo, RoomOptions, RoomSeriesSettlement } from './room.js';
import type { ChatMessage } from './message.js';

export interface ServerToClientEvents {
  'room:updated': (room: RoomInfo) => void;
  'room:list': (rooms: RoomInfo[]) => void;
  'room:joined': (room: RoomInfo) => void;
  'room:left': (roomId: string) => void;
  'room:dissolved': (roomId: string) => void;
  'room:dissolve_vote': (vote: DissolveVoteState) => void;
  'room:dissolve_vote_result': (result: DissolveVoteResult) => void;
  'session:restored': () => void;

  'match:searching': () => void;
  'match:found': (room: RoomInfo) => void;
  'match:cancelled': () => void;

  'game:started': (state: GameState) => void;
  'game:state': (state: GameState) => void;
  'game:event': (event: GameEvent) => void;
  'game:play_rejected': (reason: string) => void;
  'game:hint': (hands: Card[][]) => void;
  'game:closed': (room: RoomInfo) => void;
  'game:final_settlement': (settlement: RoomSeriesSettlement) => void;

  'chat:message': (message: ChatMessage) => void;

  'error': (message: string) => void;
}

export interface ClientToServerEvents {
  'room:create': (data: {
    roomName: string;
    stakes: number;
    ruleConfig?: {
      declareRequiresBothSpades: boolean;
      straightStartRank: '3' | '4';
      straightEndRank: 'K' | 'A';
    };
    roomOptions?: RoomOptions;
  }, ack: (room: RoomInfo) => void) => void;
  'room:join': (data: { roomId: string }, ack: (room: RoomInfo | null, error?: string) => void) => void;
  'room:leave': (ack: (result: { ok: boolean; error?: string }) => void) => void;
  'room:dissolve_request': (ack: (result: { ok: boolean; error?: string }) => void) => void;
  'room:dissolve_vote': (data: { agree: boolean }) => void;
  'room:ready': () => void;
  'room:unready': () => void;
  'room:add_ai': (data: { seatIndex: number; aiEngine?: AIEngineType }) => void;
  'room:remove_ai': (data: { seatIndex: number }) => void;
  'room:update_ai_engine': (data: { seatIndex: number; aiEngine: AIEngineType }) => void;
  'room:list': (ack: (rooms: RoomInfo[]) => void) => void;
  'room:get_ai_engines': (ack: (engines: AIEngineInfo[]) => void) => void;

  'match:request': (data: { stakesLevel: string }) => void;
  'match:cancel': () => void;

  'game:play': (data: { cards: Card[]; actionId: string }) => void;
  'game:pass': (data: { actionId: string }) => void;
  'game:declare': (data: { actionId: string }) => void;
  'game:pass_declare': (data: { actionId: string }) => void;
  'game:hint': () => void;
  'game:trustee': (data: { enable: boolean }) => void;
  'game:continue': () => void;
  /** Confirms that a replayable final settlement has reached the client. */
  'game:final_settlement_ack': (data: { roomId: string }) => void;

  'chat:send': (data: { content: string; type: 'text' | 'quick' }) => void;

  'spectate:join': (data: { roomId: string }) => void;
  'spectate:leave': () => void;
}

export interface InterServerEvents {
  ping: () => void;
}

export interface SocketData {
  userId: string;
  username: string;
}
