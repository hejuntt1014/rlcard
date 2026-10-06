export interface ChatMessage {
  id: string;
  roomId: string;
  senderId: string;
  senderName: string;
  content: string;
  type: 'text' | 'quick' | 'system';
  timestamp: number;
}

export const QUICK_MESSAGES = [
  '快点吧，我等到花儿都谢了！',
  '不要走，决战到天亮！',
  '大家好，多多关照！',
  '你的牌打得太好了！',
  '不好意思，我要出大牌了！',
  '有没有搞错？！',
  '等一下，我想想...',
  '过过过！',
] as const;
