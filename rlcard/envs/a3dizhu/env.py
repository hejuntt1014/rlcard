"""
A3 地主 RLCard 环境接口

状态特征编码（参考 DouZero 的设计）：

每个玩家的观测向量包含：
  [0:52]   - 自己当前手牌（52 维 one-hot）
  [52:104] - 上一手出牌（52 维，pass=全零）
  [104:416]- 最近 6 步出牌历史（6 × 52 维，pass=全零）
  [416:468]- 玩家0已打出的牌（累计，52 维）
  [468:520]- 玩家1已打出的牌（52 维）
  [520:572]- 玩家2已打出的牌（52 维）
  [572:624]- 玩家3已打出的牌（52 维）
  [624:628]- 各玩家剩余手牌数（4 维，归一化 /13）
  [628:632]- 队伍 one-hot：[spade_a3, opponent, solo, unknown] × 本玩家
  [632:636]- 当前玩家是否自由出牌轮（1 维）+ 是否第一手（1 维）+ pass计数/3（1维）+ 填充（1维）

总计：636 维（约）

动作用 52 维 one-hot 表示（哪些牌被打出），pass=全零。
DMC 的 action_feature 就是这种格式。
"""

from __future__ import annotations
from collections import OrderedDict
import numpy as np

from rlcard.envs import Env
from rlcard.games.a3dizhu import Game
from rlcard.games.a3dizhu.utils import (
    hand_to_feature, cards_to_feature, NUM_CARDS, ACTION_FEATURE_DIM,
)
from rlcard.games.a3dizhu.game import TEAM_SPADE_A3, TEAM_OPPONENT, TEAM_SOLO, TEAM_UNKNOWN

HISTORY_LEN = 6       # 保留最近 N 步出牌历史
TEAMS_ORDER = [TEAM_SPADE_A3, TEAM_OPPONENT, TEAM_SOLO, TEAM_UNKNOWN]

# 状态向量总维度
# 手牌(52) + 上一手(52) + 历史(6×52) + 4×已打出(4×52)
# + 剩余数one-hot(4×14) + 队伍(4×4) + 杂项(4)
STATE_DIM = (
    NUM_CARDS               # 自己手牌
    + NUM_CARDS             # 上一手出牌
    + HISTORY_LEN * NUM_CARDS  # 历史
    + 4 * NUM_CARDS         # 4人已打出牌
    + 4 * 14                # 剩余手牌数 one-hot
    + 4 * len(TEAMS_ORDER)  # 各玩家队伍 one-hot
    + 4                     # 杂项
)


class A3DizhuEnv(Env):
    """A3 地主强化学习环境（适配 RLCard DMC 训练框架）

    支持混合对手训练（League Training）：
      每个座位独立掷骰子决定是否用规则 agent，消除位置偏差。
      config['greedy_ratio'] = 0.10  → 每座位 10% 概率为贪婪 agent
      config['random_ratio'] = 0.05  → 每座位 5% 概率为纯随机 agent
    """

    name = 'a3dizhu'

    def __init__(self, config):
        self.game = Game()
        self.greedy_ratio = config.get('greedy_ratio', 0.0)
        self.random_ratio = config.get('random_ratio', 0.0)
        # 兼容旧的 rule_opponent_ratio 参数
        old_ratio = config.get('rule_opponent_ratio', 0.0)
        if old_ratio > 0 and self.greedy_ratio == 0:
            self.greedy_ratio = old_ratio * 0.67  # 2/3 给贪婪
            self.random_ratio = old_ratio * 0.33  # 1/3 给随机
        super().__init__(config)
        self.state_shape = [[STATE_DIM] for _ in range(4)]
        self.action_shape = [[ACTION_FEATURE_DIM] for _ in range(4)]
        self._played_cards = [[] for _ in range(4)]
        self._action_history: list = []
        self._key_to_hand: dict[str, object] = {}
        self._greedy_agent = None
        self._random_agent = None
        self._seat_agents: dict[int, object] = {}  # seat → agent（只对规则seat有值）

    # ─── RLCard 接口实现 ───────────────────────────────────────────────────────

    def reset(self):
        import random as _rng
        self._played_cards = [[] for _ in range(4)]
        self._action_history = []
        self._key_to_hand: dict[str, object] = {}

        # 每个座位独立掷骰子，决定用什么 agent
        self._seat_agents = {}
        for seat in range(4):
            roll = _rng.random()
            if roll < self.greedy_ratio:
                if self._greedy_agent is None:
                    from rlcard.games.a3dizhu.rule_agent import GreedyRuleAgent
                    self._greedy_agent = GreedyRuleAgent()
                self._seat_agents[seat] = self._greedy_agent
            elif roll < self.greedy_ratio + self.random_ratio:
                if self._random_agent is None:
                    from rlcard.games.a3dizhu.rule_agent import RandomRuleAgent
                    self._random_agent = RandomRuleAgent()
                self._seat_agents[seat] = self._random_agent

        state, player_id = self.game.init_game()
        self.action_recorder = []
        return self._extract_state(state), player_id

    def step(self, action, raw_action=False):
        """action: Hand 对象或 None(pass)"""
        if not raw_action:
            action = self._decode_action(action)

        self.timestep += 1
        player_id = self.game.get_player_id()
        self.action_recorder.append((player_id, action))
        self._action_history.append((player_id, action))

        # 追踪已打出的牌
        if action is not None:
            self._played_cards[player_id].extend(action.cards)

        next_state, next_player_id = self.game.step(action)
        return self._extract_state(next_state), next_player_id

    def run(self, is_training=False):
        """覆盖父类 run()，支持混合对手训练。
        规则 agent 座位的动作不记录到轨迹中（不用于训练更新）。"""
        if not self._seat_agents:
            return super().run(is_training=is_training)

        trajectories = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()

        trajectories[player_id].append(state)
        while not self.is_over():
            if player_id in self._seat_agents:
                raw_state = state.get('raw_obs', state)
                action = self._seat_agents[player_id].step(raw_state)
                next_state, next_player_id = self.step(action, raw_action=True)
                trajectories[player_id].append('pass')
            else:
                if not is_training:
                    action, _ = self.agents[player_id].eval_step(state)
                else:
                    action = self.agents[player_id].step(state)
                next_state, next_player_id = self.step(action, self.agents[player_id].use_raw)
                trajectories[player_id].append(action)

            state = next_state
            player_id = next_player_id

            if not self.game.is_over():
                trajectories[player_id].append(state)

        for pid in range(self.num_players):
            s = self.get_state(pid)
            trajectories[pid].append(s)

        payoffs = self.get_payoffs()
        return trajectories, payoffs

    def get_payoffs(self):
        return self.game.get_payoffs()

    def get_perfect_information(self):
        s = self.game.state
        return {
            'all_hands': s.hands,
            'current_player': s.current_player,
            'last_play': s.last_play,
            'rankings': s.rankings,
            'teams': s.teams,
        }

    def get_action_feature(self, action) -> np.ndarray:
        """动作特征：52 维 one-hot，pass=全零。
        action 可能是 Hand 对象、None、或字符串 key（DMC act() 传入的是 key）"""
        if action is None or action == 'pass':
            return hand_to_feature(None)
        if isinstance(action, str):
            hand = self._key_to_hand.get(action, None)
            return hand_to_feature(hand)
        return hand_to_feature(action)

    # ─── 内部方法 ──────────────────────────────────────────────────────────────

    def _extract_state(self, raw_state: dict) -> OrderedDict:
        player_id = raw_state['player_id']
        obs = self._encode_obs(raw_state, player_id)
        legal_actions = self._get_legal_actions(raw_state)

        extracted = OrderedDict({
            'obs': obs,
            'legal_actions': legal_actions,
        })
        extracted['raw_obs'] = raw_state
        extracted['raw_legal_actions'] = raw_state['legal_actions']
        extracted['action_record'] = self.action_recorder
        return extracted

    def _encode_obs(self, raw_state: dict, player_id: int) -> np.ndarray:
        """全部使用 int8 类型，与 DMC buffer 兼容。"""
        parts = []

        # 1. 自己手牌（52维，0/1）
        my_cards = raw_state['current_hand']
        parts.append(cards_to_feature(my_cards))

        # 2. 上一手出牌（52维，0/1）
        last_play = raw_state['last_play']
        parts.append(hand_to_feature(last_play))

        # 3. 最近 HISTORY_LEN 步历史（HISTORY_LEN × 52 维）
        history = self._action_history[-HISTORY_LEN:]
        for _, h in history:
            parts.append(hand_to_feature(h))
        for _ in range(HISTORY_LEN - len(history)):
            parts.append(np.zeros(NUM_CARDS, dtype=np.int8))

        # 4. 4人已打出的牌（4 × 52 维）
        for i in range(4):
            parts.append(cards_to_feature(self._played_cards[i]))

        # 5. 各玩家剩余手牌数 one-hot（4 × 14 维）
        # 0-13张 → 14维 one-hot，比归一化 float 对 int8 buffer 更精确
        all_hands = raw_state['all_hands']
        for i in range(4):
            n = min(len(all_hands[i]), 13)
            one_hot = np.zeros(14, dtype=np.int8)
            one_hot[n] = 1
            parts.append(one_hot)

        # 6. 各玩家队伍 one-hot（4 × 4 维）
        # 每个玩家的队伍信息都编码（不只是自己的），让AI学习全局合作关系
        teams = raw_state['teams']
        for i in range(4):
            team = teams[i] if i < len(teams) else TEAM_UNKNOWN
            team_vec = np.zeros(len(TEAMS_ORDER), dtype=np.int8)
            if team in TEAMS_ORDER:
                team_vec[TEAMS_ORDER.index(team)] = 1
            parts.append(team_vec)

        # 7. 杂项（4 维, int8: 0 或 1）
        misc = np.zeros(4, dtype=np.int8)
        misc[0] = 1 if raw_state['last_play'] is None else 0       # 是否自由出牌
        misc[1] = 1 if raw_state['is_first_turn'] else 0            # 是否第一手
        misc[2] = min(raw_state.get('pass_count', 0), 3)            # pass计数 (0-3)
        misc[3] = len(raw_state['rankings'])                        # 已完成玩家数 (0-4)
        parts.append(misc)

        return np.concatenate(parts)

    def _get_legal_actions(self, raw_state: dict) -> dict:
        """返回 {动作: 特征向量} 的字典。DMC 需要这种格式。"""
        legal_hands = raw_state['legal_actions']
        result = {}
        has_pass = False
        for h in legal_hands:
            if h is None:
                has_pass = True
                continue
            feat = hand_to_feature(h)
            key = _hand_key(h)
            result[key] = feat
            self._key_to_hand[key] = h
        # 只在跟牌时才有 pass 选项（自由出牌不能 pass）
        if has_pass or raw_state.get('last_play') is not None:
            result['pass'] = hand_to_feature(None)
            self._key_to_hand['pass'] = None
        return result

    def _decode_action(self, action):
        """将动作 key（字符串）转回 Hand 对象（或 None=pass）"""
        if action == 'pass' or action is None:
            return None
        # 如果已经是 Hand 对象（raw_action=True 时），直接返回
        if hasattr(action, 'cards'):
            return action
        # 从 key 查找 Hand 对象
        return self._key_to_hand.get(action, None)

    def get_state(self, player_id: int) -> OrderedDict:
        raw_state = self.game.get_state(player_id)
        return self._extract_state(raw_state)


def _hand_key(hand) -> str:
    """将 Hand 转为可哈希的字符串 key"""
    if hand is None:
        return 'pass'
    ids = sorted(c.to_id() for c in hand.cards)
    return '|'.join(ids)
