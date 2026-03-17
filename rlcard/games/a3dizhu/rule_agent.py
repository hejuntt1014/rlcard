"""
A3 地主贪心规则 Agent（用于评估 RL 训练效果）

策略：
  自由出牌：
    1. 剩1张 → 出掉赢了
    2. 剩2张且有对子 → 出对子
    3. 优先出最小单张（保留大牌）
  跟牌：
    1. 炸弹/四带一能管就管
    2. 对手出的 → 能管就管（出最小能赢的）
    3. 队友出的 → 不管（pass）
    4. 未知关系 → 能管就管
"""

from __future__ import annotations
import numpy as np
from typing import Optional
from .card import Card
from .hand import Hand, can_beat, compare_hands, SINGLE, PAIR, FOUR_WITH_ONE, STRAIGHT_FLUSH
from .hint import get_playable_hands


class GreedyRuleAgent:
    """贪心规则 Agent，用于评估 RL 模型的真实水平"""

    def __init__(self):
        self.use_raw = True  # 直接使用原始动作（Hand 对象）

    def step(self, state: dict):
        """训练时调用（和 eval_step 相同）"""
        return self._decide(state)

    def eval_step(self, state: dict):
        """评估时调用"""
        action = self._decide(state)
        return action, {}

    def _decide(self, state: dict):
        raw = state.get('raw_obs', state)
        legal_actions = raw.get('legal_actions', [])
        if not legal_actions:
            return None

        my_cards = raw['current_hand']
        last_play = raw['last_play']
        teams = raw.get('teams', [])
        player_id = raw.get('player_id', 0)
        last_play_player = raw.get('last_play_player', -1)

        # 自由出牌
        if last_play is None:
            return self._free_play(my_cards, legal_actions)

        # 跟牌
        return self._follow_play(my_cards, legal_actions, last_play,
                                  teams, player_id, last_play_player)

    def _free_play(self, my_cards: list[Card], legal_actions) -> Optional[Hand]:
        """自由出牌策略"""
        playable = [h for h in legal_actions if h is not None]
        if not playable:
            return None

        n = len(my_cards)

        # 剩1张，直接出
        if n == 1:
            return playable[0]

        # 剩2张，如果有对子出对子（一手走完）
        if n == 2:
            pairs = [h for h in playable if h.size == 2]
            if pairs:
                return pairs[0]

        # 优先出最小单张
        singles = sorted(
            [h for h in playable if h.size == 1],
            key=lambda h: h.primary_card.score()
        )
        if singles:
            return singles[0]

        # 没单张就出最小的任意牌
        playable.sort(key=lambda h: (h.size, h.primary_card.score()))
        return playable[0]

    def _follow_play(self, my_cards, legal_actions, last_play,
                      teams, player_id, last_play_player):
        """跟牌策略"""
        beating = [h for h in legal_actions if h is not None]
        if not beating:
            return None  # pass

        # 判断关系
        is_teammate = self._is_teammate(teams, player_id, last_play_player)

        # 队友出的 → pass（不管）
        if is_teammate:
            return None

        # 剩余手牌少（≤3张） → 必须抢先出完，能管就管
        if len(my_cards) <= 3 and beating:
            beating.sort(key=lambda h: h.primary_card.score())
            return beating[0]

        # 对手/未知关系 → 出最小能赢的
        beating.sort(key=lambda h: h.primary_card.score())
        return beating[0]

    def _is_teammate(self, teams, my_id, other_id) -> bool:
        """判断 other_id 是否是队友"""
        if other_id < 0 or other_id >= len(teams) or my_id >= len(teams):
            return False
        my_team = teams[my_id]
        other_team = teams[other_id]
        if my_team == 'unknown' or other_team == 'unknown':
            return False
        return my_team == other_team


class RandomRuleAgent:
    """纯随机 Agent（5% 噪音注入，提升鲁棒性）"""
    def __init__(self):
        self.use_raw = True

    def step(self, state):
        import random as _rng
        raw = state if 'legal_actions' in state else state.get('raw_obs', state)
        legal = raw.get('legal_actions', [])
        if not legal:
            return None
        return _rng.choice(legal)

    def eval_step(self, state):
        return self.step(state), {}
