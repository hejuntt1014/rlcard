"""
A3 地主 RLCard 环境接口

状态特征编码（STATE_DIM = 850 维）：

每个玩家的观测向量包含（使用相对位置编码，slot 0=我, 1=下家, 2=对家, 3=上家）：
  [0:52]       - 自己当前手牌（52 维 one-hot）
  [52:104]     - 上一手出牌（52 维，pass/自由出牌=全零）
  [104:108]    - 上一手出牌者（4 维相对位置 one-hot；自由出牌=全零）
  [108:564]    - 最近 8 步出牌历史（8 × 57 维）
                 每步 = 玩家(4D 相对位置 one-hot) + 是否有效(1D) + 牌面(52D)
                 空槽 valid=0 + 全零，有效步 valid=1
  [564:772]    - 4 人已打出的牌（4 × 52 维，相对位置顺序）
  [772:828]    - 4 人剩余手牌数 one-hot（4 × 14 维，相对位置，0-13 张）
  [828:844]    - 4 人队伍 one-hot（4 × 4 维，相对位置）
               ※ slot 0（我）使用 actual_teams（自己知道自己的队伍，BUG-3修复）
               ※ 其他 slot 使用 observed_teams（逐步暴露，BUG-1/2已修复）
  [844:850]    - 杂项（6 维）：自由出牌、是否第一手、pass 计数、已完成人数、是否独食、是否报牌阶段

总计：850 维

动作用 52 维 one-hot 表示（哪些牌被打出），pass=全零。
报牌阶段特殊编码：
  declare 动作特征 = all-ones (52 维全 1，与任何真实出牌不同)
  pass_declare 动作特征 = all-zeros (52 维全 0，同 in-game pass)
  通过状态中的 is_declaration_phase=1 标志区分两个阶段的 pass 语义。

重要：
- observed_teams 现在正确处理 SOLO（同一人打出♠3+♠A），以及全局推断
  （两人各打一张时，其余人自动推断为 OPPONENT）
- 相对位置编码确保权重共享时所有座位输入格式一致
- 历史记录中包含玩家身份（4D one-hot），AI 可区分"队友"和"对手"的出牌
"""

from __future__ import annotations
from collections import OrderedDict
import numpy as np

from rlcard.envs import Env
from rlcard.games.a3dizhu import Game
from rlcard.games.a3dizhu.utils import (
    hand_to_feature, cards_to_feature, NUM_CARDS, ACTION_FEATURE_DIM,
)
from rlcard.games.a3dizhu.game import (
    TEAM_SPADE_A3, TEAM_OPPONENT, TEAM_SOLO, TEAM_UNKNOWN,
    compute_step_reward,
)

HISTORY_LEN = 8          # 保留最近 N 步出牌历史（所有玩家加起来）
HISTORY_STEP_DIM = 4 + 1 + NUM_CARDS  # 玩家(4D 相对) + 是否有效(1D) + 牌面(52D) = 57D
TEAMS_ORDER = [TEAM_SPADE_A3, TEAM_OPPONENT, TEAM_SOLO, TEAM_UNKNOWN]

# 状态向量总维度
# 手牌(52) + 上一手(52) + 上一手出牌者(4)
# + 历史(8×57=456) + 4×已打出(4×52=208) + 剩余数one-hot(4×14=56)
# + 队伍(4×4=16) + 杂项(6)
STATE_DIM = (
    NUM_CARDS                           # 1. 自己手牌
    + NUM_CARDS                         # 2. 上一手出牌
    + 4                                 # 3. 上一手出牌者（相对位置）
    + HISTORY_LEN * HISTORY_STEP_DIM   # 4. 历史 8×57=456
    + 4 * NUM_CARDS                     # 5. 4人已打出牌（相对位置）
    + 4 * 14                            # 6. 剩余牌数 one-hot（相对位置）
    + 4 * len(TEAMS_ORDER)              # 7. 队伍（相对位置）
    + 6                                 # 8. 杂项（含 is_declaration_phase）
)
# = 52+52+4+456+208+56+16+6 = 850


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
            self.greedy_ratio = old_ratio * 0.67
            self.random_ratio = old_ratio * 0.33
        super().__init__(config)
        self.state_shape = [[STATE_DIM] for _ in range(4)]
        self.action_shape = [[ACTION_FEATURE_DIM] for _ in range(4)]
        self._played_cards = [[] for _ in range(4)]
        self._action_history: list = []
        self._key_to_hand: dict[str, object] = {}
        self._greedy_agent = None
        self._random_agent = None
        self._seat_agents: dict[int, object] = {}

    # ─── RLCard 接口实现 ───────────────────────────────────────────────────────

    def reset(self):
        import random as _rng
        self._played_cards = [[] for _ in range(4)]
        self._action_history = []
        self._key_to_hand: dict[str, object] = {}
        self._step_rewards: list[list[float]] = [[] for _ in range(4)]

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
        """action: 'declare'(报牌) | Hand(出牌) | None(pass/不报)"""
        if not raw_action:
            action = self._decode_action(action)

        self.timestep += 1
        player_id = self.game.get_player_id()
        prev_state = self.game.state

        # 报牌阶段（declare / pass_declare）不记入出牌历史，只记真实出牌动作
        if not self.game.state.is_declaration_phase:
            self.action_recorder.append((player_id, action))
            self._action_history.append((player_id, action))

        if action is not None and action != 'declare' and hasattr(action, 'cards'):
            self._played_cards[player_id].extend(action.cards)

        next_state, next_player_id = self.game.step(action)

        sr = compute_step_reward(
            prev_state, action, self.game.state, player_id)
        self._step_rewards[player_id].append(sr)

        return self._extract_state(next_state), next_player_id

    def run(self, is_training=False):
        """覆盖父类 run()，支持混合对手训练 + 训练时奖励塑形 + 中间奖励。

        返回 (trajectories, payoffs, step_rewards)
          step_rewards[p] = 玩家 p 每步的中间奖励列表
        """
        if not self._seat_agents:
            trajectories, _ = super().run(is_training=is_training)
            payoffs = self.get_training_payoffs() if is_training else self.get_payoffs()
            step_rewards = [list(sr) for sr in self._step_rewards]
            return trajectories, payoffs, step_rewards

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
                next_state, next_player_id = self.step(
                    action, self.agents[player_id].use_raw)
                trajectories[player_id].append(action)

            state = next_state
            player_id = next_player_id

            if not self.game.is_over():
                trajectories[player_id].append(state)

        for pid in range(self.num_players):
            s = self.get_state(pid)
            trajectories[pid].append(s)

        payoffs = self.get_training_payoffs() if is_training else self.get_payoffs()
        step_rewards = [list(sr) for sr in self._step_rewards]
        return trajectories, payoffs, step_rewards

    def get_payoffs(self):
        """原始游戏回报（评估用）"""
        return self.game.get_payoffs()

    def get_training_payoffs(self):
        """带奖励塑形的回报（仅训练用）"""
        from rlcard.games.a3dizhu.game import compute_training_payoffs, compute_declared_payoffs
        s = self.game.state
        if s.is_declared and s.declarant >= 0:
            return compute_declared_payoffs(s.rankings, s.declarant, s.num_players)
        return compute_training_payoffs(s.rankings, s.actual_teams, s.is_solo, s.num_players)

    def get_aux_targets(self):
        """返回每个玩家的辅助监督标签（预测其他 3 人的队伍）。

        Returns: list of 4 np.ndarray, each shape (3,) dtype int64
          aux_targets[p][j] = 相对位置 j+1 处玩家的队伍类别
        类别: 0=SPADE_A3, 1=OPPONENT, 2=SOLO, -1=不可判断(mask)

        mask 逻辑：基于玩家 p 的可观测信息——
          - p 自己的队伍总是已知的（看自己手牌）
          - 对于其他人，如果 observed_teams 为 UNKNOWN → 不可判断 → -1
          - 如果 observed_teams 已暴露（A3/OPPONENT/SOLO）→ 用 actual_teams 监督
          - 特殊情况：p 知道自己是 A3 队，看到另一人也是 A3 → 可推断剩余人是 OPPONENT
            （此时 observed_teams 已经不含 UNKNOWN，由 _compute_observed_teams 处理）
        """
        _CLS = {TEAM_SPADE_A3: 0, TEAM_OPPONENT: 1, TEAM_SOLO: 2}
        s = self.game.state
        actual = s.actual_teams
        observed = s.teams
        targets = []
        for p in range(4):
            t = np.empty(3, dtype=np.int64)
            for j in range(3):
                other = (p + j + 1) % 4
                if observed[other] == TEAM_UNKNOWN:
                    t[j] = -1
                else:
                    t[j] = _CLS.get(actual[other], 1)
            targets.append(t)
        return targets

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
        if action == 'declare':
            # 报牌动作：all-ones 52D（与任何真实出牌不同，是唯一信号）
            return np.ones(ACTION_FEATURE_DIM, dtype=np.int8)
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
        """全部使用 int8 类型，与 DMC buffer 兼容。

        相对位置编码：slot 0=我, 1=下家, 2=对家, 3=上家。
        同一网络权重对所有座位均适用（配合权重共享）。
        """
        parts = []
        n = 4
        # 相对座位顺序：[我, 下家, 对家, 上家]
        rel_order = [(player_id + i) % n for i in range(n)]

        # 1. 自己手牌（52维）
        parts.append(cards_to_feature(raw_state['current_hand']))

        # 2. 上一手出牌（52维，pass/自由出牌=全零）
        parts.append(hand_to_feature(raw_state['last_play']))

        # 3. 上一手出牌者（4维相对位置 one-hot；自由出牌=全零）
        last_play_player_vec = np.zeros(n, dtype=np.int8)
        lpp = raw_state.get('last_play_player', -1)
        if lpp >= 0:
            rel_pos = rel_order.index(lpp)
            last_play_player_vec[rel_pos] = 1
        parts.append(last_play_player_vec)

        # 4. 最近 HISTORY_LEN 步历史（每步 = 玩家4D + 是否有效1D + 牌面52D）
        # 所有玩家共用同一个历史队列，包含 pass 动作
        history = self._action_history[-HISTORY_LEN:]
        for pid, h in history:
            rel_pos = rel_order.index(pid)
            player_vec = np.zeros(n, dtype=np.int8)
            player_vec[rel_pos] = 1
            parts.append(player_vec)
            parts.append(np.array([1], dtype=np.int8))  # valid = 1（真实历史步）
            parts.append(hand_to_feature(h))
        # 填充空历史槽（valid=0，其余全零；可与 pass 步区分）
        empty_step = np.zeros(HISTORY_STEP_DIM, dtype=np.int8)
        for _ in range(HISTORY_LEN - len(history)):
            parts.append(empty_step.copy())

        # 5. 4人已打出的牌（4×52维，相对位置顺序）
        for abs_i in rel_order:
            parts.append(cards_to_feature(self._played_cards[abs_i]))

        # 6. 各玩家剩余手牌数 one-hot（4×14维，相对位置）
        all_hands = raw_state['all_hands']
        for abs_i in rel_order:
            cnt = min(len(all_hands[abs_i]), 13)
            one_hot = np.zeros(14, dtype=np.int8)
            one_hot[cnt] = 1
            parts.append(one_hot)

        # 7. 各玩家队伍 one-hot（4×4维，相对位置）
        # BUG-3 修复：当前玩家(slot 0)使用 actual_teams（自己始终知道自己的队伍）
        # BUG-1/2 修复：observed_teams 由 _compute_observed_teams 正确推断
        obs_teams = raw_state['teams']       # 观测队伍（BUG-1/2已修复）
        actual_teams = raw_state['actual_teams']  # 实际队伍（仅用于自己的slot）
        for i, abs_i in enumerate(rel_order):
            if abs_i == player_id:
                team = actual_teams[abs_i]   # 我知道自己的实际队伍
            else:
                team = obs_teams[abs_i]      # 他人：仅观测信息
            team_vec = np.zeros(len(TEAMS_ORDER), dtype=np.int8)
            if team in TEAMS_ORDER:
                team_vec[TEAMS_ORDER.index(team)] = 1
            parts.append(team_vec)

        # 8. 杂项（6维）
        misc = np.zeros(6, dtype=np.int8)
        misc[0] = 1 if raw_state['last_play'] is None else 0       # 自由出牌
        misc[1] = 1 if raw_state['is_first_turn'] else 0            # 第一手
        misc[2] = min(raw_state.get('pass_count', 0),
                      3)            # pass计数 (0-3)
        # 已完成玩家数 (0-4)
        misc[3] = len(raw_state['rankings'])
        misc[4] = 1 if raw_state.get('is_solo', False) else 0      # 是否独食局
        misc[5] = 1 if raw_state.get(
            'is_declaration_phase', False) else 0  # 是否报牌阶段
        parts.append(misc)

        return np.concatenate(parts)

    def _get_legal_actions(self, raw_state: dict) -> dict:
        """返回 {动作: 特征向量} 的字典。DMC 需要这种格式。"""
        # 报牌阶段：只有 declare 和 pass_declare 两个动作
        if raw_state.get('is_declaration_phase', False):
            return {
                'declare': np.ones(ACTION_FEATURE_DIM, dtype=np.int8),
                'pass': hand_to_feature(None),  # all-zeros，不报
            }

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
        if has_pass or raw_state.get('last_play') is not None:
            result['pass'] = hand_to_feature(None)
            self._key_to_hand['pass'] = None
        return result

    def _decode_action(self, action):
        if action == 'declare':
            return 'declare'
        if action == 'pass' or action is None:
            return None
        if hasattr(action, 'cards'):
            return action
        return self._key_to_hand.get(action, None)

    def get_state(self, player_id: int) -> OrderedDict:
        raw_state = self.game.get_state(player_id)
        return self._extract_state(raw_state)


def _hand_key(hand) -> str:
    if hand is None:
        return 'pass'
    ids = sorted(c.to_id() for c in hand.cards)
    return '|'.join(ids)
