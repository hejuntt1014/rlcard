"""
A3 地主 RLCard 环境接口

支持两种后端:
  - C++ 后端 (a3dizhu_cpp.CppEngine) — 安装扩展后默认使用
  - 纯 Python 后端 (Game) — 自动回退

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
               ※ slot 0（我）使用 actual_teams（自己知道自己的队伍）
               ※ 其他 slot 使用 observed_teams（逐步暴露）
  [844:850]    - 杂项（6 维）：自由出牌、是否第一手、pass 计数、已完成人数、是否独食、是否报牌阶段

总计：850 维

动作用 52 维 one-hot 表示（哪些牌被打出），pass=全零。
报牌阶段特殊编码：
  declare 动作特征 = all-ones (52 维全 1)
  pass_declare 动作特征 = all-zeros (52 维全 0)
"""

from __future__ import annotations
from collections import OrderedDict
import numpy as np

# C++ 后端 (优先)
try:
    from a3dizhu_cpp import CppEngine
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

# Python 后端 (回退)
from rlcard.games.a3dizhu.utils import NUM_CARDS, ACTION_FEATURE_DIM

STATE_DIM = 850

# ─── 纯 Python 实现（原始版本，作为回退和对拍参考） ─────────────────────────

from rlcard.envs import Env
from rlcard.games.a3dizhu import Game
from rlcard.games.a3dizhu.utils import hand_to_feature, cards_to_feature
from rlcard.games.a3dizhu.game import (
    TEAM_SPADE_A3, TEAM_OPPONENT, TEAM_SOLO, TEAM_UNKNOWN,
    compute_step_reward,
)

HISTORY_LEN = 8
HISTORY_STEP_DIM = 4 + 1 + NUM_CARDS  # 57
TEAMS_ORDER = [TEAM_SPADE_A3, TEAM_OPPONENT, TEAM_SOLO, TEAM_UNKNOWN]


def _hand_key(hand) -> str:
    if hand is None:
        return 'pass'
    ids = sorted(c.to_id() for c in hand.cards)
    return '|'.join(ids)


# ═══════════════════════════════════════════════════════════════════════════
#  C++ 加速版环境
# ═══════════════════════════════════════════════════════════════════════════

class A3DizhuCppEnv:
    """A3 地主 RL 环境 — C++ 加速版

    与 A3DizhuEnv 完全兼容（同样的 state_shape / action_shape / 接口），
    但内部用 C++ 引擎运行游戏逻辑、编码状态、枚举合法动作。
    """

    name = 'a3dizhu'

    def __init__(self, config):
        self._config = dict(config)  # save for pickle
        self._engine = CppEngine()
        self.greedy_ratio = config.get('greedy_ratio', 0.0)
        self.random_ratio = config.get('random_ratio', 0.0)
        old_ratio = config.get('rule_opponent_ratio', 0.0)
        if old_ratio > 0 and self.greedy_ratio == 0:
            self.greedy_ratio = old_ratio * 0.67
            self.random_ratio = old_ratio * 0.33
        self._engine.set_greedy_ratio(self.greedy_ratio)
        self._engine.set_random_ratio(self.random_ratio)

        self.num_players = 4
        self.state_shape = [[STATE_DIM] for _ in range(4)]
        self.action_shape = [[ACTION_FEATURE_DIM] for _ in range(4)]
        self.agents = []
        self.timestep = 0
        self.allow_step_back = config.get('allow_step_back', False)
        self.action_recorder = []
        self.np_random = None
        self.num_actions = 1  # DMC uses action features, not IDs
        self.seed(config.get('seed', None))

    def __getstate__(self):
        """Pickle support: CppEngine is not picklable, drop it."""
        state = self.__dict__.copy()
        state.pop('_engine', None)
        state.pop('agents', None)
        return state

    def __setstate__(self, state):
        """Unpickle: recreate CppEngine in the child process."""
        self.__dict__.update(state)
        self._engine = CppEngine()
        self._engine.set_greedy_ratio(self.greedy_ratio)
        self._engine.set_random_ratio(self.random_ratio)
        self.agents = []

    def seed(self, seed_val=None):
        import random
        if seed_val is None:
            seed_val = random.randint(0, 2**31 - 1)
        self._engine.seed(seed_val)

    def set_agents(self, agents):
        self.agents = agents

    def reset(self):
        player_id = self._engine.reset()
        self.timestep = 0
        self.action_recorder = []
        return self._make_state(player_id), player_id

    def step(self, action, raw_action=False):
        self.timestep += 1
        if isinstance(action, str):
            key = action
        elif action is None:
            key = 'pass'
        elif hasattr(action, 'cards'):
            key = _hand_key(action)
        else:
            key = str(action)
        next_player_id = self._engine.step(key)
        return self._make_state(next_player_id), next_player_id

    def run(self, is_training=False):
        """运行一局完整游戏，返回 (trajectories, payoffs, step_rewards)"""
        state, player_id = self.reset()
        trajectories = [[] for _ in range(self.num_players)]
        trajectories[player_id].append(state)

        while not self._engine.is_over():
            if self._engine.is_rule_agent_seat(player_id):
                action_key = self._engine.get_rule_agent_action()
                self._engine.step(action_key)
                trajectories[player_id].append(action_key)
            else:
                if is_training:
                    action = self.agents[player_id].step(state)
                else:
                    action, _ = self.agents[player_id].eval_step(state)
                if isinstance(action, str):
                    key = action
                elif action is None:
                    key = 'pass'
                elif hasattr(action, 'cards'):
                    key = _hand_key(action)
                else:
                    key = str(action)
                self._engine.step(key)
                trajectories[player_id].append(action)

            player_id = self._engine.get_player_id()
            if not self._engine.is_over():
                state = self._make_state(player_id)
                trajectories[player_id].append(state)

        for pid in range(self.num_players):
            trajectories[pid].append(self._make_state(pid))

        payoffs = list(self._engine.get_training_payoffs()
                       if is_training else self._engine.get_payoffs())
        step_rewards = [list(self._engine.get_step_rewards(p))
                        for p in range(self.num_players)]
        return trajectories, payoffs, step_rewards

    def is_over(self):
        return self._engine.is_over()

    def get_player_id(self):
        return self._engine.get_player_id()

    def get_state(self, player_id):
        return self._make_state(player_id)

    def get_payoffs(self):
        return list(self._engine.get_payoffs())

    def get_training_payoffs(self):
        return list(self._engine.get_training_payoffs())

    def get_aux_targets(self):
        return self._engine.get_aux_targets()

    def get_action_feature(self, action) -> np.ndarray:
        if action == 'declare':
            return np.ones(ACTION_FEATURE_DIM, dtype=np.int8)
        if action is None or action == 'pass':
            return np.zeros(ACTION_FEATURE_DIM, dtype=np.int8)
        if isinstance(action, str):
            return np.array(self._engine.get_action_feature(action), dtype=np.int8)
        if hasattr(action, 'cards'):
            return hand_to_feature(action)
        return np.zeros(ACTION_FEATURE_DIM, dtype=np.int8)

    def _make_state(self, player_id):
        obs = self._engine.encode_obs(player_id)
        legal_actions = self._engine.get_legal_actions()
        return OrderedDict({
            'obs': obs,
            'legal_actions': legal_actions,
        })


# ═══════════════════════════════════════════════════════════════════════════
#  纯 Python 版环境（原始版本）
# ═══════════════════════════════════════════════════════════════════════════

class A3DizhuPyEnv(Env):
    """A3 地主 RL 环境 — 纯 Python 版（原始实现，用于对拍和回退）"""

    name = 'a3dizhu'

    def __init__(self, config):
        self.game = Game()
        self.greedy_ratio = config.get('greedy_ratio', 0.0)
        self.random_ratio = config.get('random_ratio', 0.0)
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
        if not raw_action:
            action = self._decode_action(action)
        self.timestep += 1
        player_id = self.game.get_player_id()
        prev_state = self.game.state
        if not self.game.state.is_declaration_phase:
            self.action_recorder.append((player_id, action))
            self._action_history.append((player_id, action))
        if action is not None and action != 'declare' and hasattr(action, 'cards'):
            self._played_cards[player_id].extend(action.cards)
        next_state, next_player_id = self.game.step(action)
        sr = compute_step_reward(prev_state, action, self.game.state, player_id)
        self._step_rewards[player_id].append(sr)
        return self._extract_state(next_state), next_player_id

    def run(self, is_training=False):
        trajectories = [[] for _ in range(self.num_players)]
        state, player_id = self.reset()
        trajectories[player_id].append(state)
        while not self.is_over():
            if player_id in self._seat_agents:
                raw_state = state.get('raw_obs', state)
                action = self._seat_agents[player_id].step(raw_state)
                next_state, next_player_id = self.step(action, raw_action=True)
                trajectories[player_id].append(action if isinstance(action, str) else _hand_key(action))
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
        payoffs = self.get_training_payoffs() if is_training else self.get_payoffs()
        step_rewards = [list(sr) for sr in self._step_rewards]
        return trajectories, payoffs, step_rewards

    def get_payoffs(self):
        return self.game.get_payoffs()

    def get_training_payoffs(self):
        from rlcard.games.a3dizhu.game import compute_training_payoffs, compute_declared_payoffs
        s = self.game.state
        if s.is_declared and s.declarant >= 0:
            return compute_declared_payoffs(s.rankings, s.declarant, s.num_players)
        return compute_training_payoffs(s.rankings, s.actual_teams, s.is_solo, s.num_players)

    def get_aux_targets(self):
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
            return np.ones(ACTION_FEATURE_DIM, dtype=np.int8)
        if action is None or action == 'pass':
            return hand_to_feature(None)
        if isinstance(action, str):
            hand = self._key_to_hand.get(action, None)
            return hand_to_feature(hand)
        return hand_to_feature(action)

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
        parts = []
        n = 4
        rel_order = [(player_id + i) % n for i in range(n)]
        parts.append(cards_to_feature(raw_state['current_hand']))
        parts.append(hand_to_feature(raw_state['last_play']))
        last_play_player_vec = np.zeros(n, dtype=np.int8)
        lpp = raw_state.get('last_play_player', -1)
        if lpp >= 0:
            rel_pos = rel_order.index(lpp)
            last_play_player_vec[rel_pos] = 1
        parts.append(last_play_player_vec)
        history = self._action_history[-HISTORY_LEN:]
        for pid, h in history:
            rel_pos = rel_order.index(pid)
            player_vec = np.zeros(n, dtype=np.int8)
            player_vec[rel_pos] = 1
            parts.append(player_vec)
            parts.append(np.array([1], dtype=np.int8))
            parts.append(hand_to_feature(h))
        empty_step = np.zeros(HISTORY_STEP_DIM, dtype=np.int8)
        for _ in range(HISTORY_LEN - len(history)):
            parts.append(empty_step.copy())
        for abs_i in rel_order:
            parts.append(cards_to_feature(self._played_cards[abs_i]))
        all_hands = raw_state['all_hands']
        for abs_i in rel_order:
            cnt = min(len(all_hands[abs_i]), 13)
            one_hot = np.zeros(14, dtype=np.int8)
            one_hot[cnt] = 1
            parts.append(one_hot)
        obs_teams = raw_state['teams']
        actual_teams = raw_state['actual_teams']
        for i, abs_i in enumerate(rel_order):
            if abs_i == player_id:
                team = actual_teams[abs_i]
            else:
                team = obs_teams[abs_i]
            team_vec = np.zeros(len(TEAMS_ORDER), dtype=np.int8)
            if team in TEAMS_ORDER:
                team_vec[TEAMS_ORDER.index(team)] = 1
            parts.append(team_vec)
        misc = np.zeros(6, dtype=np.int8)
        misc[0] = 1 if raw_state['last_play'] is None else 0
        misc[1] = 1 if raw_state['is_first_turn'] else 0
        misc[2] = min(raw_state.get('pass_count', 0), 3)
        misc[3] = len(raw_state['rankings'])
        misc[4] = 1 if raw_state.get('is_solo', False) else 0
        misc[5] = 1 if raw_state.get('is_declaration_phase', False) else 0
        parts.append(misc)
        return np.concatenate(parts)

    def _get_legal_actions(self, raw_state: dict) -> dict:
        if raw_state.get('is_declaration_phase', False):
            return {
                'declare': np.ones(ACTION_FEATURE_DIM, dtype=np.int8),
                'pass': hand_to_feature(None),
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
        if has_pass:
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


# ═══════════════════════════════════════════════════════════════════════════
#  统一入口：自动选择最快的后端
# ═══════════════════════════════════════════════════════════════════════════

def A3DizhuEnv(config):
    """Create an A3 environment with an optional native backend."""
    backend = config.get('backend', 'auto')
    if backend not in ('auto', 'python', 'cpp'):
        raise ValueError('backend must be auto, python or cpp')
    if backend == 'cpp' and not _HAS_CPP:
        raise ImportError('Build a3dizhu_cpp with python setup_cpp.py build_ext --inplace')
    if _HAS_CPP and backend != 'python':
        return A3DizhuCppEnv(config)
    return A3DizhuPyEnv(config)
