"""
A3 地主游戏核心逻辑

规则摘要：
- 4名玩家，52张牌（无大小王），每人13张
- 第一手必须打出包含方块4(diamond_4)的组合
- 黑桃A队 vs 对手队，独食玩家单独一方
- 出牌类型：单/对/三/顺子/同花/三带对/四带一/同花顺
- 同尺寸跟牌，5张牌型之间可互压（同花顺>四带一>三带对>同花>顺子）
- 玩家出完即完成，最先出完第1名，最后1名末位
- 回合：当其余活跃玩家全部pass时新开一轮，上一个出牌者自由出牌
"""

from __future__ import annotations
import random
from typing import Optional
from .card import Card, make_deck, SUIT_VALUE, RANK_VALUE
from .hand import Hand, detect_hand, can_beat
from .hint import get_playable_hands, has_playable_hand

# 队伍常量
TEAM_SPADE_A3  = 'spade_a3'   # 黑桃A队
TEAM_OPPONENT  = 'opponent'    # 对手队
TEAM_SOLO      = 'solo'        # 独食
TEAM_UNKNOWN   = 'unknown'     # 未知


def assign_teams(hands: list[list[Card]]) -> tuple[list[str], bool]:
    """根据发牌结果分配队伍。返回 (teams, is_solo)
    - 持有♠3的人和持有♠A的人是一队(spade_a3)
    - 如果同一人持有♠3和♠A → 独食(solo)，其余3人是 opponent
    - 其余2人是 opponent
    """
    spade3_owner = -1
    spadeA_owner = -1
    for i, hand in enumerate(hands):
        for c in hand:
            if c.suit == 'spade' and c.rank == '3':
                spade3_owner = i
            if c.suit == 'spade' and c.rank == 'A':
                spadeA_owner = i

    teams = [TEAM_OPPONENT] * len(hands)
    is_solo = False

    if spade3_owner == spadeA_owner and spade3_owner >= 0:
        # 独食：一人持有♠3和♠A
        teams[spade3_owner] = TEAM_SOLO
        is_solo = True
    else:
        if spade3_owner >= 0:
            teams[spade3_owner] = TEAM_SPADE_A3
        if spadeA_owner >= 0:
            teams[spadeA_owner] = TEAM_SPADE_A3

    return teams, is_solo


class GameState:
    """轻量级游戏状态（用于 RL 环境和 MCTS 等）"""

    def __init__(
        self,
        hands: list[list[Card]],   # hands[i] = 玩家i的手牌
        current_player: int,
        last_play: Optional[Hand],
        last_play_player: int,     # -1 表示自由出牌轮
        pass_count: int,
        is_first_turn: bool,
        rankings: list[int],       # 已完成的玩家索引（按完成顺序）
        actual_teams: list[str],   # actual_teams[i] = 实际队伍（发牌时确定）
        is_solo: bool,             # 是否独食局
        num_players: int = 4,
        spade3_player: int = -1,   # 打出♠3的玩家索引（-1=未打出）
        spadeA_player: int = -1,   # 打出♠A的玩家索引（-1=未打出）
        # ── 报牌阶段字段 ──────────────────────────────────────────────────
        is_declaration_phase: bool = False,  # 是否在报牌阶段
        declaration_turn: int = 0,           # 当前轮到谁报牌（顺序决策）
        declaration_passes: int = 0,         # 已 pass 的人数
        is_declared: bool = False,           # 是否有人报牌
        declarant: int = -1,                 # 报牌者索引（-1=无人报牌）
    ):
        self.hands = hands
        self.current_player = current_player
        self.last_play = last_play
        self.last_play_player = last_play_player
        self.pass_count = pass_count
        self.is_first_turn = is_first_turn
        self.rankings = rankings
        self.actual_teams = actual_teams
        self.is_solo = is_solo
        self.num_players = num_players
        self.spade3_player = spade3_player
        self.spadeA_player = spadeA_player
        # ── 报牌阶段 ─────────────────────────────────────────────────────
        self.is_declaration_phase = is_declaration_phase
        self.declaration_turn = declaration_turn
        self.declaration_passes = declaration_passes
        self.is_declared = is_declared
        self.declarant = declarant
        # observed teams 从♠牌出牌记录动态计算，确保逻辑唯一真相
        self.teams = self._compute_observed_teams()

    def _compute_observed_teams(self) -> list[str]:
        """根据♠3/♠A的出牌历史推断各玩家的可观测队伍。

        与 TS BotContextHelpers.buildPlayedCardsAndTeams() 完全一致：
        - 同一人打出♠3和♠A → SOLO，其余全是 OPPONENT
        - 不同人各打一张   → 两人 SPADE_A3，其余 OPPONENT
        - 仅打出♠3 或 ♠A  → 该人 SPADE_A3，其余 UNKNOWN
        - 都未打出         → 全部 UNKNOWN
        """
        s3, sA = self.spade3_player, self.spadeA_player
        if s3 >= 0 and sA >= 0:
            if s3 == sA:
                # 独食：同一人持有并打出了♠3和♠A
                teams = [TEAM_OPPONENT] * self.num_players
                teams[s3] = TEAM_SOLO
            else:
                # 两人分别亮牌 → 全局队伍已知，其余必为 OPPONENT
                teams = [TEAM_OPPONENT] * self.num_players
                teams[s3] = TEAM_SPADE_A3
                teams[sA] = TEAM_SPADE_A3
        elif s3 >= 0:
            teams = [TEAM_UNKNOWN] * self.num_players
            teams[s3] = TEAM_SPADE_A3
        elif sA >= 0:
            teams = [TEAM_UNKNOWN] * self.num_players
            teams[sA] = TEAM_SPADE_A3
        else:
            teams = [TEAM_UNKNOWN] * self.num_players
        return teams

    def clone(self) -> 'GameState':
        return GameState(
            hands=[list(h) for h in self.hands],
            current_player=self.current_player,
            last_play=self.last_play,
            last_play_player=self.last_play_player,
            pass_count=self.pass_count,
            is_first_turn=self.is_first_turn,
            rankings=list(self.rankings),
            actual_teams=list(self.actual_teams),
            is_solo=self.is_solo,
            num_players=self.num_players,
            spade3_player=self.spade3_player,
            spadeA_player=self.spadeA_player,
            is_declaration_phase=self.is_declaration_phase,
            declaration_turn=self.declaration_turn,
            declaration_passes=self.declaration_passes,
            is_declared=self.is_declared,
            declarant=self.declarant,
        )

    def is_terminal(self) -> bool:
        """游戏结束条件：
        0. 报牌阶段永远不是终局
        1. 报牌局：任何人出完即结束（只判断第1名是否为报牌者）
        2. 只剩 ≤1 名活跃玩家
        3. 某一队伍的所有成员都已出完
        4. 独食模式：独食者出完 或 其余3人全出完
        """
        if self.is_declaration_phase:
            return False

        # 报牌局：第一个出完牌的人决定输赢，立即结束
        if self.is_declared and self.declarant >= 0:
            return len(self.rankings) >= 1

        if len(self.rankings) >= self.num_players - 1:
            return True

        finished = set(self.rankings)

        if self.is_solo:
            solo_idx = next((i for i, t in enumerate(self.actual_teams) if t == TEAM_SOLO), -1)
            if solo_idx >= 0:
                if solo_idx in finished:
                    return True
                opponents = [i for i, t in enumerate(self.actual_teams) if t != TEAM_SOLO]
                if all(i in finished for i in opponents):
                    return True

        # 检查是否有一整队全部出完
        from collections import defaultdict
        team_members: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(self.actual_teams):
            if t != TEAM_UNKNOWN:
                team_members[t].append(i)

        for team, members in team_members.items():
            if team == TEAM_SOLO:
                continue
            if all(m in finished for m in members) and len(members) > 0:
                return True

        return False

    def get_legal_moves(self) -> list:
        """返回合法动作列表。
        报牌阶段：返回 ['declare', None]（None=不报）
        出牌阶段：返回 Hand 列表，可能包含 None（pass）
        """
        if self.is_declaration_phase:
            return ['declare', None]  # 'declare'=报牌, None=不报

        my_cards = self.hands[self.current_player]
        if not my_cards:
            return []

        if self.last_play is None:
            # 自由出牌，不能 pass
            hands = get_playable_hands(my_cards, None)
            if self.is_first_turn:
                hands = [h for h in hands
                         if any(c.suit == 'diamond' and c.rank == '4' for c in h.cards)]
            return hands
        else:
            beating = get_playable_hands(my_cards, self.last_play)
            # 只剩1张牌且能打出 → 必须出，不能 pass
            if len(my_cards) == 1 and beating:
                return beating
            result: list[Optional[Hand]] = list(beating)
            result.append(None)  # 可以 pass
            return result

    def apply_declaration(self, declare: bool) -> 'GameState':
        """处理报牌阶段的单步决策。

        declare=True: 报牌，当前玩家成为 SOLO，报牌阶段结束
        declare=False: 不报，轮到下一人；全部 pass 后进入出牌阶段
        """
        p = self.declaration_turn

        if declare:
            # 报牌者成为独食，覆盖原有队伍
            new_actual_teams = [TEAM_OPPONENT] * self.num_players
            new_actual_teams[p] = TEAM_SOLO
            return GameState(
                hands=self.hands,
                current_player=self.current_player,
                last_play=None,
                last_play_player=-1,
                pass_count=0,
                is_first_turn=True,
                rankings=[],
                actual_teams=new_actual_teams,
                is_solo=True,
                num_players=self.num_players,
                spade3_player=-1,
                spadeA_player=-1,
                is_declaration_phase=False,
                declaration_turn=0,
                declaration_passes=0,
                is_declared=True,
                declarant=p,
            )
        else:
            # 不报，轮到下一人
            new_passes = self.declaration_passes + 1
            next_turn = (p + 1) % self.num_players
            if new_passes >= self.num_players:
                # 全员 pass → 进入正常出牌阶段
                return GameState(
                    hands=self.hands,
                    current_player=self.current_player,
                    last_play=None,
                    last_play_player=-1,
                    pass_count=0,
                    is_first_turn=True,
                    rankings=[],
                    actual_teams=list(self.actual_teams),
                    is_solo=self.is_solo,
                    num_players=self.num_players,
                    spade3_player=self.spade3_player,
                    spadeA_player=self.spadeA_player,
                    is_declaration_phase=False,
                    declaration_turn=0,
                    declaration_passes=new_passes,
                    is_declared=False,
                    declarant=-1,
                )
            return GameState(
                hands=self.hands,
                current_player=self.current_player,
                last_play=None,
                last_play_player=-1,
                pass_count=0,
                is_first_turn=True,
                rankings=[],
                actual_teams=list(self.actual_teams),
                is_solo=self.is_solo,
                num_players=self.num_players,
                spade3_player=self.spade3_player,
                spadeA_player=self.spadeA_player,
                is_declaration_phase=True,
                declaration_turn=next_turn,
                declaration_passes=new_passes,
                is_declared=False,
                declarant=-1,
            )

    def apply_move(self, move) -> 'GameState':
        """应用动作，返回新状态。
        报牌阶段: move='declare'(报牌) 或 None(不报)
        出牌阶段: move=Hand对象(出牌) 或 None(pass)
        """
        if self.is_declaration_phase:
            return self.apply_declaration(move == 'declare')
        if move is None:
            return self._apply_pass()
        return self._apply_play(move)

    def _apply_pass(self) -> 'GameState':
        finished = set(self.rankings)
        active_count = self.num_players - len(self.rankings)
        new_pass_count = self.pass_count + 1

        last_still_active = (
            self.last_play_player >= 0
            and self.last_play_player not in finished
        )
        passes_needed = active_count - 1 if last_still_active else active_count

        if new_pass_count >= passes_needed:
            # 新轮次
            if last_still_active:
                new_leader = self.last_play_player
            else:
                new_leader = self._next_active(
                    self.last_play_player if self.last_play_player >= 0 else self.current_player,
                    finished
                )
            return GameState(
                hands=self.hands,
                current_player=new_leader,
                last_play=None,
                last_play_player=-1,
                pass_count=0,
                is_first_turn=False,
                rankings=list(self.rankings),
                actual_teams=self.actual_teams,
                is_solo=self.is_solo,
                num_players=self.num_players,
                spade3_player=self.spade3_player,
                spadeA_player=self.spadeA_player,
                is_declared=self.is_declared,
                declarant=self.declarant,
            )

        next_player = self._next_active(self.current_player, finished)
        return GameState(
            hands=self.hands,
            current_player=next_player,
            last_play=self.last_play,
            last_play_player=self.last_play_player,
            pass_count=new_pass_count,
            is_first_turn=False,
            rankings=list(self.rankings),
            actual_teams=self.actual_teams,
            is_solo=self.is_solo,
            num_players=self.num_players,
            spade3_player=self.spade3_player,
            spadeA_player=self.spadeA_player,
            is_declared=self.is_declared,
            declarant=self.declarant,
        )

    def _apply_play(self, move: Hand) -> 'GameState':
        played_ids = frozenset(c.to_id() for c in move.cards)
        new_hands = []
        for i, hand in enumerate(self.hands):
            if i == self.current_player:
                new_hands.append([c for c in hand if c.to_id() not in played_ids])
            else:
                new_hands.append(hand)

        # 追踪♠3/♠A出牌（正确处理 SOLO：同一人打出两张才标记为独食）
        new_spade3 = self.spade3_player
        new_spadeA = self.spadeA_player
        for c in move.cards:
            if c.suit == 'spade' and c.rank == '3':
                new_spade3 = self.current_player
            if c.suit == 'spade' and c.rank == 'A':
                new_spadeA = self.current_player

        new_rankings = list(self.rankings)
        if len(new_hands[self.current_player]) == 0:
            new_rankings.append(self.current_player)

        # 将未出完的玩家按手牌数量多→少的顺序补入排名（与TS端一致）
        finished_set = set(new_rankings)
        remaining = [i for i in range(self.num_players) if i not in finished_set]

        # 创建临时状态检查是否游戏结束（队伍全出完等）
        tmp = GameState(
            hands=new_hands, current_player=self.current_player,
            last_play=move, last_play_player=self.current_player,
            pass_count=0, is_first_turn=False,
            rankings=new_rankings,
            actual_teams=self.actual_teams, is_solo=self.is_solo,
            num_players=self.num_players,
            spade3_player=new_spade3, spadeA_player=new_spadeA,
            is_declared=self.is_declared, declarant=self.declarant,
        )
        if tmp.is_terminal() and remaining:
            remaining.sort(key=lambda i: len(new_hands[i]), reverse=True)
            for i in remaining:
                new_rankings.append(i)

        new_finished = set(new_rankings)
        next_player = self._next_active(self.current_player, new_finished)

        return GameState(
            hands=new_hands,
            current_player=next_player,
            last_play=move,
            last_play_player=self.current_player,
            pass_count=0,
            is_first_turn=False,
            rankings=new_rankings,
            actual_teams=self.actual_teams,
            is_solo=self.is_solo,
            num_players=self.num_players,
            spade3_player=new_spade3,
            spadeA_player=new_spadeA,
            is_declared=self.is_declared,
            declarant=self.declarant,
        )

    def _next_active(self, from_player: int, finished: set[int]) -> int:
        for offset in range(1, self.num_players + 1):
            p = (from_player + offset) % self.num_players
            if p not in finished:
                return p
        return from_player

    def get_score(self, player_idx: int) -> float:
        """返回 player_idx 的得分，用于 MCTS 等（与 get_payoffs 一致，归一化到 [0,1]）"""
        payoffs = compute_payoffs(self.rankings, self.actual_teams, self.is_solo, self.num_players)
        # 归一化到 [0,1]：最大回报 +2，最小 -2
        return (payoffs[player_idx] + 2.0) / 4.0


def compute_payoffs(rankings: list[int], actual_teams: list[str],
                     is_solo: bool, num_players: int = 4) -> list[float]:
    """按照真实 A3 地主规则计算各玩家回报（评估用，纯规则分数）。

    普通 2v2: 队伍平均排名分之差, 范围 [-2, +2]
    独食 1v3: 独食者 [-2, +2]，每个对手 [-2/3, +2/3]，3:1 零和
              (规则: 1st→+6/-2, 2nd→+3/-1, 3rd→-3/+1, 4th→-6/+2)
    """
    rank_points = {0: 3, 1: 2, 2: 1, 3: 0}
    payoffs = [0.0] * num_players

    if is_solo:
        solo_idx = next((i for i, t in enumerate(actual_teams) if t == TEAM_SOLO), -1)
        if solo_idx < 0:
            return payoffs

        solo_rank = rankings.index(solo_idx) if solo_idx in rankings else num_players - 1
        solo_payoff_table = {0: 2.0, 1: 1.0, 2: -1.0, 3: -2.0}
        payoffs[solo_idx] = solo_payoff_table.get(solo_rank, -2.0)
        for i in range(num_players):
            if i != solo_idx:
                payoffs[i] = -payoffs[solo_idx] / 3.0
    else:
        team_a_members = [i for i, t in enumerate(actual_teams) if t == TEAM_SPADE_A3]
        team_b_members = [i for i, t in enumerate(actual_teams) if t == TEAM_OPPONENT]

        def team_avg(members):
            if not members:
                return 0.0
            total = 0
            for m in members:
                rank = rankings.index(m) if m in rankings else num_players - 1
                total += rank_points.get(rank, 0)
            return total / len(members)

        diff = team_avg(team_a_members) - team_avg(team_b_members)
        for i in team_a_members:
            payoffs[i] = diff
        for i in team_b_members:
            payoffs[i] = -diff

    return payoffs


# ─── 报牌结算 ──────────────────────────────────────────────────────────────

def compute_declared_payoffs(rankings: list[int], declarant: int,
                              num_players: int = 4) -> list[float]:
    """报牌结算（零和，3:1 比例）。

    真实规则：报牌赢 → 其余三人各输 4分；报牌输 → 报牌者各输 4分。
    普通局最大 ±2分/人，报牌 = 2× 普通局赌注。

    归一化后：报牌者 ±4.0，每个对手 ∓4/3 ≈ ∓1.333（零和）。
    对比普通局 ±2.0 / 独食 ±2.0，报牌回报 2× 放大，
    使 AI 正确学习报牌的高风险高回报特征。
    """
    payoffs = [0.0] * num_players
    declared_rank = rankings.index(declarant) if declarant in rankings else num_players - 1
    if declared_rank == 0:
        payoffs[declarant] = 4.0
        for i in range(num_players):
            if i != declarant:
                payoffs[i] = -4.0 / 3.0
    else:
        payoffs[declarant] = -4.0
        for i in range(num_players):
            if i != declarant:
                payoffs[i] = 4.0 / 3.0
    return payoffs


# ─── 训练用奖励塑形 ────────────────────────────────────────────────────────

_RANK_BONUS = [0.3, 0.1, -0.1, -0.3]

def compute_training_payoffs(rankings: list[int], actual_teams: list[str],
                              is_solo: bool, num_players: int = 4) -> list[float]:
    """带奖励塑形的训练回报。在基础回报上叠加：

    1. 个人排名奖励: 1st→+0.3, 2nd→+0.1, 3rd→-0.1, 4th→-0.3
       让 AI 有个人出完牌的动力，而非只依赖终局队伍分
    """
    payoffs = compute_payoffs(rankings, actual_teams, is_solo, num_players)

    for i in range(num_players):
        rank = rankings.index(i) if i in rankings else num_players - 1
        payoffs[i] += _RANK_BONUS[min(rank, 3)]

    return payoffs


# ─── 中间奖励：队友合作 ───────────────────────────────────────────────────

def get_teammate_confidence(
    player_id: int,
    actual_teams: list[str],
    observed_teams: list[str],
    num_players: int = 4,
) -> tuple[Optional[int], float]:
    """基于 AI 可观测信息判断能否识别队友。

    返回 (actual_teammate_idx, confidence)
      1.0 = 完全确认队友身份
      0.5 = 缩小到二选一
      0.0 = 无法判断 / 无队友（独食）
    """
    my_team = actual_teams[player_id]

    if my_team == TEAM_SOLO:
        return None, 0.0

    # 找到实际队友 (ground truth)
    actual_teammate = None
    for i in range(num_players):
        if i != player_id and actual_teams[i] == my_team:
            actual_teammate = i
            break
    if actual_teammate is None:
        return None, 0.0

    if my_team == TEAM_SPADE_A3:
        if observed_teams[actual_teammate] == TEAM_SPADE_A3:
            return actual_teammate, 1.0
        return actual_teammate, 0.0

    # OPPONENT 队
    revealed = [
        i for i in range(num_players)
        if i != player_id
        and observed_teams[i] in (TEAM_SPADE_A3, TEAM_SOLO)
    ]
    if len(revealed) >= 2:
        return actual_teammate, 1.0
    if len(revealed) == 1:
        return actual_teammate, 0.5
    return actual_teammate, 0.0


# 保守值：确保一局累积 ≈ ±0.06，仅占终局回报的 ~3%
_REWARD_LET_TEAMMATE    =  0.015   # 让队友的牌站住 (pass)
_REWARD_BEAT_TEAMMATE   = -0.01    # 压了队友的牌 (轻微)
_REWARD_FEED_TEAMMATE   =  0.03    # 出完后队友接风

def compute_step_reward(
    prev_state: 'GameState',
    action,
    new_state: 'GameState',
    player_id: int,
) -> float:
    """计算单步合作中间奖励。

    仅在出牌阶段 + 非独食 + 能识别队友时生效。
    用 actual_teams 判断行为是否合作，用 observed_teams 的置信度缩放。
    """
    if prev_state.is_declaration_phase:
        return 0.0

    teammate, confidence = get_teammate_confidence(
        player_id,
        prev_state.actual_teams,
        prev_state.teams,
        prev_state.num_players,
    )
    if confidence <= 0.0 or teammate is None:
        return 0.0

    reward = 0.0
    last_player = prev_state.last_play_player

    # 行为 1：让牌——pass 让队友的牌站住
    if action is None and last_player == teammate:
        reward += _REWARD_LET_TEAMMATE

    # 行为 2：压队友——出牌压了队友刚打出的牌
    if action is not None and action != 'declare' and last_player == teammate:
        reward += _REWARD_BEAT_TEAMMATE

    # 行为 3：接风——出完牌后下一个出牌的人是队友
    if (action is not None
            and action != 'declare'
            and hasattr(action, 'cards')
            and len(new_state.hands[player_id]) == 0
            and new_state.current_player == teammate):
        reward += _REWARD_FEED_TEAMMATE

    return reward * confidence


class Game:
    """A3 地主游戏引擎（RLCard Game 接口）"""

    def __init__(self):
        self.np_random = random.Random()
        self.allow_step_back = False
        self.state: Optional[GameState] = None
        self._history: list[GameState] = []

    def get_num_players(self) -> int:
        return 4

    def get_num_actions(self) -> int:
        # 动作空间大小，见 utils.py
        from .utils import NUM_ACTIONS
        return NUM_ACTIONS

    def init_game(self):
        """洗牌发牌，初始化游戏状态"""
        deck = make_deck()
        self.np_random.shuffle(deck)

        hands = [deck[i*13:(i+1)*13] for i in range(4)]

        # 找到持有 diamond_4 的玩家作为起始玩家
        start_player = 0
        for i, hand in enumerate(hands):
            if any(c.suit == 'diamond' and c.rank == '4' for c in hand):
                start_player = i
                break

        # 根据♠3/♠A分配实际队伍（用于计分）
        actual_teams, is_solo = assign_teams(hands)
        # observed_teams 由 GameState._compute_observed_teams() 动态计算

        self.state = GameState(
            hands=hands,
            current_player=start_player,
            last_play=None,
            last_play_player=-1,
            pass_count=0,
            is_first_turn=True,
            rankings=[],
            actual_teams=actual_teams,
            is_solo=is_solo,
            num_players=4,
            spade3_player=-1,
            spadeA_player=-1,
            # 报牌阶段：从座位0开始依次决策
            is_declaration_phase=True,
            declaration_turn=0,
            declaration_passes=0,
            is_declared=False,
            declarant=-1,
        )
        self._history = []

        return self.get_state(start_player), start_player

    def step(self, action):
        """执行一步动作。
        报牌阶段: action='declare' 或 None(不报)
        出牌阶段: action=Hand 或 None(pass)
        """
        if self.allow_step_back:
            self._history.append(self.state.clone())

        self.state = self.state.apply_move(action)
        # 报牌阶段：current_player 是报牌顺序的当前人
        if self.state.is_declaration_phase:
            player_id = self.state.declaration_turn
        else:
            player_id = self.state.current_player
        return self.get_state(player_id), player_id

    def step_back(self) -> bool:
        if not self._history:
            return False
        self.state = self._history.pop()
        return True

    def get_state(self, player_id: int) -> dict:
        """返回指定玩家的可观测状态"""
        s = self.state
        return {
            'current_hand': list(s.hands[player_id]),
            'other_hands_count': [len(s.hands[i]) for i in range(4) if i != player_id],
            'all_hands': s.hands,
            'current_player': s.current_player,
            'last_play': s.last_play,
            'last_play_player': s.last_play_player,
            'pass_count': s.pass_count,
            'is_first_turn': s.is_first_turn,
            'rankings': list(s.rankings),
            'teams': list(s.teams),           # 观测用（逐步暴露）
            'actual_teams': list(s.actual_teams),  # 实际队伍（训练用）
            'is_solo': s.is_solo,
            'player_id': player_id,
            'legal_actions': s.get_legal_moves(),
            # 报牌阶段字段
            'is_declaration_phase': s.is_declaration_phase,
            'declaration_turn': s.declaration_turn,
            'is_declared': s.is_declared,
            'declarant': s.declarant,
        }

    def get_player_id(self) -> int:
        if self.state.is_declaration_phase:
            return self.state.declaration_turn
        return self.state.current_player

    def is_over(self) -> bool:
        return self.state.is_terminal()

    def get_payoffs(self) -> list[float]:
        """游戏结束时返回各玩家回报（按照真实 A3 地主规则计分）

        报牌局: 按 compute_declared_payoffs 结算
        普通2v2: 队伍平均排名分之差 (范围 -3 到 +3)
        独食1v3: 独食者1st→+2, 2nd→+1, 3rd→-1, 4th→-2; 其余反向
        """
        s = self.state
        if s.is_declared and s.declarant >= 0:
            return compute_declared_payoffs(s.rankings, s.declarant, s.num_players)
        return compute_payoffs(s.rankings, s.actual_teams, s.is_solo, s.num_players)
