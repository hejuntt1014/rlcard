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
        teams: list[str],          # teams[i] = 玩家i队伍（观测用，初始UNKNOWN）
        actual_teams: list[str],   # actual_teams[i] = 实际队伍（发牌时确定）
        is_solo: bool,             # 是否独食局
        num_players: int = 4,
    ):
        self.hands = hands
        self.current_player = current_player
        self.last_play = last_play
        self.last_play_player = last_play_player
        self.pass_count = pass_count
        self.is_first_turn = is_first_turn
        self.rankings = rankings
        self.teams = teams
        self.actual_teams = actual_teams
        self.is_solo = is_solo
        self.num_players = num_players

    def clone(self) -> 'GameState':
        return GameState(
            hands=[list(h) for h in self.hands],
            current_player=self.current_player,
            last_play=self.last_play,
            last_play_player=self.last_play_player,
            pass_count=self.pass_count,
            is_first_turn=self.is_first_turn,
            rankings=list(self.rankings),
            teams=list(self.teams),
            actual_teams=list(self.actual_teams),
            is_solo=self.is_solo,
            num_players=self.num_players,
        )

    def is_terminal(self) -> bool:
        """游戏结束条件：
        1. 只剩 ≤1 名活跃玩家
        2. 某一队伍的所有成员都已出完
        3. 独食模式：独食者出完 或 其余3人全出完
        """
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

    def get_legal_moves(self) -> list[Optional[Hand]]:
        """返回合法出牌列表；None 表示 pass"""
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

    def apply_move(self, move: Optional[Hand]) -> 'GameState':
        """应用出牌，返回新状态"""
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
                teams=self.teams,
                actual_teams=self.actual_teams,
                is_solo=self.is_solo,
                num_players=self.num_players,
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
            teams=self.teams,
            actual_teams=self.actual_teams,
            is_solo=self.is_solo,
            num_players=self.num_players,
        )

    def _apply_play(self, move: Hand) -> 'GameState':
        played_ids = frozenset(c.to_id() for c in move.cards)
        new_hands = []
        for i, hand in enumerate(self.hands):
            if i == self.current_player:
                new_hands.append([c for c in hand if c.to_id() not in played_ids])
            else:
                new_hands.append(hand)

        # 队伍暴露：打出♠3或♠A → 暴露为黑桃A队
        new_teams = list(self.teams)
        for c in move.cards:
            if (c.suit == 'spade' and c.rank == '3') or (c.suit == 'spade' and c.rank == 'A'):
                if new_teams[self.current_player] == TEAM_UNKNOWN:
                    new_teams[self.current_player] = TEAM_SPADE_A3

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
            rankings=new_rankings, teams=new_teams,
            actual_teams=self.actual_teams, is_solo=self.is_solo,
            num_players=self.num_players,
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
            teams=new_teams,
            actual_teams=self.actual_teams,
            is_solo=self.is_solo,
            num_players=self.num_players,
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
    独食 1v3: 按独食者名次, 范围 [-2, +2]
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
                payoffs[i] = -payoffs[solo_idx]
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
        # 观测用队伍：初始 UNKNOWN，打出♠3/♠A时暴露
        observed_teams = [TEAM_UNKNOWN] * 4

        self.state = GameState(
            hands=hands,
            current_player=start_player,
            last_play=None,
            last_play_player=-1,
            pass_count=0,
            is_first_turn=True,
            rankings=[],
            teams=observed_teams,
            actual_teams=actual_teams,
            is_solo=is_solo,
            num_players=4,
        )
        self._history = []

        return self.get_state(start_player), start_player

    def step(self, action):
        """执行一步动作（action 是 Hand 或 None=pass）"""
        if self.allow_step_back:
            self._history.append(self.state.clone())

        self.state = self.state.apply_move(action)
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
        }

    def get_player_id(self) -> int:
        return self.state.current_player

    def is_over(self) -> bool:
        return self.state.is_terminal()

    def get_payoffs(self) -> list[float]:
        """游戏结束时返回各玩家回报（按照真实 A3 地主规则计分）

        普通2v2: 队伍平均排名分之差 (范围 -3 到 +3)
        独食1v3: 独食者1st→+2, 2nd→+1, 3rd→-1, 4th→-2; 其余反向
        """
        s = self.state
        return compute_payoffs(s.rankings, s.actual_teams, s.is_solo, s.num_players)
