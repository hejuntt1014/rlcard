"""
A3 地主牌型检测与比较

牌型（对应 TypeScript HandType）:
  single       - 单张 (1张)
  pair         - 对子 (2张同点)
  triple       - 三条 (3张同点)
  straight     - 顺子 (5张连续，不含2)
  flush        - 同花 (5张同花，非顺)
  full_house   - 三带对 (3+2)
  four_with_one- 四带一 (4+1)
  straight_flush- 同花顺 (5张同花连续)

五张牌型优先级: straight_flush > four_with_one > full_house > flush > straight
"""

from __future__ import annotations
from itertools import combinations as _combinations
from typing import Optional
from .card import Card, RANK_VALUE, SUIT_VALUE, STRAIGHT_RANK_VALUE

# 牌型常量
SINGLE        = 'single'
PAIR          = 'pair'
TRIPLE        = 'triple'
STRAIGHT      = 'straight'
FLUSH         = 'flush'
FULL_HOUSE    = 'full_house'
FOUR_WITH_ONE = 'four_with_one'
STRAIGHT_FLUSH= 'straight_flush'

# 五张牌型优先级（越大越强）
FIVE_CARD_PRIORITY = {
    STRAIGHT:      0,
    FLUSH:         1,
    FULL_HOUSE:    2,
    FOUR_WITH_ONE: 3,
    STRAIGHT_FLUSH:4,
}


class Hand:
    """表示一手牌及其类型"""
    __slots__ = ('type', 'cards', 'primary_card', 'size')

    def __init__(self, hand_type: str, cards: list[Card], primary_card: Card):
        self.type = hand_type
        self.cards = cards
        self.primary_card = primary_card
        self.size = len(cards)

    def card_ids(self) -> frozenset[str]:
        return frozenset(c.to_id() for c in self.cards)

    def __repr__(self):
        return f'Hand({self.type}, {self.cards})'


# ─── 牌型检测 ─────────────────────────────────────────────────────────────────

def detect_hand(cards: list[Card]) -> Optional[Hand]:
    """检测给定的牌是否构成合法牌型，返回 Hand 或 None"""
    n = len(cards)
    if n == 1: return _detect_single(cards)
    if n == 2: return _detect_pair(cards)
    if n == 3: return _detect_triple(cards)
    if n == 5: return _detect_five(cards)
    return None


def _detect_single(cards: list[Card]) -> Hand:
    return Hand(SINGLE, cards, cards[0])


def _detect_pair(cards: list[Card]) -> Optional[Hand]:
    a, b = cards[0], cards[1]
    if a.rank != b.rank:
        return None
    primary = a if a.score() >= b.score() else b
    return Hand(PAIR, cards, primary)


def _detect_triple(cards: list[Card]) -> Optional[Hand]:
    if not (cards[0].rank == cards[1].rank == cards[2].rank):
        return None
    primary = max(cards, key=lambda c: c.score())
    return Hand(TRIPLE, cards, primary)


def _detect_five(cards: list[Card]) -> Optional[Hand]:
    sf = _detect_straight_flush(cards)
    if sf: return sf
    fwo = _detect_four_with_one(cards)
    if fwo: return fwo
    fh = _detect_full_house(cards)
    if fh: return fh
    fl = _detect_flush(cards)
    if fl: return fl
    st = _detect_straight(cards)
    if st: return st
    return None


def _detect_straight(cards: list[Card]) -> Optional[Hand]:
    """顺子：5张连续点数，不含2（3可用，在顺子中是最小牌）"""
    sorted_cards = sorted(cards, key=lambda c: STRAIGHT_RANK_VALUE.get(c.rank, 0))
    # 含2时不合法
    if any(c.rank == '2' for c in sorted_cards):
        return None
    ranks = [STRAIGHT_RANK_VALUE[c.rank] for c in sorted_cards]
    # 检查是否有重复点数
    if len(set(ranks)) != 5:
        return None
    # 检查是否连续
    if ranks[-1] - ranks[0] != 4:
        return None
    # 主牌：最高点数中花色最大的
    max_rank_val = ranks[-1]
    top_cards = [c for c in sorted_cards if STRAIGHT_RANK_VALUE[c.rank] == max_rank_val]
    primary = max(top_cards, key=lambda c: SUIT_VALUE[c.suit])
    return Hand(STRAIGHT, cards, primary)


def _detect_flush(cards: list[Card]) -> Optional[Hand]:
    """同花：5张同花色，但不构成顺子"""
    if len(set(c.suit for c in cards)) != 1:
        return None
    if _detect_straight(cards):
        return None  # 同花顺，不是普通同花
    primary = max(cards, key=lambda c: c.score())
    return Hand(FLUSH, cards, primary)


def _detect_full_house(cards: list[Card]) -> Optional[Hand]:
    """三带对"""
    groups = _group_by_rank(cards)
    triple_rank = pair_rank = None
    for rank, grp in groups.items():
        if len(grp) == 3: triple_rank = rank
        elif len(grp) == 2: pair_rank = rank
    if triple_rank is None or pair_rank is None:
        return None
    triple_cards = groups[triple_rank]
    primary = max(triple_cards, key=lambda c: c.score())
    return Hand(FULL_HOUSE, cards, primary)


def _detect_four_with_one(cards: list[Card]) -> Optional[Hand]:
    """四带一"""
    groups = _group_by_rank(cards)
    quad_rank = single_exists = None
    for rank, grp in groups.items():
        if len(grp) == 4: quad_rank = rank
        elif len(grp) == 1: single_exists = True
    if quad_rank is None or not single_exists:
        return None
    quad_cards = groups[quad_rank]
    primary = max(quad_cards, key=lambda c: c.score())
    return Hand(FOUR_WITH_ONE, cards, primary)


def _detect_straight_flush(cards: list[Card]) -> Optional[Hand]:
    """同花顺"""
    if len(set(c.suit for c in cards)) != 1:
        return None
    straight = _detect_straight(cards)
    if not straight:
        return None
    return Hand(STRAIGHT_FLUSH, cards, straight.primary_card)


# ─── 牌型比较 ─────────────────────────────────────────────────────────────────

def compare_hands(a: Hand, b: Hand) -> int:
    """比较两手牌的大小。正数表示 a > b，负数表示 a < b，0 相等。
    只有相同 size 的牌才能比较，5张牌型之间可以互压。"""
    if a.size != b.size:
        raise ValueError(f'Cannot compare hands of different sizes: {a.size} vs {b.size}')
    if a.size == 5:
        return _compare_five(a, b)
    if a.type != b.type:
        raise ValueError(f'Cannot compare different non-five-card types: {a.type} vs {b.type}')
    return _compare_primary(a, b)


def can_beat(a: Hand, b: Hand) -> bool:
    """判断 a 能否压过 b"""
    if a.size != b.size:
        return False
    try:
        return compare_hands(a, b) > 0
    except ValueError:
        return False


def _compare_five(a: Hand, b: Hand) -> int:
    pa = FIVE_CARD_PRIORITY.get(a.type, -1)
    pb = FIVE_CARD_PRIORITY.get(b.type, -1)
    if pa != pb:
        return pa - pb
    return _compare_primary(a, b)


def _compare_primary(a: Hand, b: Hand) -> int:
    """通过 primary_card 比较大小（必须与 TS HandComparator 完全一致）"""
    if a.type in (STRAIGHT, STRAIGHT_FLUSH) and b.type in (STRAIGHT, STRAIGHT_FLUSH):
        ra = STRAIGHT_RANK_VALUE.get(a.primary_card.rank, 0)
        rb = STRAIGHT_RANK_VALUE.get(b.primary_card.rank, 0)
        if ra != rb: return ra - rb
        return SUIT_VALUE[a.primary_card.suit] - SUIT_VALUE[b.primary_card.suit]
    # 同花: 先比花色(♠>♥>♣>♦), 同花色才比最大牌点数
    if a.type == FLUSH and b.type == FLUSH:
        suit_diff = SUIT_VALUE[a.primary_card.suit] - SUIT_VALUE[b.primary_card.suit]
        if suit_diff != 0:
            return suit_diff
        return RANK_VALUE[a.primary_card.rank] - RANK_VALUE[b.primary_card.rank]
    sa = a.primary_card.score()
    sb = b.primary_card.score()
    return sa - sb


# ─── 工具函数 ─────────────────────────────────────────────────────────────────

def _group_by_rank(cards: list[Card]) -> dict[str, list[Card]]:
    groups: dict[str, list[Card]] = {}
    for c in cards:
        groups.setdefault(c.rank, []).append(c)
    return groups


def _group_by_suit(cards: list[Card]) -> dict[str, list[Card]]:
    groups: dict[str, list[Card]] = {}
    for c in cards:
        groups.setdefault(c.suit, []).append(c)
    return groups
