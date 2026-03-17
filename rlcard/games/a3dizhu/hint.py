"""
A3 地主出牌提示引擎（对应 TypeScript HintEngine）

getPlayableHands(my_cards, last_hand) -> list[Hand]
  - last_hand=None：自由出牌，返回所有合法牌型
  - last_hand!=None：跟牌，返回所有能压过的牌型 + 空列表表示pass
"""

from __future__ import annotations
from itertools import combinations as _comb
from typing import Optional
from .card import Card, STRAIGHT_RANK_VALUE
from .hand import (
    Hand, detect_hand, can_beat, compare_hands,
    _group_by_rank, _group_by_suit,
    SINGLE, PAIR, TRIPLE, STRAIGHT, FLUSH, FULL_HOUSE, FOUR_WITH_ONE, STRAIGHT_FLUSH,
    FIVE_CARD_PRIORITY,
)


def get_playable_hands(my_cards: list[Card], last_hand: Optional[Hand]) -> list[Hand]:
    """返回所有合法（且能压过 last_hand）的出牌方案，已按从小到大排序"""
    if not last_hand:
        return _get_all_hands(my_cards)
    return _find_beating_hands(my_cards, last_hand)


def has_playable_hand(my_cards: list[Card], last_hand: Optional[Hand]) -> bool:
    """是否存在能压过 last_hand 的出法"""
    if not last_hand:
        return len(my_cards) > 0
    results = _find_beating_hands(my_cards, last_hand)
    return len(results) > 0


# ─── 内部实现 ─────────────────────────────────────────────────────────────────

def _find_beating_hands(my_cards: list[Card], last_hand: Hand) -> list[Hand]:
    results: list[Hand] = []
    size = last_hand.size
    if size == 1:
        _find_beating_singles(my_cards, last_hand, results)
    elif size == 2:
        _find_beating_pairs(my_cards, last_hand, results)
    elif size == 3:
        _find_beating_triples(my_cards, last_hand, results)
    elif size == 5:
        _find_beating_five(my_cards, last_hand, results)
    results.sort(key=lambda h: _hand_sort_key(h))
    return results


def _find_beating_singles(cards: list[Card], last: Hand, results: list[Hand]) -> None:
    for c in cards:
        h = Hand(SINGLE, [c], c)
        if can_beat(h, last):
            results.append(h)


def _find_beating_pairs(cards: list[Card], last: Hand, results: list[Hand]) -> None:
    groups = _group_by_rank(cards)
    for rank, grp in groups.items():
        if len(grp) < 2: continue
        for combo in _comb(grp, 2):
            h = detect_hand(list(combo))
            if h and can_beat(h, last):
                results.append(h)


def _find_beating_triples(cards: list[Card], last: Hand, results: list[Hand]) -> None:
    groups = _group_by_rank(cards)
    for rank, grp in groups.items():
        if len(grp) < 3: continue
        for combo in _comb(grp, 3):
            h = detect_hand(list(combo))
            if h and can_beat(h, last):
                results.append(h)


def _find_beating_five(cards: list[Card], last: Hand, results: list[Hand]) -> None:
    last_priority = FIVE_CARD_PRIORITY.get(last.type, -1)
    seen: set[frozenset] = set()

    def _add(h: Optional[Hand]):
        if h is None: return
        key = h.card_ids()
        if key in seen: return
        seen.add(key)
        results.append(h)

    if last_priority <= FIVE_CARD_PRIORITY[STRAIGHT]:
        _find_straights(cards, last, _add)
    if last_priority <= FIVE_CARD_PRIORITY[FLUSH]:
        _find_flushes(cards, last, _add)
    if last_priority <= FIVE_CARD_PRIORITY[FULL_HOUSE]:
        _find_full_houses(cards, last, _add)
    if last_priority <= FIVE_CARD_PRIORITY[FOUR_WITH_ONE]:
        _find_four_with_ones(cards, last, _add)
    if last_priority <= FIVE_CARD_PRIORITY[STRAIGHT_FLUSH]:
        _find_straight_flushes(cards, last, _add)


def _find_straights(cards, last, add_fn):
    groups = _group_by_rank_local([c for c in cards if c.rank != '2'])
    # 按顺子点数排序的key列表
    rank_keys = sorted(groups.keys(), key=lambda r: STRAIGHT_RANK_VALUE[r])
    for i in range(len(rank_keys) - 4):
        window = rank_keys[i:i+5]
        if not _is_consecutive(window): continue
        card_options = [groups[r] for r in window]
        for combo in _cartesian(card_options):
            h = detect_hand(list(combo))
            if h and (not last or can_beat(h, last)):
                add_fn(h)


def _group_by_rank_local(cards) -> dict:
    groups = {}
    for c in cards:
        groups.setdefault(c.rank, []).append(c)
    return groups


def _find_flushes(cards, last, add_fn):
    suit_groups = _group_by_suit(cards)
    for grp in suit_groups.values():
        if len(grp) < 5: continue
        for combo in _comb(grp, 5):
            h = detect_hand(list(combo))
            if h and h.type == FLUSH and (not last or can_beat(h, last)):
                add_fn(h)


def _find_full_houses(cards, last, add_fn):
    groups = _group_by_rank(cards)
    entries = list(groups.items())
    for triple_rank, triple_grp in entries:
        if len(triple_grp) < 3: continue
        for triple in _comb(triple_grp, 3):
            for pair_rank, pair_grp in entries:
                if pair_rank == triple_rank or len(pair_grp) < 2: continue
                for pair in _comb(pair_grp, 2):
                    combo = list(triple) + list(pair)
                    h = detect_hand(combo)
                    if h and (not last or can_beat(h, last)):
                        add_fn(h)


def _find_four_with_ones(cards, last, add_fn):
    groups = _group_by_rank(cards)
    for quad_rank, quad_grp in groups.items():
        if len(quad_grp) < 4: continue
        quads = quad_grp[:4]
        for kicker in cards:
            if kicker.rank == quad_rank: continue
            combo = quads + [kicker]
            h = detect_hand(combo)
            if h and (not last or can_beat(h, last)):
                add_fn(h)


def _find_straight_flushes(cards, last, add_fn):
    suit_groups = _group_by_suit(cards)
    for grp in suit_groups.values():
        if len(grp) < 5: continue
        _find_straights(grp, last, lambda h: add_fn(h) if h and h.type == STRAIGHT_FLUSH else None)


def _get_all_hands(cards: list[Card]) -> list[Hand]:
    results: list[Hand] = []
    seen: set[frozenset] = set()

    def _add(h: Optional[Hand]):
        if h is None: return
        key = h.card_ids()
        if key in seen: return
        seen.add(key)
        results.append(h)

    # 单张
    for c in cards:
        _add(Hand(SINGLE, [c], c))

    groups = _group_by_rank(cards)
    for rank, grp in groups.items():
        if len(grp) >= 2:
            for combo in _comb(grp, 2):
                _add(detect_hand(list(combo)))
        if len(grp) >= 3:
            for combo in _comb(grp, 3):
                _add(detect_hand(list(combo)))

    _find_straights(cards, None, _add)
    _find_flushes(cards, None, _add)
    _find_full_houses(cards, None, _add)
    _find_four_with_ones(cards, None, _add)
    # 同花顺已包含在 straight + flush 检测中（detect_hand 优先返回 straight_flush）

    results.sort(key=lambda h: (h.size, _hand_sort_key(h)))
    return results


def _is_consecutive(rank_keys: list[str]) -> bool:
    vals = [STRAIGHT_RANK_VALUE[r] for r in rank_keys]
    return all(vals[i] - vals[i-1] == 1 for i in range(1, len(vals)))


def _cartesian(arrays):
    """笛卡尔积，用于枚举顺子的所有花色组合"""
    if not arrays: yield []; return
    for item in arrays[0]:
        for rest in _cartesian(arrays[1:]):
            yield [item] + rest


def _hand_sort_key(h: Hand):
    """排序键：同 size 内从弱到强"""
    if h.size == 5:
        priority = FIVE_CARD_PRIORITY.get(h.type, -1)
        return (priority, h.primary_card.score())
    return (0, h.primary_card.score())
