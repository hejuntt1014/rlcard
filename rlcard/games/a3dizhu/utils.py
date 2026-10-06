"""
A3 地主动作空间定义与特征工具

动作空间设计：
  每个动作用一个整数 ID 表示。
  动作 = pass(0) | 出牌组合(1..N)

  为了让神经网络能学习，我们需要把所有可能出现的牌型组合预先枚举。
  但 52 张牌的组合数太大，所以用「类型+值」编码：

  单张:  52种  (suit*13 + rank_idx)  → id 0..51
  对子:  13种  (只按 rank)           → id 52..64
  三条:  13种                         → id 65..77
  顺子:  Cx5 (从4连到A连，共7种 rank起点 × 4^5花色) → 压缩为 rank级别 id
  ...

  实际训练中，我们直接用 「合法动作 mask + 动作特征向量」 的方式，
  而不是预定义全动作表。这样扩展性更好。

  我们把动作用一个 52 维的 one-hot 向量表示（每张牌是否打出）。
  pass 用全零向量表示。
  这对 DMC 算法完全兼容（DMC 的 action_feature 就是这种格式）。
"""

from __future__ import annotations
import numpy as np
from .card import Card, SUITS, RANKS

# 牌到索引的映射
CARD_TO_IDX: dict[str, int] = {}
IDX_TO_CARD: dict[int, str] = {}

for _suit_i, _suit in enumerate(SUITS):
    for _rank_i, _rank in enumerate(RANKS):
        _idx = _suit_i * 13 + _rank_i
        _card_id = f'{_suit}_{_rank}'
        CARD_TO_IDX[_card_id] = _idx
        IDX_TO_CARD[_idx] = _card_id

NUM_CARDS = 52  # 52张牌（无大小王）

# 动作特征维度 = 52（每张牌是否打出）
ACTION_FEATURE_DIM = NUM_CARDS

# pass 动作的特征向量（全零）
PASS_FEATURE = np.zeros(ACTION_FEATURE_DIM, dtype=np.int8)


def hand_to_feature(hand) -> np.ndarray:
    """将一手牌（Hand 对象或 None=pass）转换为 52 维特征向量"""
    if hand is None:
        return PASS_FEATURE.copy()
    vec = np.zeros(ACTION_FEATURE_DIM, dtype=np.int8)
    for c in hand.cards:
        idx = CARD_TO_IDX.get(c.to_id())
        if idx is not None:
            vec[idx] = 1
    return vec


def cards_to_feature(cards: list[Card]) -> np.ndarray:
    """将手牌列表转换为 52 维 one-hot 特征（值为持有张数 0/1）"""
    vec = np.zeros(NUM_CARDS, dtype=np.int8)
    for c in cards:
        idx = CARD_TO_IDX.get(c.to_id())
        if idx is not None:
            vec[idx] = 1
    return vec


def cards_count_feature(cards: list[Card]) -> np.ndarray:
    """将手牌列表转换为 52 维计数特征（每张牌可出现多次，但A3地主每张唯一）"""
    return cards_to_feature(cards)  # 无重复


# 动作空间大小：我们用 「合法动作列表+特征」 模式，不预定义固定大小
# 但 RLCard DMC 需要 num_actions，我们设置为一个足够大的上界
# 实际上 DMC 是通过 action_feature 来区分动作，而不是动作ID，所以这个值影响不大
NUM_ACTIONS = 1  # DMC 使用 action_feature，这里设 1 仅作占位
