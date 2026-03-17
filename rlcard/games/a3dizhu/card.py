"""
A3 地主牌面定义与数值映射

花色: diamond(方块) < club(梅花) < heart(红桃) < spade(黑桃)
点数: 4 < 5 < 6 < 7 < 8 < 9 < 10 < J < Q < K < A < 2 < 3
顺子点数(不含2): 4=1, 5=2, ..., K=10, A=11  (2不能出现在顺子中)
"""

SUITS = ['diamond', 'club', 'heart', 'spade']
RANKS = ['4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', '3']

# 花色数值（用于打破同点数时的大小）
SUIT_VALUE = {s: i for i, s in enumerate(SUITS)}

# 点数数值（用于普通比较）
RANK_VALUE = {r: i for i, r in enumerate(RANKS)}

# 顺子点数映射（与 TS STRAIGHT_RANK_VALUE 完全一致）
# 在顺子中 3 是最小牌(=1)，A 是最大(=12)，2 不能出现在顺子中(=0)
STRAIGHT_RANK_VALUE = {
    '3': 1, '4': 2, '5': 3, '6': 4, '7': 5,
    '8': 6, '9': 7, '10': 8, 'J': 9, 'Q': 10,
    'K': 11, 'A': 12, '2': 0,
}


class Card:
    """单张牌"""
    __slots__ = ('suit', 'rank')

    def __init__(self, suit: str, rank: str):
        self.suit = suit
        self.rank = rank

    def score(self) -> int:
        """用于排序的综合分值（rank权重大，suit用于区分同rank）"""
        return RANK_VALUE[self.rank] * 10 + SUIT_VALUE[self.suit]

    def __eq__(self, other):
        if isinstance(other, Card):
            return self.suit == other.suit and self.rank == other.rank
        return NotImplemented

    def __hash__(self):
        return hash((self.suit, self.rank))

    def __repr__(self):
        return f'{self.rank[0]}{self.suit[0].upper()}'

    def to_id(self) -> str:
        return f'{self.suit}_{self.rank}'

    @staticmethod
    def from_id(card_id: str) -> 'Card':
        suit, rank = card_id.split('_', 1)
        return Card(suit, rank)


def make_deck() -> list[Card]:
    """生成一副完整的52张牌（无大小王）"""
    return [Card(s, r) for s in SUITS for r in RANKS]
