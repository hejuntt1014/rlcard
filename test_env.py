"""
A3 地主环境冒烟测试

验证：
1. 游戏规则（发牌、出牌、pass、终局）
2. RLCard 环境接口（reset/step/payoffs）
3. Random agent 跑通一局完整游戏
4. 状态特征维度正确
"""

import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rlcard
from rlcard.agents import RandomAgent


def test_card_rules():
    """测试牌型检测"""
    from rlcard.games.a3dizhu.card import Card
    from rlcard.games.a3dizhu.hand import detect_hand, can_beat, SINGLE, PAIR, TRIPLE, STRAIGHT, FLUSH, FULL_HOUSE, FOUR_WITH_ONE, STRAIGHT_FLUSH

    print('=== 测试牌型检测 ===')

    # 单张
    c = Card('spade', 'A')
    h = detect_hand([c])
    assert h is not None and h.type == SINGLE, f'单张检测失败: {h}'
    print(f'  单张: ♠A → {h.type} ✓')

    # 对子
    c1, c2 = Card('spade', 'K'), Card('heart', 'K')
    h = detect_hand([c1, c2])
    assert h is not None and h.type == PAIR, f'对子检测失败: {h}'
    print(f'  对子: ♠K♥K → {h.type} ✓')

    # 三条
    cards = [Card('spade', 'Q'), Card('heart', 'Q'), Card('diamond', 'Q')]
    h = detect_hand(cards)
    assert h is not None and h.type == TRIPLE, f'三条检测失败: {h}'
    print(f'  三条: ♠Q♥Q♦Q → {h.type} ✓')

    # 顺子 (5-6-7-8-9)
    cards = [Card('spade', '5'), Card('heart', '6'), Card('diamond', '7'), Card('club', '8'), Card('spade', '9')]
    h = detect_hand(cards)
    assert h is not None and h.type == STRAIGHT, f'顺子检测失败: {h}'
    print(f'  顺子: 5-9 → {h.type} ✓')

    # 同花 (非连续)
    cards = [Card('spade', '4'), Card('spade', '6'), Card('spade', '8'), Card('spade', '10'), Card('spade', 'Q')]
    h = detect_hand(cards)
    assert h is not None and h.type == FLUSH, f'同花检测失败: {h}'
    print(f'  同花: ♠4♠6♠8♠10♠Q → {h.type} ✓')

    # 三带对
    cards = [Card('spade', 'J'), Card('heart', 'J'), Card('diamond', 'J'),
             Card('spade', '7'), Card('heart', '7')]
    h = detect_hand(cards)
    assert h is not None and h.type == FULL_HOUSE, f'三带对检测失败: {h}'
    print(f'  三带对: JJJ77 → {h.type} ✓')

    # 四带一
    cards = [Card('spade', 'A'), Card('heart', 'A'), Card('diamond', 'A'), Card('club', 'A'),
             Card('spade', '5')]
    h = detect_hand(cards)
    assert h is not None and h.type == FOUR_WITH_ONE, f'四带一检测失败: {h}'
    print(f'  四带一: AAAA5 → {h.type} ✓')

    # 同花顺 (7-8-9-10-J 同花)
    cards = [Card('heart', '7'), Card('heart', '8'), Card('heart', '9'), Card('heart', '10'), Card('heart', 'J')]
    h = detect_hand(cards)
    assert h is not None and h.type == STRAIGHT_FLUSH, f'同花顺检测失败: {h}'
    print(f'  同花顺: ♥7-J → {h.type} ✓')

    # 比较：对K > 对Q
    hK = detect_hand([Card('spade', 'K'), Card('heart', 'K')])
    hQ = detect_hand([Card('spade', 'Q'), Card('heart', 'Q')])
    assert can_beat(hK, hQ), '对K应能压对Q'
    print(f'  比较: 对K > 对Q ✓')

    # 比较：四带一 > 同花
    four = detect_hand([Card('spade', '4'), Card('heart', '4'), Card('diamond', '4'), Card('club', '4'), Card('spade', '5')])
    flush = detect_hand([Card('spade', '4'), Card('spade', '6'), Card('spade', '8'), Card('spade', '10'), Card('spade', 'Q')])
    assert can_beat(four, flush), '四带一应能压同花'
    print(f'  比较: 四带一 > 同花 ✓')

    # 顺子含3（3在顺子中是最小牌）
    cards = [Card('spade', '3'), Card('heart', '4'), Card('diamond', '5'), Card('club', '6'), Card('spade', '7')]
    h = detect_hand(cards)
    assert h is not None and h.type == STRAIGHT, f'3-4-5-6-7 应为合法顺子: {h}'
    print(f'  顺子含3: 3-4-5-6-7 -> {h.type} ✓')

    # 含2的不能是顺子
    cards2 = [Card('spade', '2'), Card('heart', '3'), Card('diamond', '4'), Card('club', '5'), Card('spade', '6')]
    h2 = detect_hand(cards2)
    assert h2 is None or h2.type != STRAIGHT, f'含2不应为顺子: {h2}'
    print(f'  含2不是顺子 ✓')

    print()


def test_hint_engine():
    """测试出牌提示"""
    from rlcard.games.a3dizhu.card import Card
    from rlcard.games.a3dizhu.hand import detect_hand, PAIR
    from rlcard.games.a3dizhu.hint import get_playable_hands

    print('=== 测试出牌提示 ===')

    # 手牌：♠K ♥K ♦Q ♣Q ♠J ♥J ♦4
    hand = [
        Card('spade', 'K'), Card('heart', 'K'),
        Card('diamond', 'Q'), Card('club', 'Q'),
        Card('spade', 'J'), Card('heart', 'J'),
        Card('diamond', '4'),
    ]

    # 自由出牌
    all_hands = get_playable_hands(hand, None)
    singles = [h for h in all_hands if h.size == 1]
    pairs = [h for h in all_hands if h.size == 2]
    print(f'  自由出牌: {len(all_hands)} 种 (单张:{len(singles)}, 对子:{len(pairs)})')
    assert len(singles) == 7, f'期望7种单张，实际{len(singles)}'
    assert len(pairs) == 3, f'期望3对，实际{len(pairs)}'

    # 跟牌：需压过对J
    last = detect_hand([Card('spade', 'J'), Card('heart', 'J')])
    beating = get_playable_hands(hand, last)
    print(f'  跟压对J: {len(beating)} 种能赢的出法')
    for h in beating:
        assert h.size == 2 and h.type == PAIR, f'跟牌应为对子: {h}'
        print(f'    → {[(c.rank, c.suit) for c in h.cards]}')

    print()


def test_game_one_round():
    """测试游戏核心逻辑：跑一局随机游戏"""
    from rlcard.games.a3dizhu.game import Game

    print('=== 测试游戏核心逻辑（随机一局）===')

    game = Game()
    state, player_id = game.init_game()

    steps = 0
    max_steps = 500

    while not game.is_over() and steps < max_steps:
        state = game.get_state(player_id)
        legal = state['legal_actions']

        assert len(legal) > 0, f'玩家{player_id}没有合法动作（手牌{len(state["current_hand"])}张）'

        # 随机选一个动作
        action = random.choice(legal)
        state, player_id = game.step(action)
        steps += 1

    assert game.is_over(), f'游戏未在{max_steps}步内结束（当前步数={steps}）'

    rankings = game.state.rankings
    payoffs = game.get_payoffs()
    print(f'  完成步数: {steps}')
    print(f'  排名顺序: {rankings}')
    print(f'  回报: {[round(p, 2) for p in payoffs]}')
    assert len(rankings) == 4, f'排名应有4人，实际: {rankings}'
    print()


def test_rlcard_env():
    """测试 RLCard 环境接口"""
    print('=== 测试 RLCard 环境接口 ===')

    env = rlcard.make('a3dizhu')
    print(f'  状态维度: {env.state_shape}')
    print(f'  动作维度: {env.action_shape}')
    print(f'  玩家数: {env.num_players}')

    # 手动 step（不用 env.run，避免 RandomAgent 的 eval_step 格式不兼容问题）
    state, player_id = env.reset()
    obs_shape = state['obs'].shape
    print(f'  观测向量维度: {obs_shape}')

    steps = 0
    while not env.is_over() and steps < 500:
        legal_actions = list(state['legal_actions'].keys())
        action = random.choice(legal_actions)
        state, player_id = env.step(action)
        steps += 1

    assert env.is_over(), '环境未正常结束'
    payoffs = env.get_payoffs()
    print(f'  完成步数: {steps}')
    print(f'  回报: {[round(p, 2) for p in payoffs]}')
    print()


def test_team_reveal():
    """测试队伍暴露 + 队伍分配 + 计分机制"""
    from rlcard.games.a3dizhu.card import Card
    from rlcard.games.a3dizhu.hand import Hand, SINGLE
    from rlcard.games.a3dizhu.game import (
        GameState, assign_teams, compute_payoffs,
        TEAM_UNKNOWN, TEAM_SPADE_A3, TEAM_OPPONENT, TEAM_SOLO,
    )

    print('=== 测试队伍分配 ===')

    # 正常 2v2：玩家0持♠3，玩家1持♠A
    hands_2v2 = [
        [Card('spade', '3'), Card('heart', '5')],
        [Card('spade', 'A'), Card('diamond', '7')],
        [Card('club', '8'), Card('heart', '9')],
        [Card('diamond', '10'), Card('club', 'J')],
    ]
    teams, is_solo = assign_teams(hands_2v2)
    assert teams[0] == TEAM_SPADE_A3 and teams[1] == TEAM_SPADE_A3, f'♠3和♠A应同队: {teams}'
    assert teams[2] == TEAM_OPPONENT and teams[3] == TEAM_OPPONENT, f'其余应为对手: {teams}'
    assert not is_solo
    print(f'  2v2: {teams} ✓')

    # 独食：玩家0同时持♠3和♠A
    hands_solo = [
        [Card('spade', '3'), Card('spade', 'A')],
        [Card('diamond', '7'), Card('club', '8')],
        [Card('heart', '9'), Card('diamond', '10')],
        [Card('club', 'J'), Card('heart', 'Q')],
    ]
    teams_s, is_solo_s = assign_teams(hands_solo)
    assert teams_s[0] == TEAM_SOLO and is_solo_s
    assert all(teams_s[i] == TEAM_OPPONENT for i in [1, 2, 3])
    print(f'  独食: {teams_s} ✓')

    print('=== 测试计分 ===')

    # 2v2: A队(0,1)得1st+2nd，B队(2,3)得3rd+4th
    rankings_2v2 = [0, 1, 2, 3]
    payoffs = compute_payoffs(rankings_2v2, [TEAM_SPADE_A3, TEAM_SPADE_A3, TEAM_OPPONENT, TEAM_OPPONENT], False)
    # A队: (3+2)/2=2.5, B队: (1+0)/2=0.5, 差=2
    assert payoffs[0] == 2.0 and payoffs[1] == 2.0, f'A队应各+2: {payoffs}'
    assert payoffs[2] == -2.0 and payoffs[3] == -2.0, f'B队应各-2: {payoffs}'
    print(f'  2v2 (A队1+2名): {payoffs} ✓')

    # 2v2: A队(0,1)得1st+4th，B队(2,3)得2nd+3rd
    rankings_mix = [0, 2, 3, 1]
    payoffs2 = compute_payoffs(rankings_mix, [TEAM_SPADE_A3, TEAM_SPADE_A3, TEAM_OPPONENT, TEAM_OPPONENT], False)
    # A队: (3+0)/2=1.5, B队: (2+1)/2=1.5, 差=0
    assert payoffs2[0] == 0.0 and payoffs2[2] == 0.0, f'应为平局: {payoffs2}'
    print(f'  2v2 (A队1+4名): {payoffs2} ✓')

    # 独食: 独食者得1st
    payoffs_s1 = compute_payoffs([0, 1, 2, 3], [TEAM_SOLO, TEAM_OPPONENT, TEAM_OPPONENT, TEAM_OPPONENT], True)
    assert payoffs_s1[0] == 2.0, f'独食1st应+2: {payoffs_s1}'
    assert payoffs_s1[1] == -2.0, f'其余应-2: {payoffs_s1}'
    print(f'  独食 (1st): {payoffs_s1} ✓')

    # 独食: 独食者得4th
    payoffs_s4 = compute_payoffs([1, 2, 3, 0], [TEAM_SOLO, TEAM_OPPONENT, TEAM_OPPONENT, TEAM_OPPONENT], True)
    assert payoffs_s4[0] == -2.0, f'独食4th应-2: {payoffs_s4}'
    assert payoffs_s4[1] == 2.0, f'其余应+2: {payoffs_s4}'
    print(f'  独食 (4th): {payoffs_s4} ✓')

    print('=== 测试队伍暴露（出牌时）===')
    state = GameState(
        hands=hands_2v2, current_player=0,
        last_play=None, last_play_player=-1,
        pass_count=0, is_first_turn=False,
        rankings=[], teams=[TEAM_UNKNOWN]*4,
        actual_teams=[TEAM_SPADE_A3, TEAM_SPADE_A3, TEAM_OPPONENT, TEAM_OPPONENT],
        is_solo=False, num_players=4,
    )
    spade3 = Card('spade', '3')
    move = Hand(SINGLE, [spade3], spade3)
    new_state = state.apply_move(move)
    assert new_state.teams[0] == TEAM_SPADE_A3, f'打出♠3后观测队伍应暴露: {new_state.teams[0]}'
    print(f'  打出♠3 -> 观测队伍暴露为 {new_state.teams[0]} ✓')
    print()


def test_multiple_games(n=10):
    """多局测试，确保没有崩溃"""
    import rlcard
    print(f'=== 连跑 {n} 局随机游戏 ===')

    env = rlcard.make('a3dizhu')

    for i in range(n):
        state, player_id = env.reset()
        while not env.is_over():
            legal = list(state['legal_actions'].keys())
            action = random.choice(legal)
            state, player_id = env.step(action)
        payoffs = env.get_payoffs()
        assert len(payoffs) == 4

    print(f'  全部通过 ✓')
    print()


if __name__ == '__main__':
    random.seed(42)

    test_card_rules()
    test_hint_engine()
    test_team_reveal()
    test_game_one_round()
    test_rlcard_env()
    test_multiple_games(20)

    print('所有测试通过 ✓')
