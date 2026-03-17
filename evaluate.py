"""
A3 地主训练评估脚本

用训练中的 DMC 模型对阵贪心规则 Agent，测量真实水平。
可手动运行或定期 cron 调度。

用法：
  # 评估最新模型（默认跑 100 局）
  python evaluate.py

  # 自定义局数和模型路径
  python evaluate.py --num_games 200 --model_path experiments/dmc_result/a3dizhu_v1/model.tar

  # 不同对阵配置
  python evaluate.py --mode 1v3   # 1个DMC vs 3个规则AI
  python evaluate.py --mode 2v2   # 2个DMC vs 2个规则AI（交替坐）

  # 阶梯式评估（随机 + 贪心，一次全跑）
  python evaluate.py --mode ladder
"""

import os
import sys
import argparse
import random
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import rlcard
from rlcard.games.a3dizhu.rule_agent import GreedyRuleAgent


def load_dmc_agents(model_path, env):
    """加载训练好的 DMC 模型"""
    from rlcard.agents.dmc_agent.model import DMCModel

    model = DMCModel(
        env.state_shape,
        env.action_shape,
        exp_epsilon=0.0,  # 评估时不探索
        device='cpu',
    )

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        for p in range(env.num_players):
            model.get_agent(p).load_state_dict(checkpoint['model_state_dict'][p])
        print(f'Loaded model from {model_path}')
        if 'frames' in checkpoint:
            print(f'  Trained frames: {checkpoint["frames"]:,}')
    else:
        print(f'WARNING: Model not found at {model_path}, using random weights!')

    model.eval()
    return model


class RandomRuleAgent:
    """纯随机 Agent（阶梯一：验证 AI 是否学会了基本规则）"""
    def __init__(self):
        self.use_raw = True

    def step(self, state):
        raw = state if 'legal_actions' in state else state.get('raw_obs', state)
        legal = raw.get('legal_actions', [])
        if not legal:
            return None
        import random as _rng
        return _rng.choice(legal)

    def eval_step(self, state):
        return self.step(state), {}


def _run_evaluation(env, dmc_model, opponent_agent, num_games):
    """通用评估：1个DMC vs 3个opponent，DMC轮流坐每个位置
    2v2 计分：payoff ∈ {+2, +1, 0, -1, -2}
      +2 = 大胜（队伍1+2名），+1 = 小胜（1+3名），0 = 平（1+4名）
    独食 1v3：payoff ∈ {+2, +1, -1, -2}（无平局）
    """
    results = {
        'win_big': 0,       # payoff = +2
        'win_small': 0,     # payoff = +1
        'draw': 0,          # payoff = 0
        'lose_small': 0,    # payoff = -1
        'lose_big': 0,      # payoff = -2
        'total': 0,
        'total_payoff': 0.0,
        # 独食局单独统计
        'solo_total': 0,
        'solo_wins': 0,     # payoff > 0
        'solo_losses': 0,   # payoff < 0
        'solo_payoff': 0.0,
        # 普通局
        'normal_total': 0,
        'normal_wins': 0,
        'normal_losses': 0,
        'normal_draws': 0,
        'normal_payoff': 0.0,
    }

    for game_i in range(num_games):
        dmc_seat = game_i % 4

        state, player_id = env.reset()
        while not env.is_over():
            if player_id == dmc_seat:
                action = dmc_model.get_agent(player_id).step(state)
                state, player_id = env.step(action)
            else:
                raw_state = state.get('raw_obs', state)
                action = opponent_agent.step(raw_state)
                state, player_id = env.step(action, raw_action=True)

        payoffs = env.get_payoffs()
        dmc_payoff = payoffs[dmc_seat]
        results['total_payoff'] += dmc_payoff

        if dmc_payoff >= 1.5:   results['win_big'] += 1
        elif dmc_payoff >= 0.5: results['win_small'] += 1
        elif dmc_payoff >= -0.5:results['draw'] += 1
        elif dmc_payoff >= -1.5:results['lose_small'] += 1
        else:                   results['lose_big'] += 1
        results['total'] += 1

        # 独食局检测
        is_solo = getattr(getattr(env, 'game', None), 'state', None)
        is_solo = getattr(is_solo, 'is_solo', False) if is_solo is not None else False

        if is_solo:
            results['solo_total'] += 1
            results['solo_payoff'] += dmc_payoff
            if dmc_payoff > 0:   results['solo_wins'] += 1
            elif dmc_payoff < 0: results['solo_losses'] += 1
        else:
            results['normal_total'] += 1
            results['normal_payoff'] += dmc_payoff
            if dmc_payoff > 0:    results['normal_wins'] += 1
            elif dmc_payoff < 0:  results['normal_losses'] += 1
            else:                 results['normal_draws'] += 1

    return results


def evaluate_vs_random(env, dmc_model, num_games=100):
    """阶梯一：DMC vs 随机 Agent"""
    return _run_evaluation(env, dmc_model, RandomRuleAgent(), num_games)


def evaluate_1v3(env, dmc_model, num_games=100):
    """阶梯二：1个DMC vs 3个贪心规则AI"""
    return _run_evaluation(env, dmc_model, GreedyRuleAgent(), num_games)


def evaluate_2v2(env, dmc_model, num_games=100):
    """2个DMC vs 2个规则AI，覆盖全部6种座位模式消除位置偏差"""
    rule_agent = GreedyRuleAgent()
    results = {'dmc_1st': 0, 'dmc_2nd': 0, 'dmc_3rd': 0, 'dmc_4th': 0, 'total': 0}

    # 6种模式覆盖所有可能的 a3 队伍组合（{0,2}/{1,3}/{0,1}/{2,3}/{0,3}/{1,2}）
    # 确保两个引擎均等地经历所有座位位置关系
    dmc_seat_patterns = [
        {0, 2}, {1, 3},  # 对角线模式
        {0, 1}, {2, 3},  # 相邻模式
        {0, 3}, {1, 2},  # 首尾模式
    ]

    for game_i in range(num_games):
        dmc_seats_this = dmc_seat_patterns[game_i % 6]

        state, player_id = env.reset()
        while not env.is_over():
            if player_id in dmc_seats_this:
                action = dmc_model.get_agent(player_id).step(state)
                state, player_id = env.step(action)
            else:
                raw_state = state.get('raw_obs', state)
                action = rule_agent.step(raw_state)
                state, player_id = env.step(action, raw_action=True)

        payoffs = env.get_payoffs()
        for seat in dmc_seats_this:
            p = payoffs[seat]
            if p >= 0.9:   results['dmc_1st'] += 1
            elif p >= 0.4: results['dmc_2nd'] += 1
            elif p >= -0.1:results['dmc_3rd'] += 1
            else:          results['dmc_4th'] += 1
            results['total'] += 1

    return results


def print_results(results, mode):
    total = results['total']
    if total == 0:
        print('No games played.')
        return

    wb = results['win_big']
    ws = results['win_small']
    dr = results['draw']
    ls = results['lose_small']
    lb = results['lose_big']
    wins = wb + ws
    losses = ls + lb
    win_rate = wins / total * 100
    avg_payoff = results['total_payoff'] / total

    solo_n   = results['solo_total']
    normal_n = results['normal_total']

    print(f'\n{"="*54}')
    print(f'  评估结果 ({mode}, {total} 局)')
    print(f'{"="*54}')
    print(f'  大胜(+2): {wb:4d} ({wb/total*100:5.1f}%)  2v2:队伍1+2名 / 独食:1st')
    print(f'  小胜(+1): {ws:4d} ({ws/total*100:5.1f}%)  2v2:队伍1+3名 / 独食:2nd')
    print(f'  平局( 0): {dr:4d} ({dr/total*100:5.1f}%)  仅2v2:队伍1+4名')
    print(f'  小负(-1): {ls:4d} ({ls/total*100:5.1f}%)')
    print(f'  大负(-2): {lb:4d} ({lb/total*100:5.1f}%)')
    print(f'  ────────────────────────────────────')
    print(f'  综合胜率: {win_rate:.1f}%  ({wins}胜 {dr}平 {losses}负)')
    print(f'  平均回报: {avg_payoff:+.2f}  (>0=赢, <0=输)')

    # 普通局统计
    if normal_n > 0:
        nw = results['normal_wins']
        nl = results['normal_losses']
        nd = results['normal_draws']
        nwr = nw / normal_n * 100
        navg = results['normal_payoff'] / normal_n
        print(f'  ────────────────────────────────────')
        print(f'  普通2v2 ({normal_n}局): {nwr:.1f}%胜率  ({nw}胜 {nd}平 {nl}负)  均值{navg:+.2f}')

    # 独食局统计
    if solo_n > 0:
        sw = results['solo_wins']
        sl = results['solo_losses']
        swr = sw / solo_n * 100
        savg = results['solo_payoff'] / solo_n
        print(f'  独食1v3 ({solo_n}局): {swr:.1f}%胜率  ({sw}胜 {sl}负)  均值{savg:+.2f}')
    else:
        print(f'  独食1v3: 未触发（{total}局中无双地主发牌）')

    print(f'{"="*54}\n')


def main():
    parser = argparse.ArgumentParser('A3 地主模型评估')
    parser.add_argument('--model_path', type=str,
                        default='experiments/a3dizhu_v4/model.tar')
    parser.add_argument('--num_games', type=int, default=100)
    parser.add_argument('--mode', type=str, default='ladder',
                        choices=['1v3', '2v2', 'both', 'ladder', 'random'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = rlcard.make('a3dizhu')
    dmc_model = load_dmc_agents(args.model_path, env)

    t0 = time.time()

    if args.mode in ('random', 'ladder'):
        results = evaluate_vs_random(env, dmc_model, args.num_games)
        print_results(results, 'L1: DMC vs Random')

    if args.mode in ('1v3', 'both', 'ladder'):
        results = evaluate_1v3(env, dmc_model, args.num_games)
        print_results(results, 'L2: DMC vs Greedy')

    if args.mode in ('2v2', 'both'):
        results = evaluate_2v2(env, dmc_model, args.num_games)
        print_results(results, 'DMC 2 vs Rule 2')

    elapsed = time.time() - t0
    print(f'评估耗时: {elapsed:.1f}s')


if __name__ == '__main__':
    main()
