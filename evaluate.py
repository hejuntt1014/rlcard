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
    队伍计分：payoff ∈ {+2, +1, 0, -1, -2}
      +2 = 大胜（队伍1+2名），+1 = 小胜（1+3名），0 = 平（1+4名）
      -1 = 小负，-2 = 大负
    """
    results = {
        'win_big': 0,    # payoff = +2
        'win_small': 0,  # payoff = +1
        'draw': 0,       # payoff = 0
        'lose_small': 0, # payoff = -1
        'lose_big': 0,   # payoff = -2
        'total': 0,
        'total_payoff': 0.0,
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

    return results


def evaluate_vs_random(env, dmc_model, num_games=100):
    """阶梯一：DMC vs 随机 Agent"""
    return _run_evaluation(env, dmc_model, RandomRuleAgent(), num_games)


def evaluate_1v3(env, dmc_model, num_games=100):
    """阶梯二：1个DMC vs 3个贪心规则AI"""
    return _run_evaluation(env, dmc_model, GreedyRuleAgent(), num_games)


def evaluate_2v2(env, dmc_model, num_games=100):
    """2个DMC vs 2个规则AI，交替坐（0,2=DMC, 1,3=规则）"""
    rule_agent = GreedyRuleAgent()
    results = {'dmc_1st': 0, 'dmc_2nd': 0, 'dmc_3rd': 0, 'dmc_4th': 0, 'total': 0}

    dmc_seats = {0, 2}  # DMC 坐 0 和 2 号位

    for game_i in range(num_games):
        # 每局交替（偶数局 DMC 坐 0,2；奇数局 DMC 坐 1,3）
        if game_i % 2 == 1:
            dmc_seats_this = {1, 3}
        else:
            dmc_seats_this = {0, 2}

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

    print(f'\n{"="*54}')
    print(f'  评估结果 ({mode}, {total} 局)')
    print(f'{"="*54}')
    print(f'  大胜(+2): {wb:4d} ({wb/total*100:5.1f}%)  队伍1+2名')
    print(f'  小胜(+1): {ws:4d} ({ws/total*100:5.1f}%)  队伍1+3名')
    print(f'  平局( 0): {dr:4d} ({dr/total*100:5.1f}%)  队伍1+4名')
    print(f'  小负(-1): {ls:4d} ({ls/total*100:5.1f}%)')
    print(f'  大负(-2): {lb:4d} ({lb/total*100:5.1f}%)')
    print(f'  ────────────────────────────────────')
    print(f'  胜率: {win_rate:.1f}%  ({wins}胜 {dr}平 {losses}负)')
    print(f'  平均回报: {avg_payoff:+.2f}  (>0=赢, <0=输)')
    print(f'{"="*54}\n')


def main():
    parser = argparse.ArgumentParser('A3 地主模型评估')
    parser.add_argument('--model_path', type=str,
                        default='experiments/a3dizhu_v3/model.tar')
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
