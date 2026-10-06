"""
C++ 引擎 vs Python 引擎完整行为对拍测试

策略：
  Python 引擎发牌，把手牌（card id 列表）传给 C++ 引擎的 reset_with_hands()，
  让两个引擎从完全相同的初始牌面出发，走相同的动作序列，
  每步对比 obs / legal_actions / payoffs 是否完全一致。

运行：
  cd /root/a3dizhu_v6
  python3 test_parity.py
"""

import sys
import random
import numpy as np

sys.path.insert(0, '.')

try:
    from a3dizhu_cpp import CppEngine
except ImportError:
    print("ERROR: a3dizhu_cpp not found")
    sys.exit(1)

from rlcard.envs.a3dizhu.env import A3DizhuPyEnv


def run_parity_game(seed: int, verbose: bool = False):
    """
    用 Python 引擎发牌，手动同步手牌给 C++ 引擎，
    然后两者执行相同动作序列，逐步对比输出。
    """
    rng = random.Random(seed * 31337 + 7)

    # ── Python 引擎 reset ──
    cfg = {'allow_step_back': False, 'seed': seed,
           'greedy_ratio': 0.0, 'random_ratio': 0.0}
    py_env = A3DizhuPyEnv(cfg)
    py_env.game.np_random = random.Random(seed)
    py_state, _ = py_env.reset()

    # ── 从 Python game state 提取手牌 ──
    game = py_env.game
    hands_ids = [[c.to_id() for c in h] for h in game.state.hands]
    start_player = game.state.current_player  # holder of diamond_4

    # ── C++ 引擎用相同手牌初始化 ──
    cpp_eng = CppEngine()
    cpp_eng.reset_with_hands(hands_ids, start_player)

    # declaration phase starts at seat 0
    py_pid = 0
    py_state = py_env.get_state(0)

    step = 0
    while not py_env.is_over() and step < 500:
        pid = py_pid

        # ── 对比 obs ──
        py_obs = py_state['obs']
        cpp_obs = cpp_eng.encode_obs(pid)
        if not np.array_equal(py_obs, cpp_obs):
            diff = np.where(py_obs != cpp_obs)[0]
            return False, (
                f"step={step} pid={pid}: obs mismatch at {len(diff)} positions "
                f"(first indices: {diff[:8]})\n"
                f"  py:  {py_obs[diff[:5]]}\n"
                f"  cpp: {cpp_obs[diff[:5]]}"
            )

        # ── 对比合法动作 ──
        py_legal = set(py_state['legal_actions'].keys())
        cpp_legal = set(cpp_eng.get_legal_actions().keys())
        if py_legal != cpp_legal:
            only_py  = sorted(py_legal - cpp_legal)
            only_cpp = sorted(cpp_legal - py_legal)
            return False, (
                f"step={step} pid={pid}: legal actions mismatch\n"
                f"  only py:  {only_py[:5]}\n"
                f"  only cpp: {only_cpp[:5]}"
            )

        # ── 对比每个动作的 feature 向量 ──
        for key in py_legal:
            pf = py_state['legal_actions'][key]
            cf = np.array(cpp_eng.get_action_feature(key), dtype=np.int8)
            if not np.array_equal(pf, cf):
                return False, (
                    f"step={step}: action feature mismatch for '{key}'"
                )

        # ── 选动作（确定性：排序后按 seed 选） ──
        action_key = rng.choice(sorted(py_legal))

        if verbose:
            print(f"  step={step:3d} pid={pid} #legal={len(py_legal):3d} "
                  f"action={action_key[:35]}")

        # ── 两个引擎执行同一动作 ──
        py_next, py_next_pid = py_env.step(action_key)
        cpp_next_pid = cpp_eng.step(action_key)

        if py_next_pid != cpp_next_pid:
            return False, (
                f"step={step}: next_player mismatch "
                f"py={py_next_pid} cpp={cpp_next_pid}"
            )

        py_state = py_next
        py_pid   = py_next_pid
        step += 1

    if step >= 500:
        return False, "Game not finished in 500 steps"

    # ── 终局对比 ──
    py_p  = py_env.get_payoffs()
    cpp_p = list(cpp_eng.get_payoffs())
    for i in range(4):
        if abs(py_p[i] - cpp_p[i]) > 1e-4:
            return False, f"payoff mismatch py={py_p} cpp={cpp_p}"

    py_tp  = py_env.get_training_payoffs()
    cpp_tp = list(cpp_eng.get_training_payoffs())
    for i in range(4):
        if abs(py_tp[i] - cpp_tp[i]) > 1e-4:
            return False, f"training_payoff mismatch py={py_tp} cpp={cpp_tp}"

    return True, f"OK ({step} steps)"


def main():
    print("=" * 60)
    print("C++ vs Python 行为完整对拍")
    print("=" * 60)

    # 先检查 reset_with_hands 是否存在
    eng = CppEngine()
    if not hasattr(eng, 'reset_with_hands'):
        print("ERROR: CppEngine.reset_with_hands() not found!")
        print("需要先重新编译 C++ 模块（已添加该接口）")
        sys.exit(1)

    print("\n[详细对拍] seed=42:")
    ok, msg = run_parity_game(42, verbose=True)
    print(f"  {'✓ PASS' if ok else '✗ FAIL'} — {msg}")
    if not ok:
        sys.exit(1)

    print(f"\n[大规模对拍] 1000 局...")
    fails = []
    for seed in range(1000):
        ok, msg = run_parity_game(seed)
        if not ok:
            fails.append((seed, msg))
            print(f"  ✗ seed={seed}: {msg}")
            if len(fails) >= 5:
                print("  (停止，太多错误)")
                break

    if not fails:
        print("  ✓ 全部 1000 局通过！C++ 与 Python 行为完全一致")
    else:
        print(f"\n  ✗ {len(fails)} 局不一致")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("对拍通过 — C++ 引擎逻辑正确")
    print("=" * 60)


if __name__ == '__main__':
    main()
