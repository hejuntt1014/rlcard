"""诊断 seed=1 的具体不一致原因"""
import sys, random
sys.path.insert(0, '.')
from a3dizhu_cpp import CppEngine
from rlcard.envs.a3dizhu.env import A3DizhuPyEnv
import numpy as np

seed = 1
rng = random.Random(seed * 31337 + 7)
cfg = {'allow_step_back': False, 'seed': seed, 'greedy_ratio': 0.0, 'random_ratio': 0.0}
py_env = A3DizhuPyEnv(cfg)
py_env.game.np_random = random.Random(seed)
py_state, _ = py_env.reset()

game = py_env.game
hands_ids = [[c.to_id() for c in h] for h in game.state.hands]
start_player = game.state.current_player

cpp_eng = CppEngine()
cpp_eng.reset_with_hands(hands_ids, start_player)

py_pid = 0
py_state = py_env.get_state(0)

for step in range(100):
    pid = py_pid
    py_legal = set(py_state['legal_actions'].keys())
    cpp_legal = set(cpp_eng.get_legal_actions().keys())

    if py_legal != cpp_legal:
        print(f"MISMATCH at step={step} pid={pid}")
        print(f"  py_legal:  {sorted(py_legal)}")
        print(f"  cpp_legal: {sorted(cpp_legal)}")

        gs = py_env.game.state
        print(f"  my hand ({len(gs.hands[pid])} cards): {[str(c) for c in gs.hands[pid]]}")
        print(f"  last_play: {gs.last_play}")
        print(f"  last_play_player: {gs.last_play_player}")
        print(f"  pass_count: {gs.pass_count}")
        print(f"  active players: {[i for i in range(4) if i not in gs.rankings]}")
        print(f"  rankings: {gs.rankings}")
        print(f"  is_declaration_phase: {gs.is_declaration_phase}")
        break

    action_key = rng.choice(sorted(py_legal))
    print(f"  step={step:3d} pid={pid} #legal={len(py_legal)} action={action_key[:40]}")
    py_next, py_next_pid = py_env.step(action_key)
    cpp_next_pid = cpp_eng.step(action_key)
    py_state = py_next
    py_pid = py_next_pid

    if py_env.is_over():
        print(f"Game over at step {step}")
        break
