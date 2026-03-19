"""Quick validation of VectorizedEngine."""
from a3dizhu_cpp import VectorizedEngine
import numpy as np

vec = VectorizedEngine(10)
vec.set_greedy_ratio(0.05)
vec.set_random_ratio(0.01)
vec.seed(42)

for i in range(10):
    vec.reset(i)

done, rule_obs, rule_pids = vec.advance_to_decision(0)
print(f'advance_to_decision: done={done}, rule_steps={len(rule_pids)}, obs_shape={rule_obs.shape}')

for i in range(1, 10):
    vec.advance_to_decision(i)
pending = [i for i in range(10) if not vec.is_over(i)]
print(f'pending envs: {len(pending)}')

if pending:
    obs_exp, act_flat, obs_raw, offsets, all_keys = vec.prepare_batch(pending)
    print(f'prepare_batch: obs_exp={obs_exp.shape}, act_flat={act_flat.shape}, obs_raw={obs_raw.shape}')
    print(f'offsets={list(offsets[:5])}... total_actions={offsets[-1]}')
    print(f'action_keys sample: {len(all_keys[0])} actions for env 0')

# Run a full game on env 0
vec.reset(0)
steps = 0
while not vec.is_over(0):
    done, _, _ = vec.advance_to_decision(0)
    if done:
        break
    obs_exp, act_flat, obs_raw, offsets, all_keys = vec.prepare_batch([0])
    vec.step(0, all_keys[0][0])
    steps += 1
payoffs = vec.get_training_payoffs(0)
print(f'Full game: {steps} RL decisions, payoffs={payoffs}')
print('All VectorizedEngine tests passed!')
