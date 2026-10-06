"""
验证 DMC 训练链路完整性
模拟 DMC act() 的关键流程：
  env.reset → agent.step → env.step → get_action_feature
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import rlcard
from rlcard.agents.dmc_agent.model import DMCModel


def test_dmc_pipeline():
    print('=== 测试 DMC 训练链路 ===')
    env = rlcard.make('a3dizhu')

    # 创建 DMC 模型（和训练时一样）
    model = DMCModel(
        env.state_shape,
        env.action_shape,
        exp_epsilon=0.5,  # 高探索率快速测试
        device='cpu',
    )
    model.eval()
    env.set_agents(model.get_agents())

    # 模拟 act() 中的流程
    state, player_id = env.reset()
    trajectories = [[] for _ in range(4)]
    trajectories[player_id].append(state)

    steps = 0
    while not env.is_over() and steps < 500:
        # DMC agent 选动作
        action = model.get_agent(player_id).step(state)
        assert isinstance(action, (str, np.str_)), f'action 应为字符串 key, 实际: {type(action)}'

        # 记录到轨迹
        trajectories[player_id].append(action)

        # env.step
        next_state, next_player_id = env.step(action)

        # 保存 state
        if not env.is_over():
            trajectories[next_player_id].append(next_state)

        state = next_state
        player_id = next_player_id
        steps += 1

    assert env.is_over(), f'游戏未在 500 步内结束'

    # 验证 get_action_feature 对字符串 key 的处理
    payoffs = env.get_payoffs()
    for p in range(4):
        traj = trajectories[p]
        for i in range(0, len(traj) - 2, 2):
            obs = traj[i]['obs']
            action_key = traj[i + 1]
            action_feature = env.get_action_feature(action_key)
            assert obs.shape == (env.state_shape[p][0],), f'obs shape 错误: {obs.shape}'
            assert action_feature.shape == (env.action_shape[p][0],), f'action feature shape 错误: {action_feature.shape}'
            assert obs.dtype in (np.int8, np.float32), f'obs dtype 应为 int8: {obs.dtype}'
            assert action_feature.dtype == np.int8, f'action feature dtype 应为 int8: {action_feature.dtype}'

    print(f'  步数: {steps}')
    print(f'  回报: {[round(p, 2) for p in payoffs]}')
    print(f'  obs 维度: {env.state_shape[0][0]}, dtype: int8')
    print(f'  action 维度: {env.action_shape[0][0]}, dtype: int8')

    # 验证 obs 全为整数（int8 buffer 不会截断）
    state_check, _ = env.reset()
    obs = state_check['obs']
    non_int = np.sum(obs != obs.astype(np.int8).astype(obs.dtype))
    assert non_int == 0, f'obs 中有 {non_int} 个非整数值，会被 int8 buffer 截断！'
    print(f'  obs 全部为整数值 ✓')
    print()


if __name__ == '__main__':
    test_dmc_pipeline()
    print('DMC 链路验证通过 ✓')
