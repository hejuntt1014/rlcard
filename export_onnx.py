"""
将 DMC 训练好的模型导出为 ONNX 格式

DMC 有 4 个 agent（每个玩家位置一个），但网络结构完全相同，
只是权重不同。实际推理时我们只需要一个通用模型（用 player 0 的权重，
或者取所有位置的平均权重），因为 RL 对局时每个座位的策略应该是对称的。

ONNX 模型接口：
  输入: obs_action  shape=[N, 752]  (700维obs + 52维action 拼接)
  输出: q_value     shape=[N]       (Q值，越大越好)

Node.js 推理时：
  1. 构造 obs 向量（700维）
  2. 对每个合法动作构造 action 向量（52维）
  3. 拼接 [obs, action] 送入模型
  4. 取 Q 值最大的动作
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class DMCNetForExport(nn.Module):
    """导出用的网络，把 obs+action 合并为单一输入"""

    def __init__(self, state_dim, action_dim, mlp_layers=[512, 512, 512, 512, 512]):
        super().__init__()
        input_dim = state_dim + action_dim
        layer_dims = [input_dim] + mlp_layers
        fc = []
        for i in range(len(layer_dims)-1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(layer_dims[-1], 1))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, obs_action):
        return self.fc_layers(obs_action).flatten()


def export(model_path, output_path, state_dim=700, action_dim=52, average_weights=True):
    print(f'Loading model from {model_path}')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    state_dicts = checkpoint['model_state_dict']
    frames = checkpoint.get('frames', 0)
    print(f'  Trained frames: {frames:,}')
    print(f'  Number of agents: {len(state_dicts)}')

    export_net = DMCNetForExport(state_dim, action_dim)

    if average_weights:
        avg_state = {}
        for key in state_dicts[0]:
            tensors = [state_dicts[i][key].float()
                       for i in range(len(state_dicts))]
            avg_state[key] = torch.mean(torch.stack(tensors), dim=0)
        export_net.load_state_dict(avg_state)
        print('  Using averaged weights from all 4 agents')
    else:
        export_net.load_state_dict(state_dicts[0])
        print('  Using agent 0 weights')

    export_net.eval()

    dummy_input = torch.randn(1, state_dim + action_dim)

    torch.onnx.export(
        export_net,
        dummy_input,
        output_path,
        input_names=['obs_action'],
        output_names=['q_value'],
        dynamic_axes={
            'obs_action': {0: 'batch_size'},
            'q_value': {0: 'batch_size'},
        },
        opset_version=17,
    )

    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f'\nExported to {output_path} ({file_size:.1f} MB)')

    # 验证 ONNX 模型
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        test_input = np.random.randn(
            5, state_dim + action_dim).astype(np.float32)
        result = session.run(None, {'obs_action': test_input})
        print(
            f'ONNX verification: input shape {test_input.shape} -> output shape {result[0].shape}')
        print(f'Sample Q values: {result[0]}')
        print('ONNX export verified OK!')
    except ImportError:
        print('onnxruntime not installed, skipping verification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Export DMC model to ONNX')
    parser.add_argument('--model_path', type=str,
                        default='experiments/a3dizhu_v3/model.tar')
    parser.add_argument('--output', type=str, default='a3dizhu_model.onnx')
    parser.add_argument('--no_average', action='store_true',
                        help='Use agent 0 weights instead of averaging all 4')
    args = parser.parse_args()

    export(args.model_path, args.output, average_weights=not args.no_average)
