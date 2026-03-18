"""
将 DMC 训练好的模型导出为 ONNX 格式

支持两种架构：
  - 旧版：纯 MLP (fc_layers)
  - 新版：残差网络 (input_proj + res_blocks + output_head)

自动检测 checkpoint 中的架构版本。

ONNX 模型接口：
  输入: obs_action  shape=[N, 902]  (850维obs + 52维action)
  输出: q_value     shape=[N]       (Q值，越大越好)

状态维度变更历史：
  v4: 700维 obs (6步历史，无玩家ID，无is_solo)
  v5: 849维 obs (8步历史含玩家ID，相对位置编码，is_solo，BUG修复)
  v6: 850维 obs (新增 is_declaration_phase 位，支持报牌阶段学习)
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ResBlockForExport(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class DMCNetForExport(nn.Module):
    """导出用的网络，把 obs+action 合并为单一输入"""

    def __init__(self, state_dim, action_dim, mlp_layers=[512, 512, 512, 512, 512], use_resnet=True):
        super().__init__()
        input_dim = state_dim + action_dim
        self.use_resnet = use_resnet
        hidden = mlp_layers[0]

        if use_resnet:
            self.input_proj = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
            )
            self.res_blocks = nn.Sequential(
                *[ResBlockForExport(hidden) for _ in range(len(mlp_layers) - 1)]
            )
            self.output_head = nn.Linear(hidden, 1)
        else:
            layer_dims = [input_dim] + mlp_layers
            fc = []
            for i in range(len(layer_dims)-1):
                fc.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
                fc.append(nn.ReLU())
            fc.append(nn.Linear(layer_dims[-1], 1))
            self.fc_layers = nn.Sequential(*fc)

    def forward(self, obs_action):
        if self.use_resnet:
            x = self.input_proj(obs_action)
            x = self.res_blocks(x)
            return self.output_head(x).flatten()
        else:
            return self.fc_layers(obs_action).flatten()


def detect_architecture(state_dict):
    """自动检测 checkpoint 的网络架构"""
    has_resblock = any('res_blocks' in k or 'input_proj' in k for k in state_dict.keys())
    return 'resnet' if has_resblock else 'mlp'


def export(model_path, output_path, state_dim=850, action_dim=52, average_weights=True):
    print(f'Loading model from {model_path}')
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    state_dicts = checkpoint['model_state_dict']
    frames = checkpoint.get('frames', 0)
    shared = checkpoint.get('share_weights', False)
    print(f'  Trained frames: {frames:,}')
    print(f'  Number of agents: {len(state_dicts)}')
    print(f'  Shared weights: {shared}')

    arch = detect_architecture(state_dicts[0])
    print(f'  Architecture: {arch}')

    use_resnet = (arch == 'resnet')
    export_net = DMCNetForExport(state_dim, action_dim, use_resnet=use_resnet)

    if shared or not average_weights:
        export_net.load_state_dict(state_dicts[0])
        print('  Using agent 0 weights (shared model)')
    else:
        avg_state = {}
        for key in state_dicts[0]:
            tensors = [state_dicts[i][key].float() for i in range(len(state_dicts))]
            avg_state[key] = torch.mean(torch.stack(tensors), dim=0)
        export_net.load_state_dict(avg_state)
        print('  Using averaged weights from all agents')

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

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        test_input = np.random.randn(5, state_dim + action_dim).astype(np.float32)
        result = session.run(None, {'obs_action': test_input})
        print(f'ONNX verification: input {test_input.shape} -> output {result[0].shape}')
        print(f'Sample Q values: {result[0]}')
        print('ONNX export verified OK!')
    except ImportError:
        print('onnxruntime not installed, skipping verification')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Export DMC model to ONNX')
    parser.add_argument('--model_path', type=str,
                        default='experiments/a3dizhu_v6/model.tar')
    parser.add_argument('--output', type=str, default='a3dizhu_model.onnx')
    parser.add_argument('--no_average', action='store_true',
                        help='Use agent 0 weights instead of averaging')
    args = parser.parse_args()
    export(args.model_path, args.output, average_weights=not args.no_average)
