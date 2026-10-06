# Copyright 2021 RLCard Team of Texas A&M University
# Copyright 2021 DouZero Team of Kwai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """残差块：两层 Linear + LayerNorm，带跳跃连接"""
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


class DMCNet(nn.Module):
    def __init__(
        self,
        state_shape,
        action_shape,
        mlp_layers=[512, 512, 512, 512, 512],
        architecture='mlp',
        aux_classes=(),
    ):
        super().__init__()
        input_dim = int(np.prod(state_shape) + np.prod(action_shape))
        if not mlp_layers or any(d <= 0 for d in mlp_layers):
            raise ValueError('mlp_layers must contain positive widths')
        if architecture not in ('mlp', 'resnet'):
            raise ValueError('architecture must be mlp or resnet')
        self.architecture = architecture
        self.aux_classes = tuple(aux_classes)
        if architecture == 'resnet' and len(set(mlp_layers)) != 1:
            raise ValueError('Residual block widths must be identical')
        if any(n <= 0 for n in aux_classes):
            raise ValueError('Auxiliary class counts must be positive')
        if architecture == 'mlp':
            layers = []
            for left, right in zip([input_dim] + list(mlp_layers), mlp_layers):
                layers.extend([nn.Linear(left, right), nn.ReLU()])
            layers.append(nn.Linear(mlp_layers[-1], 1))
            self.fc_layers = nn.Sequential(*layers)
            if aux_classes:
                raise ValueError('Auxiliary heads require architecture=resnet')
            return
        hidden = mlp_layers[0]

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden) for _ in range(len(mlp_layers) - 1)]
        )

        self.output_head = nn.Linear(hidden, 1)

        # 辅助任务头：预测其他 3 个玩家的队伍（相对位置）
        # 每人 3 类: SPADE_A3=0, OPPONENT=1, SOLO=2 → 共 9 个 logits
        self.aux_head = nn.Linear(hidden, sum(aux_classes)) if aux_classes else None

    def _backbone(self, obs, actions):
        obs = torch.flatten(obs, 1)
        actions = torch.flatten(actions, 1)
        x = torch.cat((obs, actions), dim=1)
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return x

    def forward(self, obs, actions):
        """Q 值输出（推理 / ONNX 导出用，不走 aux_head）"""
        if self.architecture == 'mlp':
            return self.fc_layers(torch.cat((obs.flatten(1), actions.flatten(1)), 1)).flatten()
        x = self._backbone(obs, actions)
        return self.output_head(x).flatten()

    def forward_with_aux(self, obs, actions):
        """训练用：同时返回 Q 值和辅助预测 logits"""
        if self.architecture == 'mlp':
            return self.forward(obs, actions), None
        x = self._backbone(obs, actions)
        q = self.output_head(x).flatten()
        aux = self.aux_head(x) if self.aux_head is not None else None
        return q, aux


class DMCAgent:
    def __init__(
        self,
        state_shape,
        action_shape,
        mlp_layers=[512, 512, 512, 512, 512],
        exp_epsilon=0.01,
        device="0",
        architecture='mlp',
        aux_classes=(),
    ):
        self.use_raw = False
        self.device = 'cuda:' + device if device != "cpu" else "cpu"
        self.net = DMCNet(state_shape, action_shape, mlp_layers, architecture, aux_classes).to(self.device)
        self.exp_epsilon = exp_epsilon
        self.action_shape = action_shape

    def step(self, state):
        action_keys, values = self.predict(state)
        if self.exp_epsilon > 0 and np.random.rand() < self.exp_epsilon:
            action = np.random.choice(action_keys)
        else:
            action_idx = np.argmax(values)
            action = action_keys[action_idx]
        return action

    def eval_step(self, state):
        action_keys, values = self.predict(state)
        action_idx = np.argmax(values)
        action = action_keys[action_idx]
        info = {}
        labels = state.get('raw_legal_actions', action_keys)
        info['values'] = {labels[i]: float(values[i]) for i in range(len(action_keys))}
        return action, info

    def share_memory(self):
        self.net.share_memory()

    def eval(self):
        self.net.eval()

    def parameters(self):
        return self.net.parameters()

    @torch.no_grad()
    def predict(self, state):
        obs = state['obs'].astype(np.float32)
        legal_actions = state['legal_actions']
        action_keys = np.array(list(legal_actions.keys()))
        action_values = list(legal_actions.values())
        for i in range(len(action_values)):
            if action_values[i] is None:
                action_values[i] = np.zeros(self.action_shape[0])
                action_values[i][action_keys[i]] = 1
        action_values = np.array(action_values, dtype=np.float32)
        obs = np.repeat(obs[np.newaxis, :], len(action_keys), axis=0)
        values = self.net.forward(torch.from_numpy(obs).to(self.device),
                                  torch.from_numpy(action_values).to(self.device))
        return action_keys, values.cpu().detach().numpy()

    def forward(self, obs, actions):
        return self.net.forward(obs, actions)

    def forward_with_aux(self, obs, actions):
        return self.net.forward_with_aux(obs, actions)

    def load_state_dict(self, state_dict):
        return self.net.load_state_dict(state_dict)

    def state_dict(self):
        return self.net.state_dict()

    def set_device(self, device):
        self.device = device


class DMCModel:
    """Per-role policies, with explicit optional weight sharing."""
    def __init__(
        self,
        state_shape,
        action_shape,
        mlp_layers=[512, 512, 512, 512, 512],
        exp_epsilon=0.01,
        device=0,
        share_weights=False,
        architecture='mlp',
        aux_classes=(),
    ):
        self.shared = share_weights
        num_players = len(state_shape)
        if share_weights and (any(tuple(s) != tuple(state_shape[0]) for s in state_shape)
                              or any(tuple(s) != tuple(action_shape[0]) for s in action_shape)):
            raise ValueError('Shared policies require identical observation and action shapes')

        if share_weights:
            agent = DMCAgent(
                state_shape[0], action_shape[0],
                mlp_layers, exp_epsilon, str(device),
                architecture, aux_classes,
            )
            self.agents = [agent for _ in range(num_players)]
        else:
            self.agents = []
            for pid in range(num_players):
                self.agents.append(DMCAgent(
                    state_shape[pid], action_shape[pid],
                    mlp_layers, exp_epsilon, str(device),
                    architecture, aux_classes,
                ))

    def share_memory(self):
        if self.shared:
            self.agents[0].share_memory()
        else:
            for agent in self.agents:
                agent.share_memory()

    def eval(self):
        if self.shared:
            self.agents[0].eval()
        else:
            for agent in self.agents:
                agent.eval()

    def parameters(self, index):
        return self.agents[index].parameters()

    def get_agent(self, index):
        return self.agents[index]

    def get_agents(self):
        return self.agents
