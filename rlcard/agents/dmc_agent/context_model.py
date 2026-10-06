# Copyright (c) 2026 hejuntt1014.
# Copyright 2021 RLCard Team of Texas A&M University.
# Copyright 2021 DouZero Team of Kwai.
# Modified for reusable state encoding and selectable history models.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0.
"""A3 policies with reusable state encoding and separate declaration decisions."""

import math

import torch
from torch import nn
from torch.nn import functional as F


OBS_STATIC_DIM = 556
HISTORY_LEN = 24
HIST_TOKEN_DIM = 88
STATE_DIM = OBS_STATIC_DIM + HISTORY_LEN * HIST_TOKEN_DIM
ACTION_DIM = 111
DECLARE_FLAG_INDEX = 525
CONTEXT_DIM = 512


class _ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.LayerNorm(width), nn.ReLU(),
            nn.Linear(width, width), nn.LayerNorm(width),
        )

    def forward(self, value):
        return F.relu(value + self.net(value))


class ContextDMCNet(nn.Module):
    """Encode each A3 state once, then score its legal candidate actions.

    ``encode_state`` returns ``(context, declaration_mask)``. ``score_encoded``
    accepts that pair and optional candidate-to-state indices. The flat forward
    interface remains suitable for training batches with one action per state.
    Auxiliary predictions depend only on the observation.
    """

    declaration_flag_index = DECLARE_FLAG_INDEX

    def __init__(self, state_shape, action_shape,
                 mlp_layers=(768, 768, 768, 768, 768),
                 history_encoder='mlp', aux_classes=(3, 3, 3, 4, 4),
                 history_steps=24):
        super().__init__()
        if math.prod(state_shape) != STATE_DIM or math.prod(action_shape) != ACTION_DIM:
            raise ValueError('ContextDMCNet requires A3 state/action shapes 2668/111')
        if not mlp_layers or any(width <= 0 for width in mlp_layers):
            raise ValueError('mlp_layers must contain positive widths')
        if len(set(mlp_layers)) != 1:
            raise ValueError('Residual block widths must be identical')
        if history_encoder not in ('mlp', 'transformer'):
            raise ValueError('history_encoder must be mlp or transformer')
        if type(history_steps) is not int or not 1 <= history_steps <= HISTORY_LEN:
            raise ValueError('history_steps must be an integer between 1 and 24')
        if any(count <= 0 for count in aux_classes):
            raise ValueError('Auxiliary class counts must be positive')
        self.history_encoder = history_encoder
        self.history_steps = history_steps
        self.aux_classes = tuple(aux_classes)
        hidden = mlp_layers[0]
        self.obs_static_proj = nn.Sequential(
            nn.Linear(OBS_STATIC_DIM, 256), nn.LayerNorm(256), nn.ReLU(),
        )
        if history_encoder == 'mlp':
            self.hist_encoder = nn.Sequential(
                nn.Linear(history_steps * HIST_TOKEN_DIM, 256),
                nn.LayerNorm(256), nn.ReLU(),
            )
        else:
            self.hist_token_proj = nn.Linear(HIST_TOKEN_DIM, 256)
            self.hist_pos_embed = nn.Parameter(torch.empty(1, history_steps, 256))
            nn.init.normal_(self.hist_pos_embed, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=256, nhead=8, dim_feedforward=512, dropout=0.0,
                batch_first=True, norm_first=True, activation='gelu',
            )
            self.hist_encoder = nn.TransformerEncoder(
                layer, num_layers=2, enable_nested_tensor=False,
            )
            self.hist_norm = nn.LayerNorm(256)
        self.context_proj = nn.Sequential(
            nn.Linear(512, CONTEXT_DIM), nn.LayerNorm(CONTEXT_DIM), nn.ReLU(),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(ACTION_DIM, 128), nn.LayerNorm(128), nn.ReLU(),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(CONTEXT_DIM + 128, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            *[_ResidualBlock(hidden) for _ in range(len(mlp_layers) - 1)]
        )
        self.output_head = nn.Linear(hidden, 1)
        self.declare_head = nn.Linear(CONTEXT_DIM, 2)
        self.aux_head = (nn.Linear(CONTEXT_DIM, sum(aux_classes))
                         if aux_classes else None)

    def encode_state(self, obs):
        obs = obs.flatten(1)
        static = obs[:, :OBS_STATIC_DIM]
        tokens = obs[:, OBS_STATIC_DIM:].reshape(-1, HISTORY_LEN, HIST_TOKEN_DIM)
        valid = tokens[..., 4] > 0.5
        if self.history_steps != HISTORY_LEN:
            # The native schema left-aligns valid history, padding its tail.
            # Select relative to the last valid token, not the padded buffer end,
            # so opening decisions retain their short histories as well.
            positions = torch.arange(HISTORY_LEN, device=tokens.device)
            last = torch.where(valid, positions + 1, 0).amax(dim=1)
            start = (last - self.history_steps).clamp_min(0)
            window = start[:, None] + positions[:self.history_steps]
            tokens = tokens.gather(1, window.unsqueeze(-1).expand(-1, -1, HIST_TOKEN_DIM))
            valid = valid.gather(1, window)
        # Invalid slots must not contribute, even if a caller leaves stale data.
        tokens = torch.where(valid.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        if self.history_encoder == 'mlp':
            history = self.hist_encoder(tokens.flatten(1))
        else:
            projected = self.hist_token_proj(tokens) + self.hist_pos_embed
            # All-masked attention produces undefined softmax values. A dummy
            # slot participates in attention for empty histories, then is removed
            # by the original validity mask during pooling.
            empty = ~valid.any(dim=1)
            padding = ~valid
            padding = padding.clone()
            padding[:, 0] = padding[:, 0] & ~empty
            encoded = self.hist_norm(self.hist_encoder(
                projected, src_key_padding_mask=padding,
            ))
            weights = valid.unsqueeze(-1).to(encoded.dtype)
            history = (encoded * weights).sum(1) / weights.sum(1).clamp_min(1)
        context = self.context_proj(torch.cat((self.obs_static_proj(static), history), 1))
        return context, static[:, DECLARE_FLAG_INDEX] > 0.5

    def _play_q(self, context, actions):
        fused = self.fusion_proj(torch.cat((context, self.action_proj(actions)), 1))
        return self.output_head(self.res_blocks(fused)).flatten()

    def score_encoded(self, encoded, actions, state_indices=None, declaration_mask=None,
                      phase=None):
        """Score actions, routing only relevant rows through each decision head.

        With ``state_indices``, each action selects its row from unique encoded
        states. A bare context tensor is accepted when its state-level declaration
        mask is provided explicitly. ``phase='play'`` or ``phase='declare'``
        lets a caller with a known homogeneous batch avoid device-side dynamic
        partitioning. The caller must group phases correctly when using it.
        """
        if phase not in (None, 'play', 'declare'):
            raise ValueError('phase must be play, declare, or None')
        if isinstance(encoded, tuple):
            context, mask = encoded
            if declaration_mask is not None:
                mask = declaration_mask
        else:
            context, mask = encoded, declaration_mask
            if mask is None:
                raise ValueError('A bare context requires declaration_mask')
        actions = actions.flatten(1)
        if state_indices is not None:
            context = context.index_select(0, state_indices)
            mask = mask.index_select(0, state_indices)
        if context.shape[0] != actions.shape[0] or mask.shape != (actions.shape[0],):
            raise ValueError('Each candidate action needs one context and declaration flag')
        if phase == 'play':
            return self._play_q(context, actions)
        if phase == 'declare':
            choice = (actions[:, :52].sum(1) > 26).long()
            return self.declare_head(context).gather(1, choice.unsqueeze(1)).flatten()
        mask = mask.bool()
        play_rows = (~mask).nonzero(as_tuple=True)[0]
        declare_rows = mask.nonzero(as_tuple=True)[0]
        result = context.new_zeros(actions.shape[0])
        if play_rows.numel():
            play_values = self._play_q(context[play_rows], actions[play_rows])
            result = result.index_copy(0, play_rows, play_values.to(result.dtype))
        if declare_rows.numel():
            declaration = self.declare_head(context[declare_rows])
            # A3's declaration action sets every card bit; pass sets none.
            choice = (actions[declare_rows, :52].sum(1) > 26).long()
            values = declaration.gather(1, choice.unsqueeze(1)).flatten()
            result = result.index_copy(0, declare_rows, values.to(result.dtype))
        return result

    def forward(self, obs, actions):
        return self.score_encoded(self.encode_state(obs), actions)

    def forward_with_aux(self, obs, actions):
        encoded = self.encode_state(obs)
        auxiliary = self.aux_head(encoded[0]) if self.aux_head is not None else None
        return self.score_encoded(encoded, actions), auxiliary

    def forward_with_aux_dense(self, obs, actions, phase=None):
        """Fixed-shape training graph without CUDA nonzero synchronization.

        For mixed batches both decision heads are evaluated; ``where`` sends gradients only to the
        selected head. This trades some unused play computation on declaration
        rows for a graph suitable for compilation and CUDA graph replay. Actor
        inference can still skip that computation with ``score_encoded`` phases.
        A known homogeneous ``phase`` preserves absent gradients on the unused
        head, including optimizer-state behavior.
        """
        context, declare = self.encode_state(obs)
        actions = actions.flatten(1)
        if phase is not None:
            values = self.score_encoded((context, declare), actions, phase=phase)
        else:
            play_q = self._play_q(context, actions)
            choice = (actions[:, :52].sum(1) > 26).long()
            declare_q = self.declare_head(context).gather(1, choice.unsqueeze(1)).flatten()
            values = torch.where(declare, declare_q, play_q)
        auxiliary = self.aux_head(context) if self.aux_head is not None else None
        return values, auxiliary


class ContextCompiledInference:
    """Dynamic candidate shapes with actor-local, in-place refreshed parameters."""
    declaration_flag_index = DECLARE_FLAG_INDEX

    def __init__(self, net):
        self.net = net
        options = dict(dynamic=True, fullgraph=True, options={'triton.cudagraphs': False})
        self.encode_state = torch.compile(net.encode_state, **options)
        self._play = torch.compile(net._play_q, **options)
        self._declare = torch.compile(self._declare_eager, **options)

    def _declare_eager(self, context, actions):
        choice = (actions[:, :52].sum(1) > 26).long()
        return self.net.declare_head(context).gather(1, choice.unsqueeze(1)).flatten()

    def score_encoded(self, encoded, actions, state_indices=None, phase=None):
        if phase not in ('play', 'declare'):
            return self.net.score_encoded(encoded, actions, state_indices, phase=phase)
        context = encoded[0]
        if state_indices is not None:
            context = context.index_select(0, state_indices)
        return self._play(context, actions) if phase == 'play' else self._declare(context, actions)


class ContextInferenceGraphs:
    """Bounded, lazy CUDA-graph cache for an actor-local context network.

    The network must stay in eval mode on one CUDA device. Parameters may be
    refreshed in place with ``load_state_dict`` between calls. Returned tensors
    alias graph-owned storage and must be consumed before the same graph replays.
    Small batches use powers of two; larger batches use multiples of 64 by
    default. Set ``bucket_multiple=0`` for powers of two throughout. Finer buckets
    reduce padded computation at the cost of more cached graphs. Independent rows make
    padded slots irrelevant. CPU and oversized batches use the eager network.
    """

    declaration_flag_index = DECLARE_FLAG_INDEX

    def __init__(self, net, dtype=None, max_graphs=12, max_rows=4096, bucket_multiple=64):
        if max_graphs < 1 or max_rows < 1 or bucket_multiple < 0:
            raise ValueError('Graph limits must be positive')
        self.net, self.dtype = net, dtype
        self.max_graphs, self.max_rows = max_graphs, max_rows
        self.bucket_multiple = bucket_multiple
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def _run(self, kind, inputs, function):
        if inputs[0].device.type != 'cuda':
            return function(*inputs)
        eager_function = function

        def function(*values):
            # Capture casts themselves: cached half weights would otherwise
            # become stale after the actor receives a new FP32 snapshot.
            with torch.autocast('cuda', enabled=self.dtype is not None,
                                dtype=self.dtype or torch.float16, cache_enabled=False):
                return eager_function(*values)

        if (self.net.training or torch.is_grad_enabled() or not len(inputs[0])
                or len(inputs[0]) > self.max_rows):
            return function(*inputs)
        rows = len(inputs[0])
        if self.bucket_multiple and rows > 128:
            bucket = ((rows + self.bucket_multiple - 1) // self.bucket_multiple) * self.bucket_multiple
        else:
            bucket = 1 << (rows - 1).bit_length()
        key = (kind, bucket, tuple((x.dtype, tuple(x.shape[1:])) for x in inputs))
        entry = self.cache.get(key)
        if entry is None:
            if len(self.cache) >= self.max_graphs:
                return function(*inputs)
            self.misses += 1
            buffers = tuple(torch.zeros((bucket, *x.shape[1:]), device=x.device, dtype=x.dtype)
                            for x in inputs)
            for buffer, value in zip(buffers, inputs):
                buffer[:rows].copy_(value)
            current = torch.cuda.current_stream(inputs[0].device)
            warmup = torch.cuda.Stream(device=inputs[0].device)
            warmup.wait_stream(current)
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    function(*buffers)
            current.wait_stream(warmup)
            with torch.cuda.device(inputs[0].device):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = function(*buffers)
            entry = (graph, buffers, output)
            self.cache[key] = entry
        else:
            self.hits += 1
        graph, buffers, output = entry
        for buffer, value in zip(buffers, inputs):
            buffer[:rows].copy_(value)
        graph.replay()
        if isinstance(output, tuple):
            return tuple(value[:rows] for value in output)
        return output[:rows]

    def encode_state(self, obs):
        return self._run('encode', (obs,), self.net.encode_state)

    def score_encoded(self, encoded, actions, state_indices=None, phase=None):
        if phase not in ('play', 'declare'):
            return self.net.score_encoded(encoded, actions, state_indices, phase=phase)
        context = encoded[0]
        if state_indices is not None:
            context = context.index_select(0, state_indices)
        if phase == 'play':
            return self._run('play', (context, actions), self.net._play_q)

        def declare(context, actions):
            choice = (actions[:, :52].sum(1) > 26).long()
            return self.net.declare_head(context).gather(1, choice.unsqueeze(1)).flatten()
        return self._run('declare', (context, actions), declare)
