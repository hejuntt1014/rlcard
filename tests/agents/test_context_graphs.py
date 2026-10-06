"""CUDA actor graph storage, parameter refresh, and fallback contracts."""
import contextlib
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from rlcard.agents.dmc_agent.collector import score_action_groups
from rlcard.agents.dmc_agent.context_model import (
    ContextDMCNet, ContextInferenceGraphs, DECLARE_FLAG_INDEX, STATE_DIM,
)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA graph integration test')
class TestContextInferenceGraphs(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(27)
        self.rng = np.random.default_rng(27)

    def model(self):
        return ContextDMCNet([STATE_DIM], [111], [64, 64]).cuda().eval()

    def fixture(self, states=13, candidates=17):
        obs = self.rng.integers(0, 2, (states, STATE_DIM), dtype=np.uint8)
        obs[:, DECLARE_FLAG_INDEX] = 0
        actions = self.rng.integers(0, 2, (states * candidates, 111), dtype=np.uint8)
        offsets = np.arange(states + 1) * candidates
        return obs, actions, offsets

    def score(self, net, data, dtype=None, runner=None, chunk=4096):
        agent = SimpleNamespace(net=net, device='cuda:0', inference_dtype=dtype)
        if runner is not None:
            agent.inference_runner = runner
        return score_action_groups(agent, *data, chunk, contextlib.nullcontext())

    def test_repeated_chunk_bucket_does_not_overwrite_earlier_scores(self):
        net = self.model()
        data = self.fixture()
        runner = ContextInferenceGraphs(net)
        expected = self.score(net, data, chunk=64)
        actual = self.score(net, data, runner=runner, chunk=64)
        # Three full chunks replay the same graph storage before concatenation.
        self.assertGreaterEqual(runner.hits, 2)
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA BF16 support')
    def test_fp32_snapshots_refresh_captured_linear_weights(self):
        for precast in (False, True):
            with self.subTest(precast_linear=precast):
                net = self.model()
                if precast:
                    for module in net.modules():
                        if isinstance(module, torch.nn.Linear):
                            module.to(dtype=torch.bfloat16)
                data = self.fixture(5, 7)
                runner = ContextInferenceGraphs(net, dtype=torch.bfloat16)
                before = self.score(net, data, torch.bfloat16, runner)
                snapshot = {name: value.float().clone() for name, value in net.state_dict().items()}
                snapshot['output_head.weight'].mul_(1.5)
                snapshot['output_head.bias'].add_(1.)
                snapshot['obs_static_proj.0.weight'].mul_(1.1)
                net.load_state_dict(snapshot)
                expected = self.score(net, data, torch.bfloat16)
                actual = self.score(net, data, torch.bfloat16, runner)
                self.assertGreater(float(np.max(np.abs(actual - before))), .2)
                np.testing.assert_allclose(actual, expected, rtol=.04, atol=.04)

    def test_empty_history_mixed_heads_and_bucket_switching(self):
        net = self.model()
        runner = ContextInferenceGraphs(net)
        for states in (5, 19, 3, 5):
            with self.subTest(states=states):
                obs, actions, offsets = self.fixture(states, 9)
                obs[:, 556:] = 0
                obs[::2, DECLARE_FLAG_INDEX] = 1
                for row in range(0, states, 2):
                    actions[offsets[row]:offsets[row + 1], :52] = 1
                data = obs, actions, offsets
                expected = self.score(net, data)
                actual = self.score(net, data, runner=runner)
                self.assertTrue(np.isfinite(actual).all())
                np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=2e-5)

    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'CUDA BF16 support')
    def test_cache_limit_and_oversized_fallback_keep_autocast_precision(self):
        net = self.model()
        runner = ContextInferenceGraphs(net, dtype=torch.bfloat16, max_graphs=1, max_rows=8)
        obs = torch.zeros(4, STATE_DIM, device='cuda')
        actions = torch.zeros(4, 111, device='cuda')
        # No caller autocast: the runner must preserve its own precision even
        # when the only graph slot is occupied by the state encoder.
        with torch.inference_mode():
            encoded = runner.encode_state(obs)
            actual = runner.score_encoded(encoded, actions, phase='play')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                expected = net.score_encoded(net.encode_state(obs), actions, phase='play')
            self.assertEqual(actual.dtype, torch.bfloat16)
            torch.testing.assert_close(actual, expected, atol=.04, rtol=.04)
            # Autocast LayerNorm output dtype varies across PyTorch versions;
            # verify the actual AMP context and the eager reference instead.
            flags = []
            hook = net.obs_static_proj[0].register_forward_pre_hook(
                lambda *_: flags.append(torch.is_autocast_enabled('cuda')))
            large_obs = torch.zeros(9, STATE_DIM, device='cuda')
            oversized = runner.encode_state(large_obs)
            hook.remove()
            self.assertEqual(flags, [True])
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                large_expected = net.encode_state(large_obs)
            self.assertEqual(oversized[0].dtype, large_expected[0].dtype)
            torch.testing.assert_close(oversized[0], large_expected[0], atol=.04, rtol=.04)
            self.assertEqual(len(runner.cache), 1)


if __name__ == '__main__':
    unittest.main()
