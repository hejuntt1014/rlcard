"""State reuse, decision routing, and history masking for A3 context policies."""

import copy
import unittest
from unittest.mock import patch

import torch

from rlcard.agents.dmc_agent.context_model import (
    ContextDMCNet, DECLARE_FLAG_INDEX, HIST_TOKEN_DIM, HISTORY_LEN,
    OBS_STATIC_DIM, STATE_DIM,
)


class TestContextDMCNet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(17)

    def model(self, encoder='mlp', aux_classes=(3, 3, 3, 4, 4)):
        return ContextDMCNet([STATE_DIM], [111], [32, 32], encoder, aux_classes)

    def observations(self, count=3):
        observations = torch.zeros(count, STATE_DIM)
        observations[:, :52] = torch.randint(0, 2, (count, 52)).float()
        history = observations[:, OBS_STATIC_DIM:].view(count, HISTORY_LEN, HIST_TOKEN_DIM)
        history[:, :3] = torch.randn(count, 3, HIST_TOKEN_DIM)
        history[:, :3, 4] = 1
        return observations

    def test_reuse_matches_flat_outputs_and_gradients(self):
        obs = self.observations()
        obs[1, DECLARE_FLAG_INDEX] = 1
        indices = torch.tensor([0, 0, 1, 1, 2, 2, 2])
        actions = torch.randn(7, 111)
        actions[2, :52], actions[3, :52] = 0, 1
        for encoder in ('mlp', 'transformer'):
            with self.subTest(encoder=encoder):
                flat_model = self.model(encoder).eval()
                reused_model = copy.deepcopy(flat_model)
                expected = flat_model(obs[indices], actions)
                encoded = reused_model.encode_state(obs)
                actual = reused_model.score_encoded(encoded, actions, indices)
                torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
                expected.square().sum().backward()
                actual.square().sum().backward()
                for (name, left), (_, right) in zip(
                        flat_model.named_parameters(), reused_model.named_parameters()):
                    if left.grad is None:
                        self.assertIsNone(right.grad, name)
                    else:
                        torch.testing.assert_close(left.grad, right.grad, atol=2e-5, rtol=1e-4,
                                                   msg=name)

    def test_history_order_changes_context(self):
        for encoder in ('mlp', 'transformer'):
            with self.subTest(encoder=encoder):
                net = self.model(encoder).eval()
                original = self.observations(1)
                reordered = original.clone()
                history = reordered[:, OBS_STATIC_DIM:].view(1, HISTORY_LEN, HIST_TOKEN_DIM)
                history[:, [0, 1]] = history[:, [1, 0]].clone()
                before = net.encode_state(original)[0]
                after = net.encode_state(reordered)[0]
                self.assertGreater((before - after).abs().max().item(), 1e-5)

    def test_padding_content_does_not_change_outputs(self):
        for encoder in ('mlp', 'transformer'):
            with self.subTest(encoder=encoder):
                net = self.model(encoder).eval()
                obs = self.observations()
                dirty = obs.clone()
                history = dirty[:, OBS_STATIC_DIM:].view(-1, HISTORY_LEN, HIST_TOKEN_DIM)
                history[:, 3:] = float('nan')
                history[:, 3:, 4] = 0
                torch.testing.assert_close(net.encode_state(obs)[0], net.encode_state(dirty)[0])

    def test_empty_history_is_finite_and_differentiable(self):
        for encoder in ('mlp', 'transformer'):
            with self.subTest(encoder=encoder):
                net = self.model(encoder)
                q, auxiliary = net.forward_with_aux(torch.zeros(2, STATE_DIM), torch.zeros(2, 111))
                loss = q.square().sum() + auxiliary.square().mean()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for parameter in net.parameters():
                    if parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_declaration_skips_play_and_auxiliary_inference(self):
        net = self.model().eval()
        obs = self.observations(2)
        obs[:, DECLARE_FLAG_INDEX] = 1
        actions = torch.zeros(2, 111)
        actions[1, :52] = 1
        encoded = net.encode_state(obs)
        expected = net.declare_head(encoded[0])[torch.arange(2), torch.tensor([0, 1])]
        with patch.object(net, '_play_q', side_effect=AssertionError('play branch called')), \
                patch.object(net.aux_head, 'forward', side_effect=AssertionError('aux branch called')):
            torch.testing.assert_close(net(obs, actions), expected)

    def test_play_skips_declaration_and_aux_is_action_independent(self):
        net = self.model().eval()
        obs = self.observations(2)
        with patch.object(net.declare_head, 'forward', side_effect=AssertionError('declare branch called')):
            _, first = net.forward_with_aux(obs, torch.zeros(2, 111))
            _, second = net.forward_with_aux(obs, torch.ones(2, 111))
        torch.testing.assert_close(first, second)
        self.assertEqual(first.shape, (2, 17))

    def test_no_auxiliary_head_and_explicit_mask(self):
        net = self.model(aux_classes=()).eval()
        obs, actions = self.observations(2), torch.zeros(2, 111)
        context, mask = net.encode_state(obs)
        actual = net.score_encoded(context, actions, declaration_mask=mask)
        expected, auxiliary = net.forward_with_aux(obs, actions)
        torch.testing.assert_close(actual, expected)
        self.assertIsNone(auxiliary)
        with self.assertRaisesRegex(ValueError, 'declaration_mask'):
            net.score_encoded(context, actions)

    def test_rejects_incompatible_schema(self):
        with self.assertRaisesRegex(ValueError, '2668/111'):
            ContextDMCNet([850], [52])
        with self.assertRaisesRegex(ValueError, 'identical'):
            ContextDMCNet([STATE_DIM], [111], [32, 16])

    def test_known_phase_matches_automatic_outputs_and_gradients(self):
        for phase in ('play', 'declare'):
            with self.subTest(phase=phase):
                obs, actions = self.observations(2), torch.randn(4, 111)
                obs[:, DECLARE_FLAG_INDEX] = float(phase == 'declare')
                actions[::2, :52], actions[1::2, :52] = 0, 1
                indices = torch.tensor([0, 0, 1, 1])
                automatic = self.model().eval()
                known = copy.deepcopy(automatic)
                expected = automatic.score_encoded(automatic.encode_state(obs), actions, indices)
                # Known homogeneous batches never need a device-side partition.
                with patch.object(torch.Tensor, 'nonzero', side_effect=AssertionError('dynamic partition')):
                    actual = known.score_encoded(known.encode_state(obs), actions, indices, phase=phase)
                torch.testing.assert_close(actual, expected)
                actual.square().sum().backward()
                expected.square().sum().backward()
                for (name, left), (_, right) in zip(automatic.named_parameters(), known.named_parameters()):
                    if left.grad is None:
                        self.assertIsNone(right.grad, name)
                    else:
                        torch.testing.assert_close(left.grad, right.grad, msg=name)

    def test_unknown_phase_is_rejected(self):
        net = self.model()
        with self.assertRaisesRegex(ValueError, 'phase must be'):
            net.score_encoded(net.encode_state(self.observations(1)), torch.zeros(1, 111), phase='other')


if __name__ == '__main__':
    unittest.main()
