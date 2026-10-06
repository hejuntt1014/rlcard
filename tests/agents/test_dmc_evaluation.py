"""Fixed-deal evaluation uses game rewards and seed-clustered comparisons."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import rlcard
from examples.evaluate_dmc import (EvaluationActionError, evaluate, load_policy,
                                   paired_difference, play_game, summarize)
from rlcard.agents.dmc_agent.model import DMCModel


class FakeGame:
    num_players = 4

    def __init__(self):
        self._engine = SimpleNamespace(get_rule_agent_action=lambda: 'rule')

    def seed(self, seed):
        self.deal = seed

    def state(self):
        return {'obs': np.zeros(850), 'legal_actions': {'model': None, 'rule': None}}

    def reset(self):
        self.turn = 0
        self.actions = []
        return self.state(), 0

    def is_over(self):
        return self.turn == 4

    def step(self, action):
        self.actions.append(action)
        self.turn += 1
        return self.state(), self.turn % 4

    def get_payoffs(self):
        return [(.25 if action == 'model' else -.25) for action in self.actions]

    def get_training_payoffs(self):
        raise AssertionError('Evaluation must not use shaped training returns')


class TestDMCEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_model_controls_only_selected_seat_and_uses_original_payoff(self):
        model = SimpleNamespace(get_agent=lambda p: SimpleNamespace(eval_step=lambda s: ('model', {})))
        env = FakeGame()
        env._engine.get_mode = lambda: 'solo'
        records = evaluate(model, env, [31, 32])
        self.assertEqual([(r['seed'], r['seat']) for r in records],
                         [(s, p) for s in (31, 32) for p in range(4)])
        self.assertTrue(all(r['payoff'] == .25 and r['outcome'] == 'win' for r in records))
        self.assertTrue(all(r['model_decisions'] == 1 for r in records))
        self.assertTrue(all(r['mode'] == 'solo' for r in records))
        self.assertEqual(env.actions, ['rule', 'rule', 'rule', 'model'])
        with self.assertRaisesRegex(RuntimeError, 'max_steps'):
            play_game(env, model, 31, 0, max_steps=1)

    def test_seed_clustering_and_paired_comparisons(self):
        records = [dict(seed=seed, seat=seat, payoff=score, mode='normal', outcome='win')
                   for seed, score in ((1, 1.), (2, 3.)) for seat in range(4)]
        summary = summarize(records)
        self.assertEqual(summary['mean'], 2.)
        self.assertEqual(summary['standard_error'], 1.)
        candidate = [dict(r, payoff=r['payoff'] + .25) for r in records]
        delta = paired_difference(records, candidate)
        self.assertEqual(delta['mean'], .25)
        self.assertEqual(delta['standard_error'], 0.)
        with self.assertRaisesRegex(ValueError, 'unique'):
            paired_difference(records, candidate[:-1])
        self.assertIsNone(summarize(records[:4])['standard_error'])

    def test_illegal_actions_preserve_replay_history_and_do_not_fallback(self):
        model = SimpleNamespace(get_agent=lambda p: SimpleNamespace(eval_step=lambda s: ('model', {})))
        env = FakeGame()
        env._engine.get_rule_agent_action = lambda: 'illegal'
        with self.assertRaises(EvaluationActionError) as raised:
            play_game(env, model, 31, 0)
        detail = raised.exception.details
        self.assertEqual(detail['history'], [[0, 'model']])
        self.assertEqual(detail['policy'], 'native greedy')
        self.assertEqual(detail['seed'], 31)
        self.assertEqual(env.actions, ['model'])

    def test_schema_and_shared_weights_are_validated(self):
        env = SimpleNamespace(name='test', num_players=2, state_shape=[[2], [2]],
                              action_shape=[[2], [2]], feature_schema='features-v2')
        spec = dict(state_shape=env.state_shape, action_shape=env.action_shape,
                    mlp_layers=[4], share_weights=True, architecture='mlp', aux_classes=[],
                    env_name='test', feature_schema='features-v2')
        model = DMCModel([[2], [2]], [[2], [2]], [4], device='cpu', share_weights=True)
        payload = dict(model_spec=spec, model_state_dict=[a.state_dict() for a in model.get_agents()], frames=10)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.tar'
            torch.save(payload, path)
            loaded, metadata = load_policy(path, env)
            self.assertEqual(metadata['training_frames'], 10)
            self.assertIs(loaded.get_agent(0), loaded.get_agent(1))
            env.feature_schema = 'features-v3'
            with self.assertRaisesRegex(ValueError, 'schema'):
                load_policy(path, env)
            env.feature_schema = 'features-v2'
            payload['model_state_dict'][1] = {k: v.clone() for k, v in payload['model_state_dict'][1].items()}
            key = next(iter(payload['model_state_dict'][1]))
            payload['model_state_dict'][1][key].add_(1)
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'inconsistent'):
                load_policy(path, env)

    @unittest.skipUnless(importlib.util.find_spec('a3dizhu_cpp'), 'Optional native A3 extension')
    def test_native_fixed_seed_deals_are_reproducible(self):
        # This initialization exposed an illegal greedy pass at seed 29, seat 0,
        # turn 68 (only heart_10 legal). Preserve the failing case as a regression.
        torch.manual_seed(18)
        env = rlcard.make('a3dizhu', config={'greedy_ratio': 1., 'random_ratio': 0.})
        model = DMCModel(env.state_shape, env.action_shape, [8], device='cpu')
        model.eval()
        first = evaluate(model, env, [29], max_steps=1000)
        second = evaluate(model, env, [29], max_steps=1000)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)

    @unittest.skipUnless(importlib.util.find_spec('a3dizhu_v12_cpp'), 'Optional native V12 extension')
    def test_v12_checkpoint_load_and_fixed_rule_evaluation(self):
        from rlcard.agents.dmc_agent.trainer import DMCTrainer
        torch.manual_seed(17)
        rules = dict(declare_require_both_spades=True, straight_start_val=2, straight_end_val=11)
        env = rlcard.make('a3dizhu-v12', config={'greedy_ratio': 1., 'rules': rules})
        trainer = DMCTrainer(env, mlp_layers=[16, 16], share_weights=True)
        model = trainer.model_func('cpu')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.tar'
            torch.save(dict(model_spec=trainer._model_spec(),
                            model_state_dict=[a.state_dict() for a in model.get_agents()], frames=0), path)
            loaded, _ = load_policy(path, env)
            records = evaluate(loaded, env, [29], max_steps=1000)
        self.assertEqual(len(records), 4)
        self.assertTrue(all(record['rules'] == rules for record in records))
        self.assertTrue(all(record['mode'] in ('normal', 'solo', 'declared') for record in records))


if __name__ == '__main__':
    unittest.main()
