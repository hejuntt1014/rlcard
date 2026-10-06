"""Native rich-feature regressions and per-decision transport invariants."""
import importlib.util
import random
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec('a3dizhu_v12_cpp'), 'Optional rich A3 extension')
class TestA3RichNative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import a3dizhu_v12_cpp
        cls.native = a3dizhu_v12_cpp

    def test_afterstate_full_house_requires_distinct_pair_rank(self):
        triple = ['diamond_4', 'club_4', 'heart_4']
        invalid = self.native.compute_afterstate(triple + ['diamond_5', 'diamond_7'])
        self.assertFalse(invalid['has_threepair_or_fourone_potential'])
        two_triples = self.native.compute_afterstate(triple + ['diamond_5', 'club_5', 'heart_5'])
        self.assertTrue(two_triples['has_threepair_or_fourone_potential'])
        four_one = self.native.compute_afterstate(triple + ['spade_4', 'diamond_5'])
        self.assertTrue(four_one['has_threepair_or_fourone_potential'])

    def test_afterstate_checks_later_suits_for_straight_flush(self):
        first_flush = ['diamond_' + rank for rank in ('4', '6', '8', '10', 'Q')]
        later_straight_flush = ['club_' + rank for rank in ('4', '5', '6', '7', '8')]
        info = self.native.compute_afterstate(first_flush + later_straight_flush)
        self.assertTrue(info['has_sf_potential'])

    def test_afterstate_straight_respects_rule_limits(self):
        low = ['club_' + rank for rank in ('3', '4', '5', '6', '7')]
        high = ['club_' + rank for rank in ('10', 'J', 'Q', 'K', 'A')]
        for hand, start, end in ((low, 2, 12), (high, 1, 11)):
            self.assertTrue(self.native.compute_afterstate(hand)['has_sf_potential'])
            self.assertFalse(self.native.compute_afterstate(hand, start, end)['has_sf_potential'])

    def test_compact_batch_matches_per_environment_observation_and_actions(self):
        vector = self.native.VectorizedEngine(3)
        vector.seed(127)
        for i in range(3):
            vector.reset(i)
            vector.set_rules(i, False, 1, 12)
        # Include unequal candidate counts after progressing beyond declarations.
        for _ in range(7):
            raw, actions, offsets, keys = vector.prepare_batch([2, 0, 1])
            self.assertEqual(raw.shape, (3, 2668))
            self.assertEqual(actions.shape, (offsets[-1], 111))
            self.assertEqual(raw.dtype, np.int8)
            self.assertEqual(len(offsets), 4)
            for row, env_id in enumerate((2, 0, 1)):
                np.testing.assert_array_equal(raw[row], vector.encode_obs(env_id, vector.get_player_id(env_id)))
                self.assertEqual(offsets[row + 1] - offsets[row], len(keys[row]))
                for j, key in enumerate(keys[row]):
                    np.testing.assert_array_equal(actions[offsets[row] + j], vector.get_action_feature(env_id, key))
                choice = 'pass' if vector.is_declaration_phase(env_id) else keys[row][0]
                vector.step(env_id, choice)
        raw, actions, offsets, keys = vector.prepare_batch([])
        self.assertEqual(raw.shape, (0, 2668))
        self.assertEqual(actions.shape, (0, 111))
        self.assertEqual(offsets, [0])
        self.assertEqual(keys, [])

    def test_rule_rollouts_record_real_actions_and_predecision_labels(self):
        for seed in (42, 101):
            with self.subTest(seed=seed):
                single = self.native.CppEngine()
                vector = self.native.VectorizedEngine(1)
                for engine in (single, vector):
                    engine.seed(seed)
                    engine.set_greedy_ratio(1.)
                single.reset()
                vector.reset(0)
                done, observations, players, actions, auxiliary = vector.advance_to_decision_with_data(0)
                self.assertTrue(done)
                counts = [0] * 4
                for obs, player, action, aux in zip(observations, players, actions, auxiliary):
                    self.assertEqual(player, single.get_player_id())
                    np.testing.assert_array_equal(obs, single.encode_obs(player))
                    key = single.get_rule_agent_action()
                    np.testing.assert_array_equal(action, single.get_action_feature(key))
                    np.testing.assert_array_equal(aux, single.get_aux_targets()[player])
                    single.step(key)
                    counts[player] += 1
                self.assertTrue(single.is_over())
                self.assertTrue(np.any(actions[:, :52]))
                np.testing.assert_array_equal(single.get_training_payoffs(), vector.get_training_payoffs(0))
                np.testing.assert_array_equal(single.get_payoffs(), vector.get_payoffs(0))
                for player in range(4):
                    rewards = vector.get_step_rewards(0, player)
                    self.assertEqual(len(rewards), counts[player])
                    np.testing.assert_array_equal(single.get_step_rewards(player), rewards)

    def test_old_and_rich_modules_can_coexist(self):
        if importlib.util.find_spec('a3dizhu_cpp') is None:
            self.skipTest('Optional classic A3 extension')
        import a3dizhu_cpp
        classic, rich = a3dizhu_cpp.CppEngine(), self.native.CppEngine()
        classic.reset()
        rich.reset()
        self.assertEqual(classic.encode_obs(classic.get_player_id()).shape, (850,))
        self.assertEqual(rich.encode_obs(rich.get_player_id()).shape, (2668,))
        self.assertEqual(self.native.FEATURE_VERSION, 'a3-v12-lite-v1')

    def test_hidden_opponent_cards_do_not_change_public_features(self):
        ranks = ('4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A', '2', '3')
        hands = [[suit + '_' + rank for rank in ranks]
                 for suit in ('diamond', 'club', 'heart', 'spade')]
        swapped = [list(hand) for hand in hands]
        # Change a hidden solo allocation into split A/3 ownership, keeping
        # player 0's hand and every publicly visible card count identical.
        swapped[1][0], swapped[3][-1] = swapped[3][-1], swapped[1][0]
        first, second = self.native.CppEngine(), self.native.CppEngine()
        first.reset_with_hands(hands, 0)
        second.reset_with_hands(swapped, 0)
        self.assertEqual(first.get_mode(), 'solo')
        self.assertEqual(second.get_mode(), 'normal')
        for _ in range(4):
            first.step('pass')
            second.step('pass')
        self.assertEqual(first.get_player_id(), 0)
        self.assertEqual(second.get_player_id(), 0)
        np.testing.assert_array_equal(first.encode_obs(0), second.encode_obs(0))
        a, b = first.get_legal_actions(), second.get_legal_actions()
        self.assertEqual(set(a), set(b))
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])

    def test_declared_shaping_uses_both_public_teammates(self):
        engine = self.native.CppEngine()
        engine.reset_with_hands([
            ['spade_4', 'spade_8', 'spade_9'],
            ['diamond_4', 'diamond_8', 'diamond_9'],
            ['club_4', 'club_8', 'club_9'],
            ['heart_4', 'heart_8'],
        ], 1)
        engine.set_rules(False, 1, 12)
        engine.step('declare')  # Player 0 becomes declarant; 1, 2, 3 cooperate.
        self.assertEqual(engine.get_mode(), 'declared')
        sequence = [(1, 'diamond_4'), (2, 'pass'), (3, 'heart_4'),
                    (0, 'spade_4'), (1, 'diamond_8'), (2, 'club_8'),
                    (3, 'heart_8')]
        for player, key in sequence:
            self.assertEqual(engine.get_player_id(), player)
            self.assertIn(key, engine.get_legal_actions())
            engine.step(key)
        self.assertTrue(engine.is_over())
        # Passing to a publicly known teammate has full confidence.
        np.testing.assert_allclose(engine.get_step_rewards(2), [.015, -.01], atol=1e-7)
        # Player 3's two teammates are both recognized. Finishing the game
        # must not add a feed-teammate reward after the terminal decision.
        np.testing.assert_allclose(engine.get_step_rewards(3), [-.01, -.01], atol=1e-7)
        np.testing.assert_allclose(engine.get_step_rewards(0), [0., 0.], atol=1e-7)

    def test_greedy_never_passes_when_last_card_must_beat_teammate(self):
        modules = [self.native]
        if importlib.util.find_spec('a3dizhu_cpp') is not None:
            import a3dizhu_cpp
            modules.append(a3dizhu_cpp)
        for module in modules:
            for remaining in (['club_4'], ['club_4', 'club_5']):
                with self.subTest(module=module.__name__, remaining=remaining):
                    engine = module.CppEngine()
                    engine.reset_with_hands([
                        ['spade_4', 'spade_8'],
                        ['diamond_4', 'diamond_5', 'diamond_9'],
                        remaining,
                        ['heart_4', 'heart_8'],
                    ], 1)
                    # Declaring makes players 1 and 2 publicly known teammates.
                    # Player 1 leads the required diamond 4; player 2 can beat it.
                    for player, key in ((0, 'declare'), (1, 'diamond_4')):
                        self.assertEqual(engine.get_player_id(), player)
                        self.assertIn(key, engine.get_legal_actions())
                        engine.step(key)
                    self.assertEqual(engine.get_player_id(), 2)
                    self.assertFalse(engine.is_over())
                    legal = engine.get_legal_actions()
                    greedy = engine.get_greedy_action()
                    self.assertIn(greedy, legal)
                    if len(remaining) == 1:
                        self.assertEqual(set(legal), {'club_4'})
                        self.assertEqual(greedy, 'club_4')
                    else:
                        self.assertIn('pass', legal)
                        self.assertEqual(greedy, 'pass')
                    engine.step(greedy)
                    self.assertEqual(engine.is_over(), len(remaining) == 1)

    def test_uncached_flush_keys_obey_room_straight_limits(self):
        cases = [(('3', '4', '5', '6', '7'), 2, 12, False),
                 (('10', 'J', 'Q', 'K', 'A'), 1, 11, True)]
        for ranks, start, end, warmup in cases:
            with self.subTest(start=start, end=end):
                cards = ['diamond_' + rank for rank in ranks]
                own = cards + ['diamond_9'] + (['diamond_4'] if warmup else [])
                hands = [own, ['club_' + rank for rank in ('4', '5', '6', '7', '8', '9')],
                         ['heart_4', 'heart_5', 'heart_6'], ['spade_3', 'spade_A', 'spade_6']]
                raw, cached, changed_rules = [self.native.CppEngine() for _ in range(3)]
                for engine in (raw, cached, changed_rules):
                    engine.reset_with_hands(hands, 0)
                    engine.set_rules(False, 1, 12)
                    for _ in range(4):
                        engine.step('pass')
                    if warmup:
                        engine.step('diamond_4')
                        for _ in range(3):
                            engine.step('pass')
                key = '|'.join(sorted(cards))
                # Populate a straight-flush entry before changing the rules.
                self.assertEqual(changed_rules.get_legal_actions()[key][52:61].argmax(), 7)
                for engine in (raw, cached, changed_rules):
                    engine.set_rules(False, start, end)
                legal_feature = cached.get_legal_actions()[key]
                self.assertEqual(legal_feature[52:61].argmax(), 4)
                for engine in (raw, changed_rules):
                    np.testing.assert_array_equal(engine.get_action_feature(key), legal_feature)
                for engine in (raw, cached, changed_rules):
                    engine.step(key)
                expected_actions = cached.get_legal_actions()
                for engine in (raw, changed_rules):
                    for player in range(4):
                        np.testing.assert_array_equal(engine.encode_obs(player), cached.encode_obs(player))
                    actual_actions = engine.get_legal_actions()
                    self.assertEqual(set(actual_actions), set(expected_actions))
                    for action in expected_actions:
                        np.testing.assert_array_equal(actual_actions[action], expected_actions[action])

    def test_direct_action_keys_match_cached_play_across_rule_variants(self):
        for require_both in (False, True):
            for start in (1, 2):
                for end in (11, 12):
                    with self.subTest(require_both=require_both, start=start, end=end):
                        direct, cached = self.native.CppEngine(), self.native.CppEngine()
                        for engine in (direct, cached):
                            engine.seed(61)
                            engine.reset()
                            engine.set_rules(require_both, start, end)
                        rng = random.Random(271)
                        for _ in range(300):
                            self.assertEqual(direct.is_over(), cached.is_over())
                            if cached.is_over():
                                break
                            player = cached.get_player_id()
                            self.assertEqual(direct.get_player_id(), player)
                            np.testing.assert_array_equal(direct.encode_obs(player), cached.encode_obs(player))
                            # The direct engine never enumerates legal actions,
                            # matching the uncached native rule-opponent path.
                            legal = cached.get_legal_actions()
                            key = rng.choice(list(legal))
                            np.testing.assert_array_equal(direct.get_action_feature(key), legal[key])
                            direct.step(key)
                            cached.step(key)
                        self.assertTrue(cached.is_over())
                        np.testing.assert_array_equal(direct.get_payoffs(), cached.get_payoffs())


if __name__ == '__main__':
    unittest.main()
