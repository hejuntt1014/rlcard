"""A3 environment with structured history, afterstate features and fixed-rule support."""
import random

import numpy as np


STATE_DIM = 2668
ACTION_DIM = 111
FEATURE_SCHEMA = 'a3-v12-lite-v1'
RULE_KEYS = ('declare_require_both_spades', 'straight_start_val', 'straight_end_val')


def validated_rules(rules):
    if rules is None:
        return None
    if set(rules) != set(RULE_KEYS):
        raise ValueError('rules must specify ' + ', '.join(RULE_KEYS))
    if type(rules[RULE_KEYS[0]]) is not bool:
        raise ValueError('declare_require_both_spades must be a boolean')
    if rules[RULE_KEYS[1]] not in (1, 2) or rules[RULE_KEYS[2]] not in (11, 12):
        raise ValueError('straight_start_val must be 1/2 and straight_end_val must be 11/12')
    return dict(rules)


class A3ContextEnv:
    name = 'a3dizhu-v12'
    feature_schema = FEATURE_SCHEMA
    native_module = 'a3dizhu_v12_cpp'
    aux_classes = (3, 3, 3, 4, 4)

    def __init__(self, config):
        if config.get('backend', 'auto') not in ('auto', 'cpp'):
            raise ValueError('a3dizhu-v12 requires the native backend')
        if config.get('allow_step_back', False):
            raise ValueError('a3dizhu-v12 does not support step_back')
        try:
            import a3dizhu_v12_cpp as native
        except ImportError as exc:
            raise ImportError('Build the native extensions: python setup_cpp.py build_ext --inplace') from exc
        if native.FEATURE_VERSION != FEATURE_SCHEMA:
            raise ValueError('Native feature schema mismatch; rebuild the extension')
        self._config = dict(config)
        self.rules = validated_rules(config.get('rules'))
        self.reward_mode = config.get('reward_mode', 'game')
        if self.reward_mode not in ('game', 'shaped'):
            raise ValueError('reward_mode must be game or shaped')
        self.greedy_ratio = float(config.get('greedy_ratio', 0.))
        self.random_ratio = float(config.get('random_ratio', 0.))
        if min(self.greedy_ratio, self.random_ratio) < 0 or self.greedy_ratio + self.random_ratio > 1:
            raise ValueError('Rule-opponent probabilities must be nonnegative and sum to at most one')
        self._engine = native.CppEngine()
        self._engine.set_greedy_ratio(self.greedy_ratio)
        self._engine.set_random_ratio(self.random_ratio)
        if hasattr(self._engine, 'set_reward_shaping'):
            self._engine.set_reward_shaping(self.reward_mode == 'shaped')
        self.num_players, self.num_actions = 4, 1
        self.state_shape = [[STATE_DIM] for _ in range(4)]
        self.action_shape = [[ACTION_DIM] for _ in range(4)]
        self.agents = []
        self.timestep = 0
        self.seed(config.get('seed'))

    def __getstate__(self):
        return self._config

    def __setstate__(self, config):
        self.__init__(config)

    def seed(self, seed=None):
        self._engine.seed(random.randrange(2**32) if seed is None else int(seed))

    def reset(self):
        p = self._engine.reset()
        if self.rules is not None:
            self._engine.set_rules(*(self.rules[k] for k in RULE_KEYS))
        self.timestep = 0
        return self.get_state(p), p

    def step(self, action, raw_action=False):
        key = 'pass' if action is None else str(action)
        self.timestep += 1
        p = self._engine.step(key)
        return self.get_state(p), p

    def get_state(self, player_id):
        legal = self._engine.get_legal_actions()
        return dict(obs=self._engine.encode_obs(player_id), legal_actions=legal,
                    raw_legal_actions=list(legal))

    def get_action_feature(self, action):
        """State-dependent feature; capture before executing the action."""
        return np.asarray(self._engine.get_action_feature('pass' if action is None else str(action)))

    def is_over(self):
        return self._engine.is_over()

    def get_player_id(self):
        return self._engine.get_player_id()

    def get_payoffs(self):
        return list(self._engine.get_payoffs())

    def get_training_payoffs(self):
        return (list(self._engine.get_training_payoffs()) if self.reward_mode == 'shaped'
                else self.get_payoffs())

    def get_aux_targets(self):
        return self._engine.get_aux_targets()

    def set_agents(self, agents):
        self.agents = agents

    def run(self, is_training=False):
        state, p = self.reset()
        trajectories = [[] for _ in range(4)]
        while not self.is_over():
            trajectories[p].append(state)
            if self._engine.is_rule_agent_seat(p):
                action = self._engine.get_rule_agent_action()
            else:
                action = (self.agents[p].step(state) if is_training
                          else self.agents[p].eval_step(state)[0])
            trajectories[p].append(action)
            state, p = self.step(action)
        for p in range(4):
            trajectories[p].append(self.get_state(p))
        return trajectories, self.get_training_payoffs() if is_training else self.get_payoffs()
