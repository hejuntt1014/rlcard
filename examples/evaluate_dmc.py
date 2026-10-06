"""Evaluate A3 checkpoints on fixed deals against greedy opponents.

Each seed is played four times, assigning the model to each seat in turn.
Uncertainty is calculated across seed averages, not correlated seat games.
The reported score is the original game payoff, without reward shaping.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

import rlcard
from rlcard.agents.dmc_agent.model import DMCModel


class EvaluationActionError(ValueError):
    """A reproducible illegal decision; evaluation never substitutes an action."""
    def __init__(self, details):
        self.details = details
        super().__init__('Illegal evaluation action at seed=%s seat=%s step=%s: %r' % (
            details['seed'], details['seat'], details['step'], details['action']))


def load_policy(checkpoint, env, device='cpu'):
    """Load self-describing training weights, validating the feature contract."""
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    spec = payload.get('model_spec')
    if not isinstance(spec, dict):
        raise ValueError('Checkpoint must contain model_spec; export legacy weights with their configuration first')
    state_shape = [list(s) for s in env.state_shape]
    action_shape = [list(s) if s is not None else [env.num_actions] for s in env.action_shape]
    if spec.get('state_shape') != state_shape or spec.get('action_shape') != action_shape:
        raise ValueError('Checkpoint observation/action shapes do not match the evaluation environment')
    if spec.get('env_name', env.name) != env.name:
        raise ValueError('Checkpoint environment does not match the evaluation environment')
    expected_schema = getattr(env, 'feature_schema',
                              'a3-v6-850-action52-v1' if env.name == 'a3dizhu' else None)
    schema = spec.get('feature_schema')
    if schema != expected_schema:
        # V6 checkpoints saved before feature schemas were introduced have an
        # unambiguous fixed layout. New layouts must always identify themselves.
        legacy_v6 = (schema is None and env.name == 'a3dizhu'
                     and state_shape == [[850]] * 4 and action_shape == [[52]] * 4)
        if not legacy_v6:
            raise ValueError('Checkpoint feature schema does not match the evaluation environment')
    fields = ('state_shape', 'action_shape', 'mlp_layers', 'share_weights',
              'architecture', 'aux_classes', 'history_encoder', 'history_steps')
    kwargs = {key: spec[key] for key in fields if key in spec}
    model = DMCModel(**kwargs, device=device, exp_epsilon=0.)
    states = payload.get('model_state_dict', [])
    if len(states) != env.num_players:
        raise ValueError('Checkpoint role count does not match the evaluation environment')
    if spec.get('share_weights'):
        reference = states[0]
        if any(set(state) != set(reference) or any(
                not torch.equal(reference[k], state[k]) for k in reference)
               for state in states[1:]):
            raise ValueError('Shared checkpoint contains inconsistent role weights')
    for player in range(env.num_players):
        model.get_agent(player).load_state_dict(states[player])
    model.eval()
    return model, {'checkpoint': str(Path(checkpoint).resolve()),
                   'training_frames': payload.get('frames'), 'model_spec': spec}


def play_game(env, model, seed, seat, max_steps=10000):
    """Manually dispatch each turn so the model seat is never auto-controlled."""
    env.seed(seed)
    state, player = env.reset()
    obs = np.asarray(state['obs'])
    rules = None
    if len(obs) == 2668:
        rules = dict(straight_start_val=2 if obs[538] else 1,
                     straight_end_val=12 if obs[540] else 11,
                     declare_require_both_spades=bool(obs[543]))
    steps = model_decisions = 0
    declarant = None
    history = []
    while not env.is_over():
        if steps >= max_steps:
            raise RuntimeError('Evaluation exceeded max_steps; no truncated score was recorded')
        if player == seat:
            action, _ = model.get_agent(player).eval_step(state)
            model_decisions += 1
        else:
            action = env._engine.get_rule_agent_action()
        if action not in state['legal_actions']:
            raise EvaluationActionError(dict(seed=seed, seat=seat, player=player, step=steps,
                policy='model' if player == seat else 'native greedy', action=str(action),
                legal_actions=list(state['legal_actions']), rules=rules, history=history))
        if action == 'declare':
            declarant = player
        history.append([player, str(action)])
        state, player = env.step(action)
        steps += 1
    payoffs = [float(value) for value in env.get_payoffs()]
    if not np.isfinite(payoffs).all():
        raise ValueError('Evaluation returned non-finite game payoffs')
    score = payoffs[seat]
    # Global mode can contain information hidden from a player's observation.
    # Read it only after the game for diagnostics, never as a policy input.
    mode = env._engine.get_mode() if hasattr(env._engine, 'get_mode') else 'unknown'
    return dict(seed=seed, seat=seat, payoff=score, all_payoffs=payoffs,
                outcome='win' if score > 0 else 'loss' if score < 0 else 'draw',
                mode=mode, steps=steps, model_decisions=model_decisions,
                declarant=declarant, model_declared=declarant == seat, rules=rules)


def mean_and_error(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        raise ValueError('At least one sample is required')
    return dict(mean=float(values.mean()),
                standard_error=(float(values.std(ddof=1) / np.sqrt(len(values)))
                                if len(values) > 1 else None), count=len(values))


def summarize(records):
    grouped = defaultdict(list)
    by_mode = defaultdict(list)
    for record in records:
        grouped[record['seed']].append(record['payoff'])
        by_mode[record['mode']].append(record['payoff'])
    if not grouped:
        raise ValueError('At least one complete seed is required')
    result = mean_and_error([np.mean(values) for values in grouped.values()])
    result.update(games=len(records), uncertainty_unit='seed (four seat rotations)',
                  outcomes=dict(Counter(r['outcome'] for r in records)),
                  modes={mode: dict(games=len(values), mean_payoff=float(np.mean(values)))
                         for mode, values in sorted(by_mode.items())},
                  seats={str(seat): float(np.mean([r['payoff'] for r in records if r['seat'] == seat]))
                         for seat in sorted({r['seat'] for r in records})})
    declarations = [r for r in records if r.get('model_declared', False)]
    result['model_declarations'] = dict(
        games=len(declarations), frequency=len(declarations) / len(records),
        positive_payoff_rate=(sum(r['payoff'] > 0 for r in declarations) / len(declarations)
                              if declarations else None))
    rule_groups = defaultdict(list)
    for record in records:
        if record.get('rules') is not None:
            rule_groups[json.dumps(record['rules'], sort_keys=True)].append(record['payoff'])
    result['rules'] = [dict(rules=json.loads(key), games=len(values), mean_payoff=float(np.mean(values)))
                       for key, values in sorted(rule_groups.items())]
    return result


def paired_difference(reference, candidate):
    """Compare score differences on identical seeds and seats, clustered by seed."""
    first = {(r['seed'], r['seat']): r['payoff'] for r in reference}
    second = {(r['seed'], r['seat']): r['payoff'] for r in candidate}
    if len(first) != len(reference) or len(second) != len(candidate) or first.keys() != second.keys():
        raise ValueError('Paired comparison requires identical, unique seed/seat pairs')
    differences = defaultdict(list)
    for key in first:
        differences[key[0]].append(second[key] - first[key])
    return mean_and_error([np.mean(values) for values in differences.values()])


def evaluate(model, env, seeds, max_steps=10000):
    if not hasattr(env, '_engine'):
        raise ValueError('A3 greedy evaluation requires the native engine')
    return [play_game(env, model, seed, seat, max_steps)
            for seed in seeds for seat in range(env.num_players)]


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('checkpoints', nargs='+', help='Self-describing model.tar checkpoints')
    parser.add_argument('--env', choices=['a3dizhu', 'a3dizhu-v12'], default='a3dizhu-v12')
    parser.add_argument('--seeds', type=int, default=100, help='Independent deals; each gets four seat rotations')
    parser.add_argument('--seed', type=int, default=100000, help='First evaluation seed')
    parser.add_argument('--device', default='cpu', help='cpu or CUDA index such as 0')
    parser.add_argument('--rules', type=json.loads, help='Fixed V12 rules as a JSON object; omitted uses seeded random rules')
    parser.add_argument('--max_steps', type=int, default=10000)
    parser.add_argument('--output', default='experiments/evaluation/results.json')
    args = parser.parse_args()
    if args.seeds < 1 or args.max_steps < 1 or args.seed < 0 or args.seed + args.seeds > 2**32:
        parser.error('Require positive seeds/max_steps and uint32 seed range')
    if args.rules is not None and (args.env != 'a3dizhu-v12' or not isinstance(args.rules, dict)):
        parser.error('--rules requires an object and --env a3dizhu-v12')
    torch.set_num_threads(1)
    config = dict(seed=args.seed, greedy_ratio=1., random_ratio=0., reward_mode='game')
    if args.rules is not None:
        config['rules'] = args.rules
    env = rlcard.make(args.env, config=config)
    report = dict(environment=args.env, config=config, first_seed=args.seed, seeds=args.seeds,
                  opponent='native greedy', metric='original game payoff',
                  protocol='one model seat against three greedy seats; rotate all four seats per deal',
                  models=[], paired_differences=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for checkpoint in args.checkpoints:
            model, metadata = load_policy(checkpoint, env, args.device)
            records = evaluate(model, env, range(args.seed, args.seed + args.seeds), args.max_steps)
            report['models'].append(dict(**metadata, summary=summarize(records), games=records))
            if len(report['models']) > 1:
                report['paired_differences'].append(dict(
                    reference=report['models'][0]['checkpoint'], candidate=metadata['checkpoint'],
                    **paired_difference(report['models'][0]['games'], records)))
            output.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({'checkpoint': metadata['checkpoint'], 'summary': summarize(records)}))
            del model
    except EvaluationActionError as error:
        report['failure'] = dict(checkpoint=str(Path(checkpoint).resolve()),
                                 rejected_illegal_actions=1, **error.details)
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        raise
    finally:
        if hasattr(env, 'close'):
            env.close()


if __name__ == '__main__':
    main()
