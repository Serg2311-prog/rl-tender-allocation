from dataclasses import replace

import numpy as np
import pytest

from src import train as train_module
from src.agent import QLearningAgent, StateDiscretizer
from src.config import (AGENT_SEEDS, BASE_CONDITIONS, N_TEST_EPISODES, TEST_SCENARIOS,
                        TRAIN_SEED_STRIDE, VAL_SEED_OFFSET, MarketConditions,
                        QLearningConfig, sweep_scenarios, holdout_seeds,
                        train_episode_seed, validation_seeds)
from src.environment import SKIP, SUBMIT, TenderPortfolioEnv
from src.evaluate import run_episodes
from src.policies import (HeuristicPolicy, OraclePolicy, QPolicy, RandomPolicy)


def _all_policies():
    disc = StateDiscretizer()
    return [RandomPolicy(0), HeuristicPolicy(), OraclePolicy(n_samples=500),
            QPolicy(QLearningAgent(disc.n_states), disc)]


@pytest.mark.parametrize("cond", [BASE_CONDITIONS,
                                  MarketConditions(p_scale=0.55, cost_scale=1.7,
                                                   initial_budget=15.0)])
def test_all_policies_run_without_violations(cond):
    env = TenderPortfolioEnv()
    for pol in _all_policies():
        ep = run_episodes(env, pol, range(30), cond)
        assert ep["violation"].sum() == 0
        assert ep["invalid_actions"].sum() == 0
        assert np.all(ep["budget_used"] <= 1.0 + 1e-9)


def test_heuristic_rule():
    pol = HeuristicPolicy()
    mask = {"action_mask": np.array([1, 1]), "budget": 40.0}
    good = np.array([80.0, 0.5, 8.0, 0.1, 40.0, 30.0, 1.0], dtype=np.float32)
    risky = good.copy()
    risky[3] = 0.7
    weak = good.copy()
    weak[1] = 0.1
    assert pol.act(good, mask) == SUBMIT
    assert pol.act(risky, mask) == SKIP
    assert pol.act(weak, mask) == SKIP
    assert pol.act(good, {"action_mask": np.array([1, 0]), "budget": 1.0}) == SKIP


def test_oracle_value_function_is_monotone():
    oracle = OraclePolicy(n_samples=1000)
    oracle.begin_episode(BASE_CONDITIONS)
    W = oracle.W
    assert np.all(np.diff(W, axis=0) >= -1e-9)   # больше шагов -- не хуже
    assert np.all(np.diff(W, axis=1) >= -1e-9)   # больше бюджета -- не хуже
    assert np.all(W[0] == 0.0)


def test_oracle_beats_heuristic_on_validation():
    env = TenderPortfolioEnv()
    seeds = validation_seeds(300)
    h = run_episodes(env, HeuristicPolicy(), seeds, BASE_CONDITIONS)["reward"]
    o = run_episodes(env, OraclePolicy(), seeds, BASE_CONDITIONS)["reward"]
    assert o.mean() > h.mean()


def test_seed_ranges_are_disjoint():
    max_train = train_episode_seed(max(AGENT_SEEDS), QLearningConfig().n_episodes - 1)
    assert QLearningConfig().n_episodes <= TRAIN_SEED_STRIDE
    assert max_train < VAL_SEED_OFFSET
    assert not set(validation_seeds(10_000)) & set(holdout_seeds(N_TEST_EPISODES))
    assert min(holdout_seeds()) > max(validation_seeds(QLearningConfig().n_val_episodes))


def test_no_test_seed_leaks_into_training(monkeypatch):
    used = []
    original = TenderPortfolioEnv.reset

    def spy(self, *, seed=None, options=None):
        used.append(seed)
        return original(self, seed=seed, options=options)

    monkeypatch.setattr(TenderPortfolioEnv, "reset", spy)
    cfg = replace(QLearningConfig(), n_episodes=200, eval_every=100, n_val_episodes=20)
    for condition in ("base", "dr"):
        train_module.train_agent(condition, 0, cfg, verbose=False)
    assert None not in used
    assert not set(used) & set(holdout_seeds(N_TEST_EPISODES))


def test_scenario_names_unique_and_coverage_labels_consistent():
    from src.config import DR_RANGES
    scenarios = list(TEST_SCENARIOS) + sweep_scenarios()
    assert len({s.name for s in scenarios}) == len(scenarios)
    for s in scenarios:
        c = s.conditions
        in_dr = all(lo <= getattr(c, f) <= hi for f, (lo, hi) in
                    (("p_scale", DR_RANGES.p_scale), ("cost_scale", DR_RANGES.cost_scale),
                     ("initial_budget", DR_RANGES.initial_budget)))
        if s.coverage == "base":
            assert c == BASE_CONDITIONS
        elif s.coverage == "dr":
            assert in_dr and c != BASE_CONDITIONS
        else:
            assert not in_dr
