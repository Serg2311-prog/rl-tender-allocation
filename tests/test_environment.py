import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env

from src.config import BASE_CONDITIONS, EnvConfig, MarketConditions, validation_seeds
from src.environment import SKIP, SUBMIT, TenderPortfolioEnv, expected_reward
from src.evaluate import run_episodes
from src.policies import (AlwaysSkipPolicy, AlwaysSubmitPolicy, HeuristicPolicy,
                          RandomPolicy)

EXTREME = [BASE_CONDITIONS,
           MarketConditions(p_scale=0.4, cost_scale=2.0, initial_budget=10.0),
           MarketConditions(p_scale=1.6, cost_scale=0.6, initial_budget=120.0)]


@pytest.fixture
def env():
    return TenderPortfolioEnv()


def test_initialization_and_spaces(env):
    assert isinstance(env.action_space, spaces.Discrete) and env.action_space.n == 2
    assert isinstance(env.observation_space, spaces.Box)
    assert env.observation_space.shape == (7,)
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    assert env.budget == BASE_CONDITIONS.initial_budget
    assert obs[5] == env.cfg.horizon
    assert list(info["action_mask"]) in ([1, 0], [1, 1])


def test_gymnasium_check_env():
    check_env(TenderPortfolioEnv(), skip_render_check=True)


@pytest.mark.parametrize("cond", EXTREME)
def test_observations_in_space_and_termination(env, cond):
    rng = np.random.default_rng(0)
    for seed in range(20):
        obs, info = env.reset(seed=seed, options={"conditions": cond})
        steps, terminated = 0, False
        while not terminated:
            obs, r, terminated, truncated, info = env.step(int(rng.integers(2)))
            assert env.observation_space.contains(obs)
            assert not truncated
            steps += 1
        assert steps == env.cfg.horizon
        assert obs[5] == 0
        with pytest.raises(RuntimeError):
            env.step(SKIP)


def test_invalid_action_values_rejected(env):
    env.reset(seed=0)
    for bad in (2, -1, 0.5):
        with pytest.raises(ValueError):
            env.step(bad)


@pytest.mark.parametrize("cond", EXTREME)
def test_budget_never_negative(env, cond):
    for seed in range(50):
        env.reset(seed=seed, options={"conditions": cond})
        terminated = False
        while not terminated:
            budget_before = env.budget
            _, r, terminated, _, info = env.step(SUBMIT)
            assert env.budget >= 0.0
            if info["invalid_action"]:
                assert r == 0.0 and env.budget == budget_before
                assert not info["submitted"]


def test_action_mask_always_allows_skip(env):
    obs, info = env.reset(seed=4)
    assert info["action_mask"][SKIP] == 1
    env.budget = 0.0
    mask = env.action_mask()
    assert mask[SKIP] == 1 and mask[SUBMIT] == 0
    obs, r, terminated, _, info = env.step(SKIP)
    assert r == 0.0 and info["action_mask"][SKIP] == 1


def test_unaffordable_submit_is_noop(env):
    env.reset(seed=3)
    env.budget = 0.5   # меньше любой стоимости подготовки
    mask = env.action_mask()
    assert mask[SUBMIT] == 0
    _, r, _, _, info = env.step(SUBMIT)
    assert info["invalid_action"] and r == 0.0 and env.budget == 0.5


def test_reproducibility_same_seed():
    def trajectory(seed):
        env = TenderPortfolioEnv()
        obs, _ = env.reset(seed=seed)
        out = [obs]
        terminated = False
        while not terminated:
            obs, r, terminated, _, _ = env.step(SUBMIT)
            out += [obs, np.array([r])]
        return np.concatenate(out)

    np.testing.assert_array_equal(trajectory(42), trajectory(42))
    assert not np.array_equal(trajectory(42), trajectory(43))


def test_conditions_only_scale_shared_draws(env):
    env.reset(seed=7)
    base = env._tenders
    env.reset(seed=7, options={"conditions": MarketConditions(p_scale=0.5, cost_scale=1.5)})
    shifted = env._tenders
    np.testing.assert_array_equal(base.value, shifted.value)
    np.testing.assert_array_equal(base.risk, shifted.risk)
    np.testing.assert_allclose(shifted.cost, 1.5 * base.cost)
    np.testing.assert_allclose(shifted.p_win, np.clip(0.5 * base.p_win, 0.01, 0.95))


@pytest.mark.parametrize("u_win,u_risk,expected", [
    (0.9, 0.0, -10.0),                 # проигрыш: только затраты
    (0.1, 0.9, 50.0 - 10.0),           # победа без инцидента
    (0.1, 0.1, 50.0 - 1.5 * 50.0 - 10.0),  # победа с инцидентом
])
def test_reward_formula(env, u_win, u_risk, expected):
    env.reset(seed=0)
    t = env._t
    env._tenders.value[t], env._tenders.p_win[t] = 50.0, 0.5
    env._tenders.cost[t], env._tenders.risk[t] = 10.0, 0.3
    env._u_win[t], env._u_risk[t] = u_win, u_risk
    budget = env.budget
    _, r, _, _, info = env.step(SUBMIT)
    assert r == pytest.approx(expected)
    assert env.budget == pytest.approx(budget - 10.0)
    assert info["won"] == (u_win < 0.5)


def test_skip_gives_zero_and_keeps_budget(env):
    env.reset(seed=0)
    budget = env.budget
    _, r, _, _, info = env.step(SKIP)
    assert r == 0.0 and env.budget == budget and not info["submitted"]


def test_expected_reward_matches_simulation():
    rng = np.random.default_rng(0)
    v, p, c, risk, kappa = 60.0, 0.4, 8.0, 0.3, EnvConfig().penalty_factor
    n = 200_000
    won = rng.random(n) < p
    incident = won & (rng.random(n) < risk)
    realized = won * v - incident * kappa * v - c
    assert realized.mean() == pytest.approx(expected_reward(v, p, c, risk, kappa), abs=0.3)


def test_no_trivial_exploits():
    """Постоянный отказ даёт ровно 0, бездумная трата бюджета -- хуже эвристики."""
    env = TenderPortfolioEnv()
    seeds = validation_seeds(300)
    skip = run_episodes(env, AlwaysSkipPolicy(), seeds, BASE_CONDITIONS)["reward"]
    submit = run_episodes(env, AlwaysSubmitPolicy(), seeds, BASE_CONDITIONS)
    rand = run_episodes(env, RandomPolicy(0), seeds, BASE_CONDITIONS)["reward"]
    heur = run_episodes(env, HeuristicPolicy(), seeds, BASE_CONDITIONS)["reward"]
    assert np.all(skip == 0.0)
    assert submit["budget_used"].mean() > 0.9
    assert submit["reward"].mean() < heur.mean() - 10
    assert rand.mean() < heur.mean() - 10
