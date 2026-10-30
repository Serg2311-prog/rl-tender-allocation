"""Свойства агрегации метрик: источники неопределённости не смешиваются."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import aggregate, paired_vs_heuristic, t975


def test_t975_uses_student_quantile_for_small_df():
    assert t975(4) == pytest.approx(2.776)
    assert t975(10) == pytest.approx(2.228)
    assert t975(11) == pytest.approx(1.96)
    assert np.isnan(t975(0.5))


def _frame(scenario, policy, seed, reward_mean):
    return {
        "scenario": scenario, "title": scenario, "group": "main", "coverage": "base",
        "p_scale": 1.0, "cost_scale": 1.0, "initial_budget": 40.0,
        "policy": policy, "seed": seed, "reward_mean": reward_mean,
        "profit_mean": reward_mean, "success_rate": 0.4, "unseen_state_share": 0.0,
    }


def test_aggregate_keeps_episode_and_seed_intervals_separate():
    rng = np.random.default_rng(0)
    rewards = {
        "s|heuristic|-1": rng.normal(10.0, 2.0, 40),
        "s|q_base|0": rng.normal(12.0, 2.0, 40),
        "s|q_base|1": rng.normal(11.0, 2.0, 40),
        "s|q_base|2": rng.normal(13.0, 2.0, 40),
        "s|q_dr|0": rng.normal(9.0, 2.0, 40),
        "s|q_dr|1": rng.normal(9.5, 2.0, 40),
        "s|q_dr|2": rng.normal(8.5, 2.0, 40),
        "s|oracle|-1": rng.normal(15.0, 2.0, 40),
    }
    rows = [_frame("s", "heuristic", -1, rewards["s|heuristic|-1"].mean())]
    rows += [_frame("s", "q_base", i, rewards[f"s|q_base|{i}"].mean()) for i in range(3)]
    rows += [_frame("s", "q_dr", i, rewards[f"s|q_dr|{i}"].mean()) for i in range(3)]
    rows += [_frame("s", "oracle", -1, rewards["s|oracle|-1"].mean())]
    summary = aggregate(pd.DataFrame(rows), rewards)
    heur = summary.set_index("policy").loc["heuristic"]
    qb = summary.set_index("policy").loc["q_base"]
    assert np.isnan(heur["reward_ci95_seeds"])
    assert heur["reward_ci95_episodes"] == pytest.approx(
        1.96 * rewards["s|heuristic|-1"].std(ddof=1) / np.sqrt(40))
    seed_means = np.array([rewards[f"s|q_base|{i}"].mean() for i in range(3)])
    assert qb["reward_ci95_seeds"] == pytest.approx(
        t975(2) * seed_means.std(ddof=1) / np.sqrt(3))
    ep = np.mean([rewards[f"s|q_base|{i}"] for i in range(3)], axis=0)
    assert qb["reward_ci95_episodes"] == pytest.approx(1.96 * ep.std(ddof=1) / np.sqrt(40))
    assert "reward_ci95" not in summary.columns

    paired = paired_vs_heuristic(pd.DataFrame(rows), rewards)
    base = paired.set_index("policy").loc["q_base"]
    assert base["diff_ci95_episodes"] != pytest.approx(base["diff_ci95_seeds"])
    assert "dr_minus_base_ci95_seeds" in paired.columns
    assert "heuristic_minus_oracle_ci95_episodes" in paired.columns
