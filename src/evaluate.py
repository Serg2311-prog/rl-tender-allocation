"""Оценка стратегий на заранее зафиксированных тестовых сценариях.

Запуск: python -m src.evaluate
Все стратегии проходят одни и те же эпизоды (одинаковые seed тендеров и
исходов), Q-таблицы не дообучаются.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np
import pandas as pd

from .agent import QLearningAgent, StateDiscretizer
from .config import (AGENT_SEEDS, N_TEST_EPISODES, POLICIES_DIR, RANDOM_POLICY_SEED,
                     RESULTS_DIR,
                     TEST_SCENARIOS, EnvConfig, MarketConditions, Scenario,
                     config_snapshot, sweep_scenarios, holdout_seeds)
from .environment import TenderPortfolioEnv
from .policies import (AlwaysSkipPolicy, AlwaysSubmitPolicy, HeuristicPolicy,
                       OraclePolicy, Policy, QPolicy, RandomPolicy)

EPISODE_METRICS = ("reward", "revenue", "penalties", "cost", "n_submitted", "n_won",
                   "n_incidents", "budget_used", "violation", "invalid_actions")
# Квантили t-распределения t_{0.975}(df) для доверительных интервалов по seed.
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228}


def t975(df: float) -> float:
    """Консервативно: df округляется вниз; при df > 10 -- нормальное приближение."""
    return T_975.get(int(np.floor(df)), 1.96) if df >= 1 else np.nan


def run_episode(env: TenderPortfolioEnv, policy: Policy, seed: int,
                conditions: MarketConditions) -> dict:
    obs, info = env.reset(seed=seed, options={"conditions": conditions})
    policy.begin_episode(conditions)
    m = dict.fromkeys(EPISODE_METRICS, 0.0)
    min_budget = env.budget
    done = False
    while not done:
        tender = env.current_tender()
        obs, r, terminated, truncated, info = env.step(policy.act(obs, info))
        done = terminated or truncated
        m["reward"] += r
        if info["submitted"]:
            m["n_submitted"] += 1
            m["cost"] += info["cost"]
            if info["won"]:
                m["n_won"] += 1
                m["revenue"] += tender["value"]
            if info["incident"]:
                m["n_incidents"] += 1
                m["penalties"] += env.cfg.penalty_factor * tender["value"]
        m["invalid_actions"] += info["invalid_action"]
        min_budget = min(min_budget, env.budget)
    m["budget_used"] = m["cost"] / conditions.initial_budget
    m["violation"] = float(min_budget < 0.0)
    return m


def run_episodes(env, policy, seeds, conditions) -> dict[str, np.ndarray]:
    rows = [run_episode(env, policy, s, conditions) for s in seeds]
    return {k: np.array([r[k] for r in rows]) for k in EPISODE_METRICS}


def summarize(ep: dict[str, np.ndarray]) -> dict:
    submitted = ep["n_submitted"].sum()
    return {
        "reward_mean": ep["reward"].mean(),
        "reward_std": ep["reward"].std(ddof=1),
        "profit_mean": ep["reward"].mean(),   # награда = чистая прибыль, у.е.
        "revenue_mean": ep["revenue"].mean(),
        "penalties_mean": ep["penalties"].mean(),
        "cost_mean": ep["cost"].mean(),
        "submitted_mean": ep["n_submitted"].mean(),
        "won_mean": ep["n_won"].mean(),
        "incidents_mean": ep["n_incidents"].mean(),
        "success_rate": ep["n_won"].sum() / submitted if submitted else np.nan,
        "budget_used_mean": ep["budget_used"].mean(),
        "violation_rate": ep["violation"].mean(),
        "invalid_actions_mean": ep["invalid_actions"].mean(),
    }


def load_q_policies(discretizer: StateDiscretizer) -> list[tuple[str, int, QPolicy]]:
    out = []
    for cond in ("base", "dr"):
        for seed in AGENT_SEEDS:
            path = POLICIES_DIR / f"q_{cond}_seed{seed}.npz"
            if not path.exists():
                raise FileNotFoundError(f"{path} not found: run `python -m src.train` first")
            out.append((f"q_{cond}", seed,
                        QPolicy(QLearningAgent.load(path), discretizer, f"q_{cond}")))
    return out


def evaluate_all(n_episodes: int, include_sweeps: bool = True) -> None:
    t0 = time.time()
    env_cfg = EnvConfig()
    env = TenderPortfolioEnv(env_cfg)
    disc = StateDiscretizer()
    q_policies = load_q_policies(disc)
    oracle = OraclePolicy(env_cfg)
    seeds = holdout_seeds(n_episodes)

    scenarios: list[Scenario] = list(TEST_SCENARIOS)
    if include_sweeps:
        scenarios += sweep_scenarios()

    rows, rewards = [], {}
    for sc in scenarios:
        policies: list[tuple[str, int, Policy]] = [
            ("random", -1, RandomPolicy(RANDOM_POLICY_SEED)),
            ("heuristic", -1, HeuristicPolicy(env_cfg)),
            ("oracle", -1, oracle),
        ]
        if sc.name == "base":
            policies += [("always_skip", -1, AlwaysSkipPolicy()),
                         ("always_submit", -1, AlwaysSubmitPolicy())]
        policies += q_policies
        for family, seed, pol in policies:
            if isinstance(pol, QPolicy):
                pol.decisions = pol.unseen_decisions = 0
            ep = run_episodes(env, pol, seeds, sc.conditions)
            row = {"scenario": sc.name, "title": sc.title, "group": sc.group,
                   "coverage": sc.coverage, **asdict(sc.conditions),
                   "policy": family, "seed": seed, **summarize(ep)}
            row["unseen_state_share"] = (pol.unseen_decisions / pol.decisions
                                         if isinstance(pol, QPolicy) else np.nan)
            rows.append(row)
            rewards[f"{sc.name}|{family}|{seed}"] = ep["reward"]
        print(f"[{time.time() - t0:6.1f}s] {sc.name}: done")

    per_seed = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_seed.to_csv(RESULTS_DIR / "eval_per_seed.csv", index=False)
    np.savez_compressed(RESULTS_DIR / "episode_rewards.npz", **rewards)
    save_aggregates(per_seed, rewards)
    (RESULTS_DIR / "experiment_config.json").write_text(
        json.dumps({**config_snapshot(), "n_test_episodes_used": len(seeds)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved results to {RESULTS_DIR} in {time.time() - t0:.1f}s")


def aggregate(per_seed: pd.DataFrame, rewards: dict) -> pd.DataFrame:
    """Сводка по (сценарий, стратегия): средние по seed и эпизодам.

    Два источника неопределённости хранятся в разных колонках:
    reward_std           -- разброс награды между эпизодами (по всем seed);
    reward_seed_std      -- разброс средней награды между seed обучения (0 без обучения);
    reward_ci95_episodes -- полуширина 95% ДИ по тестовым эпизодам, 1.96 * SE; для
                            Q-агентов по награде эпизода, усреднённой по seed (агенты
                            фиксированы, случайна выборка эпизодов);
    reward_ci95_seeds    -- полуширина 95% ДИ по seed обучения, t(n_seeds-1) * SE средних
                            по seed (эпизоды фиксированы, случаен seed); NaN без обучения.
    """
    metric_cols = [c for c in per_seed.columns
                   if c.endswith(("_mean", "_rate", "_share")) and c != "reward_mean"]
    out = []
    for (sc, fam), g in per_seed.groupby(["scenario", "policy"], sort=False):
        per_seed_r = np.stack([rewards[f"{sc}|{fam}|{s}"] for s in g["seed"]])
        episode_mean = per_seed_r.mean(axis=0)
        row = g.iloc[0][["scenario", "title", "group", "coverage", "p_scale",
                         "cost_scale", "initial_budget", "policy"]].to_dict()
        row["n_seeds"] = len(g)
        row["n_episodes"] = per_seed_r.shape[1]
        row["reward_mean"] = g["reward_mean"].mean()
        row["reward_std"] = per_seed_r.ravel().std(ddof=1)
        row["reward_ci95_episodes"] = (1.96 * episode_mean.std(ddof=1)
                                       / np.sqrt(len(episode_mean)))
        if len(g) > 1:
            row["reward_seed_std"] = g["reward_mean"].std(ddof=1)
            row["reward_ci95_seeds"] = (t975(len(g) - 1) * row["reward_seed_std"]
                                        / np.sqrt(len(g)))
        else:
            row["reward_seed_std"] = 0.0
            row["reward_ci95_seeds"] = np.nan
        for c in metric_cols:
            row[c] = g[c].mean()
        out.append(row)
    return pd.DataFrame(out)


def paired_vs_heuristic(per_seed: pd.DataFrame, rewards: dict) -> pd.DataFrame:
    """Сравнение Q-агентов с эвристикой и оракулом на одних и тех же эпизодах.

    Каждая колонка *_ci95_* учитывает только один источник неопределённости:
    *_ci95_episodes -- 1.96 * SE парной разности по тестовым эпизодам (награда
                       Q-агента усреднена по seed);
    *_ci95_seeds    -- t(n_seeds-1) * SE разности средних по seed обучения;
    dr_minus_base_ci95_seeds -- ДИ Уэлча по seed для разности агентов DR и base;
    heuristic_minus_oracle   -- парная разность эвристики и оракула (без seed).
    """
    out = []
    for sc in per_seed["scenario"].unique():
        h = rewards[f"{sc}|heuristic|-1"]
        o = rewards[f"{sc}|oracle|-1"]
        seed_means, episode_means = {}, {}
        for fam in ("q_base", "q_dr"):
            seeds = per_seed.query("scenario == @sc and policy == @fam")["seed"]
            per_seed_r = np.stack([rewards[f"{sc}|{fam}|{s}"] for s in seeds])
            diff_ep = per_seed_r.mean(axis=0) - h     # среднее по seed, парно по эпизодам
            seed_diffs = per_seed_r.mean(axis=1) - h.mean()
            seed_means[fam] = per_seed_r.mean(axis=1)
            episode_means[fam] = per_seed_r.mean(axis=0)
            n = len(seeds)
            out.append({
                "scenario": sc, "policy": fam,
                "diff_vs_heuristic": diff_ep.mean(),
                "diff_ci95_episodes": 1.96 * diff_ep.std(ddof=1) / np.sqrt(len(h)),
                "diff_ci95_seeds": t975(n - 1) * seed_diffs.std(ddof=1) / np.sqrt(n),
                "seeds_better_than_heuristic": int((seed_diffs > 0).sum()),
                "n_seeds": n,
                "share_of_oracle": per_seed_r.mean() / o.mean() if o.mean() else np.nan,
                "heuristic_share_of_oracle": h.mean() / o.mean() if o.mean() else np.nan,
            })
        b, d = seed_means["q_base"], seed_means["q_dr"]
        vb, vd = b.var(ddof=1) / len(b), d.var(ddof=1) / len(d)
        welch_df = (vb + vd) ** 2 / (vb ** 2 / (len(b) - 1) + vd ** 2 / (len(d) - 1))
        dr_base_ep = episode_means["q_dr"] - episode_means["q_base"]
        h_o = h - o
        for row in out[-2:]:
            row["dr_minus_base"] = d.mean() - b.mean()
            row["dr_minus_base_ci95_seeds"] = t975(welch_df) * np.sqrt(vb + vd)
            row["dr_minus_base_ci95_episodes"] = (1.96 * dr_base_ep.std(ddof=1)
                                                  / np.sqrt(len(dr_base_ep)))
            row["heuristic_minus_oracle"] = h_o.mean()
            row["heuristic_minus_oracle_ci95_episodes"] = (1.96 * h_o.std(ddof=1)
                                                           / np.sqrt(len(h_o)))
    return pd.DataFrame(out)


def save_aggregates(per_seed: pd.DataFrame, rewards: dict) -> None:
    aggregate(per_seed, rewards).to_csv(RESULTS_DIR / "eval_summary.csv", index=False)
    paired_vs_heuristic(per_seed, rewards).to_csv(RESULTS_DIR / "paired_vs_heuristic.csv",
                                                  index=False)


def reaggregate() -> None:
    """Пересчёт сводных таблиц из сохранённых наград по эпизодам без прогона эпизодов."""
    per_seed = pd.read_csv(RESULTS_DIR / "eval_per_seed.csv")
    with np.load(RESULTS_DIR / "episode_rewards.npz") as data:
        rewards = {k: data[k] for k in data.files}
    save_aggregates(per_seed, rewards)
    print(f"Recomputed eval_summary.csv and paired_vs_heuristic.csv in {RESULTS_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=None,
                    help="число тестовых эпизодов на сценарий (по умолчанию из config)")
    ap.add_argument("--no-sweeps", action="store_true", help="без однофакторных развёрток")
    ap.add_argument("--reaggregate", action="store_true",
                    help="только пересчитать сводки из results/episode_rewards.npz")
    args = ap.parse_args()
    if args.reaggregate:
        reaggregate()
    else:
        evaluate_all(args.episodes or N_TEST_EPISODES, include_sweeps=not args.no_sweeps)


if __name__ == "__main__":
    main()
