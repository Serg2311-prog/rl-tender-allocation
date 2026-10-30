"""Анализ поведения стратегий на тестовых эпизодах (без обучения).

Пошагово записывает решения эвристики, оракула и Q-агентов и сохраняет
агрегаты: долю подач по ожидаемой выгоде и стадии периода, средний остаток
бюджета по шагам, согласие с оракулом и пример одного эпизода.

Запуск: python -m src.behavior
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .agent import StateDiscretizer
from .config import RESULTS_DIR, TEST_SCENARIOS, EnvConfig, holdout_seeds
from .environment import SUBMIT, TenderPortfolioEnv, expected_reward
from .evaluate import load_q_policies
from .policies import HeuristicPolicy, OraclePolicy

BEHAVIOR_SCENARIOS = ("base", "p_0.55", "cost_1.70", "budget_15", "budget_90")
N_EPISODES = 200
EV_BINS = np.array([-np.inf, -5.0, 0.0, 2.5, 5.0, 10.0, 20.0, np.inf])
EV_LABELS = ["<-5", "-5..0", "0..2.5", "2.5..5", "5..10", "10..20", ">20"]
EXAMPLE_SEED_INDEX = 0


def stage_of(steps_left: int) -> str:
    return "начало (>15)" if steps_left > 15 else ("середина (6-15)" if steps_left > 5
                                                    else "конец (<=5)")


def log_decisions(env, policy, family, agent_seed, scenario, seeds) -> list[dict]:
    rows = []
    kappa = env.cfg.penalty_factor
    for ep_idx, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed, options={"conditions": scenario.conditions})
        policy.begin_episode(scenario.conditions)
        done = False
        while not done:
            t = env.current_tender()
            budget, steps = env.budget, env.steps_left
            a = policy.act(obs, info)
            obs, r, done, _, info = env.step(a)
            rows.append({
                "scenario": scenario.name, "policy": family, "seed": agent_seed,
                "episode": ep_idx, "t": env.cfg.horizon - steps, "steps_left": steps,
                **t, "ev": expected_reward(t["value"], t["p_win"], t["cost"], t["risk"], kappa),
                "budget_before": budget, "affordable": t["cost"] <= budget,
                "action": a, "reward": r, "won": info["won"], "incident": info["incident"],
            })
    return rows


def main() -> None:
    env_cfg = EnvConfig()
    env = TenderPortfolioEnv(env_cfg)
    q_policies = load_q_policies(StateDiscretizer())
    oracle = OraclePolicy(env_cfg)
    seeds = holdout_seeds(N_EPISODES)
    scenarios = [s for s in TEST_SCENARIOS if s.name in BEHAVIOR_SCENARIOS]

    rows = []
    for sc in scenarios:
        policies = [("heuristic", -1, HeuristicPolicy(env_cfg)), ("oracle", -1, oracle),
                    *q_policies]
        for family, agent_seed, pol in policies:
            rows += log_decisions(env, pol, family, agent_seed, sc, seeds)
        print(f"{sc.name}: logged")
    df = pd.DataFrame(rows)
    df["stage"] = df["steps_left"].map(stage_of)
    df["ev_bin"] = pd.cut(df["ev"], EV_BINS, labels=EV_LABELS)
    df["submit"] = (df["action"] == SUBMIT).astype(float)

    affordable = df[df["affordable"]]
    submit_rate = (affordable.groupby(["scenario", "policy", "stage", "ev_bin"], observed=True)
                   ["submit"].agg(["mean", "size"]).reset_index()
                   .rename(columns={"mean": "submit_rate", "size": "n"}))
    submit_rate.to_csv(RESULTS_DIR / "behavior_submit_rate.csv", index=False)

    budget_path = (df.groupby(["scenario", "policy", "t"])["budget_before"]
                   .agg(["mean", "std"]).reset_index())
    budget_path.to_csv(RESULTS_DIR / "behavior_budget_path.csv", index=False)

    key = ["scenario", "episode", "t"]
    ref = df[df["policy"] == "oracle"].set_index(key)["action"].rename("oracle_action")
    joined = df[df["policy"] != "oracle"].join(ref, on=key)
    # Согласие сравнимо только там, где состояние (бюджет) совпадает с оракулом,
    # поэтому дополнительно считаем согласие по ожидаемой выгоде тендера.
    agree = joined.groupby(["scenario", "policy", "seed"]).apply(
        lambda g: pd.Series({
            "agreement_with_oracle": (g["action"] == g["oracle_action"]).mean(),
            "submit_rate_ev_negative": g.loc[g["affordable"] & (g["ev"] < 0), "submit"].mean(),
            "submit_rate_ev_top": g.loc[g["affordable"] & (g["ev"] > 10), "submit"].mean(),
            "submit_rate_early": g.loc[g["affordable"] & (g["steps_left"] > 15), "submit"].mean(),
            "submit_rate_late": g.loc[g["affordable"] & (g["steps_left"] <= 5), "submit"].mean(),
            "budget_before_last_step": g.loc[g["steps_left"] == 1, "budget_before"].mean(),
        }), include_groups=False).reset_index()
    agree.to_csv(RESULTS_DIR / "behavior_agreement.csv", index=False)

    example = df[(df["scenario"] == "base") & (df["episode"] == EXAMPLE_SEED_INDEX)
                 & (df["seed"].isin([-1, 0]))]
    example.to_csv(RESULTS_DIR / "example_episode.csv", index=False)

    q_value_diagnostic(df, q_policies).to_csv(RESULTS_DIR / "q_value_diagnostic.csv",
                                              index=False)
    print(f"Saved behavior tables to {RESULTS_DIR}")


def q_value_diagnostic(df: pd.DataFrame, q_policies) -> pd.DataFrame:
    """Сравнивает оценку max_a Q(s_0, a) с фактической дисконтированной
    наградой жадной политики из того же начального состояния. Систематическое
    превышение указывает на смещение оценок вверх (maximization bias)."""
    disc = StateDiscretizer()
    env = TenderPortfolioEnv()
    rows = []
    for family, seed, pol in q_policies:
        gamma = pol.agent.cfg.gamma
        for sc in df["scenario"].unique():
            d = df[(df["scenario"] == sc) & (df["policy"] == family) & (df["seed"] == seed)]
            cond = next(s.conditions for s in TEST_SCENARIOS if s.name == sc)
            q0 = []
            for ep in range(N_EPISODES):
                obs, info = env.reset(seed=holdout_seeds(N_EPISODES)[ep],
                                      options={"conditions": cond})
                q0.append(max(pol.agent.q[disc(obs), a] for a in (0, 1)
                              if info["action_mask"][a]))
            disc_return = d.groupby("episode").apply(
                lambda g: float(np.sum(g.sort_values("t")["reward"].values
                                       * gamma ** np.arange(len(g)))), include_groups=False)
            rows.append({"scenario": sc, "policy": family, "seed": seed,
                         "q_start_estimate": float(np.mean(q0)),
                         "discounted_return": float(disc_return.mean()),
                         "undiscounted_return": float(d.groupby("episode")["reward"].sum().mean())})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
