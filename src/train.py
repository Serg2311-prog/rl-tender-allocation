"""Обучение табличных Q-агентов в двух условиях.

base -- все обучающие эпизоды в исходных условиях рынка;
dr   -- в каждом эпизоде условия (p_scale, cost_scale, бюджет) выбираются
        случайно из DR_RANGES (domain randomization).
Остальное совпадает: число эпизодов, гиперпараметры, seed эпизодов.

Запуск: python -m src.train [--episodes N] [--seeds 0 1 2] [--conditions base dr]
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .agent import QLearningAgent, StateDiscretizer
from .config import (AGENT_SEEDS, BASE_CONDITIONS, DR_RANGES, ORACLE_SEED,
                     POLICIES_DIR, RANDOM_POLICY_SEED, RESULTS_DIR, EnvConfig,
                     MarketConditions, QLearningConfig, train_episode_seed,
                     validation_seeds)
from .environment import TenderPortfolioEnv
from .evaluate import run_episodes
from .policies import HeuristicPolicy, OraclePolicy, QPolicy, RandomPolicy

HISTORY_BLOCK = 500   # усреднение обучающей награды по блокам эпизодов


def sample_conditions(rng: np.random.Generator) -> MarketConditions:
    return MarketConditions(
        p_scale=float(rng.uniform(*DR_RANGES.p_scale)),
        cost_scale=float(rng.uniform(*DR_RANGES.cost_scale)),
        initial_budget=float(rng.uniform(*DR_RANGES.initial_budget)),
    )


def train_agent(condition: str, agent_seed: int, qcfg: QLearningConfig,
                env_cfg: EnvConfig | None = None, verbose: bool = True):
    if condition not in ("base", "dr"):
        raise ValueError(condition)
    env_cfg = env_cfg or EnvConfig()
    env = TenderPortfolioEnv(env_cfg)
    val_env = TenderPortfolioEnv(env_cfg)
    disc = StateDiscretizer()
    agent = QLearningAgent(disc.n_states, 2, qcfg, seed=agent_seed)
    cond_rng = np.random.default_rng([agent_seed, 2024])
    val_seeds = validation_seeds(qcfg.n_val_episodes)

    ep_rewards = np.zeros(qcfg.n_episodes)
    abs_td = np.zeros(qcfg.n_episodes)
    val_curve = []
    t0 = time.time()
    for ep in range(qcfg.n_episodes):
        eps = agent.epsilon(ep)
        cond = BASE_CONDITIONS if condition == "base" else sample_conditions(cond_rng)
        obs, info = env.reset(seed=train_episode_seed(agent_seed, ep),
                              options={"conditions": cond})
        s = disc(obs)
        total, td_sum, n = 0.0, 0.0, 0
        done = False
        while not done:
            a = agent.act(s, info["action_mask"], eps)
            obs, r, terminated, truncated, info = env.step(a)
            s_next = disc(obs)
            td_sum += abs(agent.update(s, a, r, s_next, terminated, info["action_mask"]))
            total += r
            n += 1
            s = s_next
            done = terminated or truncated
        ep_rewards[ep] = total
        abs_td[ep] = td_sum / n

        if (ep + 1) % qcfg.eval_every == 0 or ep + 1 == qcfg.n_episodes:
            val = run_episodes(val_env, QPolicy(agent, disc), val_seeds, BASE_CONDITIONS)
            val_curve.append({"episode": ep + 1, "epsilon": eps,
                              "val_reward_mean": float(val["reward"].mean()),
                              "val_reward_std": float(val["reward"].std(ddof=1))})
            if verbose:
                print(f"  [{condition} seed={agent_seed}] ep {ep + 1:>6} eps={eps:.3f} "
                      f"train(last {qcfg.eval_every})={ep_rewards[ep + 1 - qcfg.eval_every:ep + 1].mean():7.2f} "
                      f"val={val_curve[-1]['val_reward_mean']:7.2f}  ({time.time() - t0:.0f}s)")

    n_blocks = qcfg.n_episodes // HISTORY_BLOCK
    history = {
        "condition": condition, "agent_seed": agent_seed,
        "qlearning": asdict(qcfg),
        "block_size": HISTORY_BLOCK,
        "train_reward_blocks": ep_rewards[:n_blocks * HISTORY_BLOCK]
            .reshape(n_blocks, HISTORY_BLOCK).mean(axis=1).tolist(),
        "abs_td_blocks": abs_td[:n_blocks * HISTORY_BLOCK]
            .reshape(n_blocks, HISTORY_BLOCK).mean(axis=1).tolist(),
        "validation": val_curve,
        "visited_state_share": float((agent.visits.sum(axis=1) > 0).mean()),
        "train_seconds": time.time() - t0,
    }
    return agent, history


def validation_baselines(n_val: int) -> dict:
    env_cfg = EnvConfig()
    env = TenderPortfolioEnv(env_cfg)
    seeds = validation_seeds(n_val)
    out = {}
    for name, pol in (("random", RandomPolicy(RANDOM_POLICY_SEED)),
                      ("heuristic", HeuristicPolicy(env_cfg)),
                      ("oracle", OraclePolicy(env_cfg, seed=ORACLE_SEED))):
        r = run_episodes(env, pol, seeds, BASE_CONDITIONS)["reward"]
        out[name] = {"mean": float(r.mean()), "std": float(r.std(ddof=1))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=None, help="число обучающих эпизодов")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(AGENT_SEEDS))
    ap.add_argument("--conditions", nargs="+", default=["base", "dr"], choices=["base", "dr"])
    ap.add_argument("--out", default=str(POLICIES_DIR), help="каталог для Q-таблиц")
    args = ap.parse_args()

    qcfg = QLearningConfig()
    if args.episodes:
        qcfg = replace(qcfg, n_episodes=args.episodes,
                       eval_every=min(qcfg.eval_every, args.episodes))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for condition in args.conditions:
        for seed in args.seeds:
            print(f"Training {condition}, seed {seed}, {qcfg.n_episodes} episodes")
            agent, history = train_agent(condition, seed, qcfg)
            meta = {"condition": condition, "agent_seed": seed}
            agent.save(out_dir / f"q_{condition}_seed{seed}.npz", meta)
            (out_dir / f"history_{condition}_seed{seed}.json").write_text(
                json.dumps(history, indent=1), encoding="utf-8")

    # Пробные запуски с --out не должны перезаписывать результаты основного эксперимента.
    baselines_dir = RESULTS_DIR if out_dir.resolve() == POLICIES_DIR.resolve() else out_dir
    baselines_dir.mkdir(parents=True, exist_ok=True)
    (baselines_dir / "validation_baselines.json").write_text(
        json.dumps(validation_baselines(qcfg.n_val_episodes), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
