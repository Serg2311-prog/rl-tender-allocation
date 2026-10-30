"""Подбор гиперпараметров Q-learning только на валидационных эпизодах.

Тестовые сценарии здесь не используются. Обучение -- в базовых условиях,
seed агента TUNE_SEED не входит в AGENT_SEEDS основного эксперимента.

Запуск: python -m src.tune
"""
from __future__ import annotations

import itertools
import json
import time
from dataclasses import asdict, replace

import pandas as pd

from .config import RESULTS_DIR, QLearningConfig
from .train import train_agent

TUNE_SEED = 99
TUNE_EPISODES = 100_000
GRID = {
    "alpha": [("constant", 0.05), ("constant", 0.1), ("visit", None)],
    "gamma": [0.95, 1.0],
}


def main() -> None:
    rows = []
    t0 = time.time()
    for (mode, a), gamma in itertools.product(GRID["alpha"], GRID["gamma"]):
        cfg = replace(QLearningConfig(), n_episodes=TUNE_EPISODES, gamma=gamma,
                      alpha_mode=mode, alpha_const=a or QLearningConfig().alpha_const,
                      eval_every=10_000, n_val_episodes=500)
        _, hist = train_agent("base", TUNE_SEED, cfg, verbose=False)
        label = f"{mode}" + (f"(a={a})" if a else "") + f", gamma={gamma}"
        for v in hist["validation"]:
            rows.append({"config": label, "alpha_mode": mode, "alpha_const": a,
                         "gamma": gamma, **v})
        print(f"[{time.time() - t0:5.0f}s] {label}: "
              f"val={hist['validation'][-1]['val_reward_mean']:.2f}")
    df = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "tuning_validation.csv", index=False)
    final = df[df["episode"] == TUNE_EPISODES].sort_values("val_reward_mean", ascending=False)
    print(final[["config", "val_reward_mean", "val_reward_std"]].to_string(index=False))
    (RESULTS_DIR / "tuning_grid.json").write_text(
        json.dumps({"seed": TUNE_SEED, "episodes": TUNE_EPISODES,
                    "base_config": asdict(QLearningConfig()), "grid": GRID}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
