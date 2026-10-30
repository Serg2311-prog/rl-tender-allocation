"""Табличный Q-learning и дискретизация наблюдений."""
from __future__ import annotations

import json
from bisect import bisect_left, bisect_right
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import DiscretizerConfig, QLearningConfig
from .environment import (OBS_AFFORD, OBS_BUDGET, OBS_COST, OBS_P, OBS_RISK,
                          OBS_STEPS, OBS_VALUE)


class StateDiscretizer:
    """Переводит непрерывное наблюдение в индекс строки Q-таблицы.

    Признаки: V, p, c, r (по абсолютным границам), темп бюджета
    pace = budget / (steps_left * cost_ref), стадия периода и флаг
    доступности ресурса. Число состояний = 3*4*3*3*5*3*2 = 3240.
    """

    def __init__(self, cfg: DiscretizerConfig | None = None):
        self.cfg = cfg or DiscretizerConfig()
        c = self.cfg
        self.sizes = (len(c.value_edges) + 1, len(c.p_edges) + 1,
                      len(c.cost_edges) + 1, len(c.risk_edges) + 1,
                      len(c.pace_edges) + 1, len(c.steps_edges) + 1, 2)
        self.n_states = int(np.prod(self.sizes))

    def components(self, obs) -> tuple[int, ...]:
        c = self.cfg
        steps = int(round(float(obs[OBS_STEPS])))
        pace = float(obs[OBS_BUDGET]) / (max(steps, 1) * c.cost_ref)
        return (bisect_right(c.value_edges, float(obs[OBS_VALUE])),
                bisect_right(c.p_edges, float(obs[OBS_P])),
                bisect_right(c.cost_edges, float(obs[OBS_COST])),
                bisect_right(c.risk_edges, float(obs[OBS_RISK])),
                bisect_right(c.pace_edges, pace),
                bisect_left(c.steps_edges, steps),
                int(obs[OBS_AFFORD] > 0.5))

    def __call__(self, obs) -> int:
        idx = 0
        for comp, size in zip(self.components(obs), self.sizes):
            idx = idx * size + comp
        return idx


def masked_argmax(q_row: np.ndarray, mask) -> int:
    """Жадное действие среди допустимых; при равенстве -- меньший индекс
    (отказ от участия), в том числе в не посещённых при обучении состояниях."""
    best, best_q = 0, -np.inf
    for a in range(len(q_row)):
        if mask[a] and q_row[a] > best_q:
            best, best_q = a, q_row[a]
    return best


class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int = 2,
                 cfg: QLearningConfig | None = None, seed: int = 0):
        self.cfg = cfg or QLearningConfig()
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)
        self.visits = np.zeros((n_states, n_actions), dtype=np.int64)
        self.rng = np.random.default_rng(seed)

    def epsilon(self, episode: int) -> float:
        c = self.cfg
        decay_eps = max(1, int(c.eps_decay_fraction * c.n_episodes))
        frac = min(1.0, episode / decay_eps)
        return c.eps_start + frac * (c.eps_end - c.eps_start)

    def learning_rate(self, s: int, a: int) -> float:
        c = self.cfg
        if c.alpha_mode == "constant":
            return c.alpha_const
        if c.alpha_mode == "visit":
            return max(c.alpha_min, (1.0 + self.visits[s, a]) ** -c.alpha_power)
        raise ValueError(f"unknown alpha_mode {c.alpha_mode!r}")

    def act(self, s: int, mask, epsilon: float = 0.0) -> int:
        """epsilon-greedy среди допустимых действий."""
        if epsilon > 0.0 and self.rng.random() < epsilon:
            valid = np.flatnonzero(mask)
            return int(valid[self.rng.integers(len(valid))])
        return masked_argmax(self.q[s], mask)

    def update(self, s: int, a: int, r: float, s_next: int, terminated: bool,
               next_mask) -> float:
        """Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a)).

        max берётся только по допустимым в s' действиям; после терминального
        шага бутстрэп не выполняется. Возвращает TD-ошибку.
        """
        target = r
        if not terminated:
            target += self.cfg.gamma * max(
                self.q[s_next, b] for b in range(self.q.shape[1]) if next_mask[b])
        td = target - self.q[s, a]
        self.q[s, a] += self.learning_rate(s, a) * td
        self.visits[s, a] += 1
        return float(td)

    # ------------------------------------------------------------------
    def save(self, path: Path, meta: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, q=self.q, visits=self.visits)
        info = {"qlearning": asdict(self.cfg), **(meta or {})}
        path.with_suffix(".json").write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                             encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "QLearningAgent":
        data = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        agent = cls(data["q"].shape[0], data["q"].shape[1],
                    QLearningConfig(**meta["qlearning"]))
        agent.q = data["q"]
        agent.visits = data["visits"]
        return agent
