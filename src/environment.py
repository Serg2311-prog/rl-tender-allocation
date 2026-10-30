"""Синтетическая среда управления тендерным портфелем (Gymnasium).

Эпизод -- период работы условной организации: последовательно поступает
`horizon` тендеров. На каждом шаге агент видит один тендер и решает,
тратить ли на подготовку заявки часть ограниченного бюджета.

Все случайные величины эпизода (признаки тендеров и исходы) генерируются
в reset() из seed. Исходы задаются равномерными числами u_win, u_risk,
поэтому один и тот же seed даёт одинаковые тендеры и исходы для любых
стратегий (common random numbers) -- сравнение стратегий парное.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import BASE_CONDITIONS, EnvConfig, MarketConditions

SKIP, SUBMIT = 0, 1

# Индексы признаков в векторе наблюдения.
OBS_VALUE, OBS_P, OBS_COST, OBS_RISK, OBS_BUDGET, OBS_STEPS, OBS_AFFORD = range(7)


@dataclass
class Tenders:
    value: np.ndarray
    p_win: np.ndarray
    cost: np.ndarray
    risk: np.ndarray


def expected_reward(value, p_win, cost, risk, penalty_factor: float):
    """E[R | подача] = p * V * (1 - kappa * r) - c."""
    return p_win * value * (1.0 - penalty_factor * risk) - cost


def sample_tenders(rng: np.random.Generator, n: int, cfg: EnvConfig,
                   cond: MarketConditions) -> tuple[Tenders, np.ndarray, np.ndarray]:
    """Генерирует n тендеров и равномерные числа для исходов.

    Базовые случайные величины не зависят от условий рынка, условия только
    масштабируют их. Поэтому при одном seed сценарии отличаются лишь
    исследуемым фактором.
    """
    value = rng.uniform(cfg.value_low, cfg.value_high, n)
    p_base = rng.uniform(cfg.p_low, cfg.p_high, n)
    cost_noise = rng.uniform(0.0, cfg.cost_noise, n)
    risk = rng.uniform(cfg.risk_low, cfg.risk_high, n)
    u_win = rng.uniform(0.0, 1.0, n)
    u_risk = rng.uniform(0.0, 1.0, n)

    p_win = np.clip(cond.p_scale * p_base, cfg.p_clip_low, cfg.p_clip_high)
    cost = cond.cost_scale * (cfg.cost_base + cfg.cost_per_value * value + cost_noise)
    return Tenders(value, p_win, cost, risk), u_win, u_risk


class TenderPortfolioEnv(gym.Env):
    """Наблюдение (Box, float32, 7 признаков):
        [value, p_win, cost, risk, budget, steps_left, can_afford]
    Действие (Discrete(2)): 0 -- отказаться, 1 -- подготовить и подать заявку.

    Подача при нехватке бюджета не выполняется: бюджет не меняется, награда 0,
    info["invalid_action"] = True. Эпизод завершается (terminated) после
    последнего тендера периода.
    """

    metadata = {"render_modes": []}

    def __init__(self, cfg: EnvConfig | None = None,
                 conditions: MarketConditions = BASE_CONDITIONS):
        super().__init__()
        self.cfg = cfg or EnvConfig()
        self.default_conditions = conditions
        self._check_conditions(conditions)

        c = self.cfg
        low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([c.value_high, 1.0, c.max_cost, 1.0, c.max_budget,
                         float(c.horizon), 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self.conditions = conditions
        self._tenders: Tenders | None = None
        self._u_win = self._u_risk = None
        self._t = 0
        self.budget = 0.0

    # ------------------------------------------------------------------
    def _check_conditions(self, cond: MarketConditions) -> None:
        c = self.cfg
        max_cost = cond.cost_scale * (c.cost_base + c.cost_per_value * c.value_high
                                      + c.cost_noise)
        if not (0.0 < cond.initial_budget <= c.max_budget):
            raise ValueError(f"initial_budget must be in (0, {c.max_budget}]")
        if not (0.0 < cond.cost_scale and max_cost <= c.max_cost):
            raise ValueError("cost_scale out of observation bounds")
        if cond.p_scale <= 0.0:
            raise ValueError("p_scale must be positive")

    @property
    def steps_left(self) -> int:
        return self.cfg.horizon - self._t

    def current_tender(self) -> dict:
        t = self._t
        return {
            "value": float(self._tenders.value[t]),
            "p_win": float(self._tenders.p_win[t]),
            "cost": float(self._tenders.cost[t]),
            "risk": float(self._tenders.risk[t]),
        }

    def _can_afford(self) -> bool:
        return self.steps_left > 0 and self._tenders.cost[self._t] <= self.budget

    def _obs(self) -> np.ndarray:
        if self.steps_left > 0:
            t = self._t
            tv = (self._tenders.value[t], self._tenders.p_win[t],
                  self._tenders.cost[t], self._tenders.risk[t])
        else:
            tv = (0.0, 0.0, 0.0, 0.0)   # фиктивный тендер в терминальном состоянии
        return np.array([*tv, self.budget, self.steps_left, float(self._can_afford())],
                        dtype=np.float32)

    def action_mask(self) -> np.ndarray:
        return np.array([1, int(self._can_afford())], dtype=np.int8)

    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        cond = (options or {}).get("conditions", self.default_conditions)
        self._check_conditions(cond)
        self.conditions = cond
        self._tenders, self._u_win, self._u_risk = sample_tenders(
            self.np_random, self.cfg.horizon, self.cfg, cond)
        self._t = 0
        self.budget = float(cond.initial_budget)
        return self._obs(), {"action_mask": self.action_mask(), "budget": self.budget}

    def step(self, action):
        if self._tenders is None:
            raise RuntimeError("call reset() before step()")
        if self.steps_left <= 0:
            raise RuntimeError("episode is over, call reset()")
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")

        t = self._t
        info = {"submitted": False, "won": False, "incident": False,
                "cost": 0.0, "invalid_action": False}
        reward = 0.0
        if int(action) == SUBMIT:
            cost = float(self._tenders.cost[t])
            if cost <= self.budget:
                value = float(self._tenders.value[t])
                won = bool(self._u_win[t] < self._tenders.p_win[t])
                incident = won and bool(self._u_risk[t] < self._tenders.risk[t])
                self.budget -= cost
                reward = (value if won else 0.0) \
                    - (self.cfg.penalty_factor * value if incident else 0.0) - cost
                info.update(submitted=True, won=won, incident=incident, cost=cost)
            else:
                info["invalid_action"] = True

        self._t += 1
        terminated = self.steps_left == 0
        info["action_mask"] = self.action_mask()
        info["budget"] = self.budget
        return self._obs(), float(reward), terminated, False, info
