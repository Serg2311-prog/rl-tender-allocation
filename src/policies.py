"""Стратегии для сравнения.

Общий интерфейс: begin_episode(conditions) перед эпизодом и
act(obs, info) -> действие. Все стратегии выбирают только допустимые
действия по info["action_mask"].
"""
from __future__ import annotations

import numpy as np

from .agent import QLearningAgent, StateDiscretizer, masked_argmax
from .config import (ORACLE_GRID_STEP, ORACLE_SAMPLES, ORACLE_SEED, EnvConfig,
                     HeuristicConfig, MarketConditions)
from .environment import (OBS_COST, OBS_P, OBS_RISK, OBS_STEPS,
                          OBS_VALUE, SKIP, SUBMIT, expected_reward, sample_tenders)


class Policy:
    name = "policy"

    def begin_episode(self, conditions: MarketConditions) -> None:
        pass

    def act(self, obs, info) -> int:
        raise NotImplementedError


class AlwaysSkipPolicy(Policy):
    name = "always_skip"

    def act(self, obs, info) -> int:
        return SKIP


class AlwaysSubmitPolicy(Policy):
    name = "always_submit"

    def act(self, obs, info) -> int:
        return SUBMIT if info["action_mask"][SUBMIT] else SKIP


class RandomPolicy(Policy):
    """Равновероятный выбор среди допустимых действий."""
    name = "random"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, info) -> int:
        valid = np.flatnonzero(info["action_mask"])
        return int(valid[self.rng.integers(len(valid))])


class HeuristicPolicy(Policy):
    """Правило аналитика: подавать, если бюджета хватает,
    E[R] >= min_roi * c и r <= max_risk. Не учитывает будущие тендеры."""
    name = "heuristic"

    def __init__(self, env_cfg: EnvConfig | None = None, cfg: HeuristicConfig | None = None):
        self.env_cfg = env_cfg or EnvConfig()
        self.cfg = cfg or HeuristicConfig()

    def act(self, obs, info) -> int:
        if not info["action_mask"][SUBMIT]:
            return SKIP
        value, p, cost, risk = (float(obs[i]) for i in (OBS_VALUE, OBS_P, OBS_COST, OBS_RISK))
        ev = expected_reward(value, p, cost, risk, self.env_cfg.penalty_factor)
        return SUBMIT if (ev >= self.cfg.min_roi * cost and risk <= self.cfg.max_risk) else SKIP


class QPolicy(Policy):
    """Жадная политика по обученной Q-таблице (без дообучения)."""

    def __init__(self, agent: QLearningAgent, discretizer: StateDiscretizer, name: str = "q"):
        self.agent = agent
        self.disc = discretizer
        self.name = name
        self.decisions = 0
        self.unseen_decisions = 0   # решения в состояниях, не посещённых при обучении

    def act(self, obs, info) -> int:
        s = self.disc(obs)
        mask = info["action_mask"]
        self.decisions += 1
        if not any(self.agent.visits[s, a] for a in range(len(mask)) if mask[a]):
            self.unseen_decisions += 1
        return masked_argmax(self.agent.q[s], mask)


class OraclePolicy(Policy):
    """Справочная верхняя граница: оптимальная по ожиданию стратегия,
    знающая распределение тендеров текущего сценария (но не будущие
    тендеры и не исходы). Функция ценности W_k(b) -- ожидаемая будущая
    награда при k оставшихся тендерах и бюджете b:
        W_k(b) = E_x[max(W_{k-1}(b), 1[c<=b] (E[R|x] + W_{k-1}(b - c)))],
    ожидание оценивается по выборке тендеров, b -- на сетке с шагом grid_step.
    """
    name = "oracle"

    def __init__(self, env_cfg: EnvConfig | None = None, n_samples: int = ORACLE_SAMPLES,
                 grid_step: float = ORACLE_GRID_STEP, seed: int = ORACLE_SEED):
        self.env_cfg = env_cfg or EnvConfig()
        self.n_samples = n_samples
        self.grid_step = grid_step
        self.seed = seed
        self._cache: dict[MarketConditions, tuple[np.ndarray, np.ndarray]] = {}
        self.grid = self.W = None

    def _solve(self, cond: MarketConditions):
        cfg = self.env_cfg
        rng = np.random.default_rng(self.seed)
        tenders, _, _ = sample_tenders(rng, self.n_samples, cfg, cond)
        ev = expected_reward(tenders.value, tenders.p_win, tenders.cost, tenders.risk,
                             cfg.penalty_factor)
        grid = np.arange(0.0, cond.initial_budget + self.grid_step, self.grid_step)
        W = np.zeros((cfg.horizon + 1, grid.size))
        rem = grid[:, None] - tenders.cost[None, :]
        feasible = rem >= 0.0
        rem = np.clip(rem, 0.0, None)
        for k in range(1, cfg.horizon + 1):
            prev = W[k - 1]
            submit = np.where(feasible, ev[None, :] + np.interp(rem, grid, prev), -np.inf)
            W[k] = np.maximum(prev[:, None], submit).mean(axis=1)
        return grid, W

    def begin_episode(self, conditions: MarketConditions) -> None:
        if conditions not in self._cache:
            self._cache[conditions] = self._solve(conditions)
        self.grid, self.W = self._cache[conditions]

    def act(self, obs, info) -> int:
        if not info["action_mask"][SUBMIT]:
            return SKIP
        value, p, cost, risk = (float(obs[i]) for i in (OBS_VALUE, OBS_P, OBS_COST, OBS_RISK))
        budget = float(info["budget"])
        prev = self.W[int(round(float(obs[OBS_STEPS]))) - 1]
        ev = expected_reward(value, p, cost, risk, self.env_cfg.penalty_factor)
        skip_value = np.interp(budget, self.grid, prev)
        submit_value = ev + np.interp(budget - cost, self.grid, prev)
        return SUBMIT if submit_value > skip_value else SKIP
