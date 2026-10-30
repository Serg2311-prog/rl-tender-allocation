"""Единая конфигурация эксперимента.

Все числовые параметры среды, награды, обучения, эвристики, seed и тестовые
сценарии зафиксированы здесь до запуска обучения и оценки. Денежные величины
выражены в условных единицах (у.е.).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
POLICIES_DIR = RESULTS_DIR / "policies"
FIGURES_DIR = PROJECT_ROOT / "report" / "figures"


# --------------------------------------------------------------------------
# Среда и генератор тендеров
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class MarketConditions:
    """Условия рынка, которые меняются между сценариями.

    p_scale         -- множитель вероятности победы (конкуренция на рынке);
    cost_scale      -- множитель стоимости подготовки заявки;
    initial_budget  -- бюджет на подготовку заявок на весь период, у.е.
    """

    p_scale: float = 1.0
    cost_scale: float = 1.0
    initial_budget: float = 40.0


@dataclass(frozen=True)
class EnvConfig:
    """Параметры генерации тендеров и функции награды.

    Тендер i описывается четырьмя признаками:
      value  V ~ U(value_low, value_high)                       -- маржа при победе;
      p_win  p = clip(p_scale * U(p_low, p_high), p_clip)       -- вероятность победы;
      cost   c = cost_scale * (cost_base + cost_per_value * V + U(0, cost_noise))
                                                                -- стоимость подготовки;
      risk   r ~ U(risk_low, risk_high)                         -- вероятность инцидента
                                                                   при исполнении контракта.
    Крупные тендеры дороже готовить (cost_per_value > 0).

    Награда за подачу заявки: R = 1[win] * V - 1[win and incident] * kappa * V - c.
    Ожидаемая награда:        E[R] = p * V * (1 - kappa * r) - c.
    """

    horizon: int = 30                 # тендеров (решений) за эпизод-период
    value_low: float = 10.0
    value_high: float = 100.0
    p_low: float = 0.05
    p_high: float = 0.60
    p_clip_low: float = 0.01
    p_clip_high: float = 0.95
    cost_base: float = 2.0
    cost_per_value: float = 0.08
    cost_noise: float = 5.0
    risk_low: float = 0.0
    risk_high: float = 0.8
    penalty_factor: float = 1.5       # kappa: штраф при инциденте в долях V
    # Границы пространства наблюдений (для корректного Box в Gymnasium).
    max_budget: float = 200.0
    max_cost: float = 80.0


BASE_CONDITIONS = MarketConditions()


@dataclass(frozen=True)
class RandomizationRanges:
    """Диапазоны domain randomization для экспериментального агента.

    В каждом обучающем эпизоде условия выбираются равномерно и независимо.
    """

    p_scale: tuple[float, float] = (0.7, 1.3)
    cost_scale: tuple[float, float] = (0.8, 1.4)
    initial_budget: tuple[float, float] = (25.0, 60.0)


DR_RANGES = RandomizationRanges()


# --------------------------------------------------------------------------
# Дискретизация состояния для табличного Q-learning
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class DiscretizerConfig:
    """Границы интервалов (абсолютные, не зависят от условий рынка).

    pace = budget / (steps_left * cost_ref) -- сколько «средних» заявок на
    один оставшийся тендер позволяет бюджет; cost_ref ~ средняя стоимость
    подготовки в базовых условиях (2 + 0.08*55 + 2.5 = 8.9).
    """

    value_edges: tuple[float, ...] = (40.0, 70.0)
    p_edges: tuple[float, ...] = (0.15, 0.30, 0.45)
    cost_edges: tuple[float, ...] = (6.0, 10.0)
    risk_edges: tuple[float, ...] = (0.25, 0.50)
    pace_edges: tuple[float, ...] = (0.10, 0.20, 0.35, 0.60)
    steps_edges: tuple[int, ...] = (5, 15)   # <=5 конец, 6..15 середина, >15 начало
    cost_ref: float = 9.0


# --------------------------------------------------------------------------
# Q-learning
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class QLearningConfig:
    """Значения gamma и режима alpha выбраны по results/tuning_validation.csv
    (src/tune.py, только валидационные эпизоды)."""

    n_episodes: int = 200_000
    gamma: float = 0.95
    # "visit": alpha = max(alpha_min, (1 + n(s,a)) ** -alpha_power);
    # "constant": alpha = alpha_const.
    alpha_mode: str = "visit"
    alpha_const: float = 0.05
    alpha_power: float = 0.6
    alpha_min: float = 0.01
    eps_start: float = 1.0
    eps_end: float = 0.02
    eps_decay_fraction: float = 0.8   # линейное убывание за долю эпизодов
    eval_every: int = 5_000           # период оценки на валидационных эпизодах
    n_val_episodes: int = 200


# --------------------------------------------------------------------------
# Эвристика (зафиксирована до обучения и оценки RL)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class HeuristicConfig:
    """Подать заявку, если хватает бюджета, ожидаемая прибыль не ниже
    min_roi * стоимость подготовки и риск не выше max_risk."""

    min_roi: float = 0.25
    max_risk: float = 0.60


# --------------------------------------------------------------------------
# Seed и разбиение на обучающие / валидационные / тестовые эпизоды
# --------------------------------------------------------------------------
AGENT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
TRAIN_SEED_STRIDE = 1_000_000          # seed эпизода = agent_seed * stride + ep
VAL_SEED_OFFSET = 500_000_000
TEST_SEED_OFFSET = 1_000_000_000
N_TEST_EPISODES = 500
RANDOM_POLICY_SEED = 12345
ORACLE_SEED = 777_000_000              # выборка тендеров для DP-оракула
ORACLE_SAMPLES = 4000
ORACLE_GRID_STEP = 0.25


def train_episode_seed(agent_seed: int, episode: int) -> int:
    if not 0 <= episode < TRAIN_SEED_STRIDE:
        raise ValueError("episode index out of range")
    return agent_seed * TRAIN_SEED_STRIDE + episode


def validation_seeds(n: int) -> list[int]:
    return [VAL_SEED_OFFSET + i for i in range(n)]


def holdout_seeds(n: int = N_TEST_EPISODES) -> list[int]:
    return [TEST_SEED_OFFSET + i for i in range(n)]


# --------------------------------------------------------------------------
# Тестовые сценарии
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    conditions: MarketConditions
    # Отношение к обучающим условиям: "base" -- совпадает с обучением базового
    # агента; "dr" -- внутри диапазона рандомизации (видел только DR-агент);
    # "out" -- вне всех обучающих диапазонов.
    coverage: str
    group: str = "main"


def _sc(name, title, coverage, **kw) -> Scenario:
    return Scenario(name, title, MarketConditions(**kw), coverage)


TEST_SCENARIOS: tuple[Scenario, ...] = (
    _sc("base", "Исходные условия", "base"),
    _sc("p_0.80", "Вероятность победы x0.8", "dr", p_scale=0.8),
    _sc("p_0.55", "Вероятность победы x0.55", "out", p_scale=0.55),
    _sc("p_1.45", "Вероятность победы x1.45", "out", p_scale=1.45),
    _sc("cost_1.25", "Стоимость подготовки x1.25", "dr", cost_scale=1.25),
    _sc("cost_1.70", "Стоимость подготовки x1.7", "out", cost_scale=1.7),
    _sc("budget_30", "Бюджет 30 у.е.", "dr", initial_budget=30.0),
    _sc("budget_15", "Бюджет 15 у.е.", "out", initial_budget=15.0),
    _sc("budget_90", "Бюджет 90 у.е.", "out", initial_budget=90.0),
    _sc("combo_dr", "p x0.8, cost x1.3, бюджет 30", "dr",
        p_scale=0.8, cost_scale=1.3, initial_budget=30.0),
    _sc("combo_out_hard", "p x0.55, cost x1.7, бюджет 15", "out",
        p_scale=0.55, cost_scale=1.7, initial_budget=15.0),
    _sc("combo_out_rich", "p x0.55, бюджет 90", "out",
        p_scale=0.55, initial_budget=90.0),
)

# Однофакторные развёртки для графиков зависимости (остальное -- как в base).
SWEEPS: dict[str, tuple[float, ...]] = {
    "initial_budget": (10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 90.0, 120.0),
    "cost_scale": (0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0),
    "p_scale": (0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45, 1.6),
}


def sweep_scenarios() -> list[Scenario]:
    out = []
    for factor, values in SWEEPS.items():
        lo, hi = getattr(DR_RANGES, factor)
        base_value = getattr(BASE_CONDITIONS, factor)
        for v in values:
            cond = MarketConditions(**{**asdict(BASE_CONDITIONS), factor: v})
            if v == base_value:
                coverage = "base"
            elif lo <= v <= hi:
                coverage = "dr"
            else:
                coverage = "out"
            out.append(Scenario(f"sweep_{factor}_{v:g}", f"{factor}={v:g}",
                                cond, coverage, group=f"sweep_{factor}"))
    return out


def config_snapshot() -> dict:
    """Все параметры для сохранения рядом с результатами."""
    return {
        "env": asdict(EnvConfig()),
        "base_conditions": asdict(BASE_CONDITIONS),
        "dr_ranges": asdict(DR_RANGES),
        "discretizer": asdict(DiscretizerConfig()),
        "qlearning": asdict(QLearningConfig()),
        "heuristic": asdict(HeuristicConfig()),
        "seeds": {
            "agent_seeds": list(AGENT_SEEDS),
            "train_seed_stride": TRAIN_SEED_STRIDE,
            "val_seed_offset": VAL_SEED_OFFSET,
            "test_seed_offset": TEST_SEED_OFFSET,
            "n_test_episodes": N_TEST_EPISODES,
            "random_policy_seed": RANDOM_POLICY_SEED,
            "oracle_seed": ORACLE_SEED,
        },
        "test_scenarios": [
            {**asdict(s), "conditions": asdict(s.conditions)} for s in TEST_SCENARIOS
        ],
        "sweeps": {k: list(v) for k, v in SWEEPS.items()},
    }
