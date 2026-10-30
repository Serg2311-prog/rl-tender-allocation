"""Графики и таблицы по сохранённым результатам (results/*.csv, *.json).

Запуск: python -m src.report
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import (AGENT_SEEDS, BASE_CONDITIONS, DR_RANGES, FIGURES_DIR,  # noqa: E402
                     POLICIES_DIR, RESULTS_DIR, TEST_SCENARIOS)

NAMES = {
    "random": "Случайная",
    "heuristic": "Эвристика",
    "q_base": "Q-learning (базовое обучение)",
    "q_dr": "Q-learning (рандомизация условий)",
    "oracle": "DP-оракул (знает распределение)",
    "always_skip": "Всегда отказ",
    "always_submit": "Всегда подавать",
}
COLORS = {"random": "#9e9e9e", "heuristic": "#ef8a17", "q_base": "#1f77b4",
          "q_dr": "#2ca02c", "oracle": "#222222", "always_skip": "#c7c7c7",
          "always_submit": "#d62728"}
COVERAGE = {"base": "[B]", "dr": "[DR]", "out": "[OUT]"}
MAIN_POLICIES = ["random", "heuristic", "q_base", "q_dr", "oracle"]
UNIT = "у.е."


def _save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=130)
    plt.close(fig)
    print(f"  saved report/figures/{name}")


def _load_histories() -> dict[str, list[dict]]:
    out = {}
    for cond in ("base", "dr"):
        out[cond] = [json.loads((POLICIES_DIR / f"history_{cond}_seed{s}.json")
                                .read_text(encoding="utf-8")) for s in AGENT_SEEDS]
    return out


# --------------------------------------------------------------------------
def fig_learning_curves(hist, baselines) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    for cond in ("base", "dr"):
        ep = np.array([v["episode"] for v in hist[cond][0]["validation"]])
        vals = np.array([[v["val_reward_mean"] for v in h["validation"]] for h in hist[cond]])
        m, s = vals.mean(0), vals.std(0, ddof=1)
        fam = f"q_{cond}"
        ax.plot(ep, m, color=COLORS[fam], label=f"{NAMES[fam]}: среднее по {len(vals)} seed")
        ax.fill_between(ep, m - s, m + s, color=COLORS[fam], alpha=0.2)
    for fam in ("heuristic", "oracle", "random"):
        ax.axhline(baselines[fam]["mean"], color=COLORS[fam], ls="--", lw=1.2,
                   label=f"{NAMES[fam]} (без обучения)")
    ax.set_title("Жадная политика на валидационных эпизодах\n"
                 "(200 эпизодов, исходные условия рынка; полоса = ±1 std по seed)")
    ax.set_xlabel("Обучающий эпизод")
    ax.set_ylabel(f"Средняя суммарная награда за эпизод, {UNIT}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for cond in ("base", "dr"):
        blocks = np.array([h["train_reward_blocks"] for h in hist[cond]])
        bs = hist[cond][0]["block_size"]
        x = (np.arange(blocks.shape[1]) + 1) * bs
        m, s = blocks.mean(0), blocks.std(0, ddof=1)
        fam = f"q_{cond}"
        ax.plot(x, m, color=COLORS[fam], lw=1, label=NAMES[fam])
        ax.fill_between(x, m - s, m + s, color=COLORS[fam], alpha=0.2)
    ax2 = ax.twinx()
    eps = [hist["base"][0]["qlearning"][k] for k in ("eps_start", "eps_end", "eps_decay_fraction", "n_episodes")]
    xe = np.linspace(0, eps[3], 200)
    ax2.plot(xe, eps[0] + np.minimum(1, xe / (eps[2] * eps[3])) * (eps[1] - eps[0]),
             color="#7f7f7f", ls=":", label="epsilon")
    ax2.set_ylabel("epsilon (доля случайных действий)")
    ax.set_title(f"Награда в обучающих эпизодах (с исследованием)\nсреднее по блокам из {bs} эпизодов, полоса = ±1 std по seed")
    ax.set_xlabel("Обучающий эпизод")
    ax.set_ylabel(f"Средняя награда за эпизод, {UNIT}")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    _save(fig, "fig1_learning_curves.png")


def fig_base_comparison(summary, per_seed) -> None:
    base = summary[summary["scenario"] == "base"].set_index("policy")
    order = ["always_skip", "always_submit", "random", "heuristic", "q_base", "q_dr", "oracle"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(order))
    ax.bar(x, base.loc[order, "reward_mean"], yerr=base.loc[order, "reward_ci95_episodes"],
           capsize=4, color=[COLORS[p] for p in order], alpha=0.85)
    for i, fam in enumerate(order):
        m, ci = base.loc[fam, "reward_mean"], base.loc[fam, "reward_ci95_episodes"]
        if fam.startswith("q_"):
            pts = per_seed.query("scenario == 'base' and policy == @fam")["reward_mean"]
            ax.scatter(np.full(len(pts), i) + np.linspace(-0.15, 0.15, len(pts)), pts,
                       color="white", edgecolor="black", zorder=3, s=25,
                       label="отдельные seed" if fam == "q_base" else None)
            ax.errorbar(i + 0.3, m, yerr=base.loc[fam, "reward_ci95_seeds"], fmt="none",
                        ecolor="#6a3d9a", elinewidth=2.5, capsize=3,
                        label="95% ДИ по 5 seed обучения" if fam == "q_base" else None)
        ax.text(i + 0.12, m + ci + 1 if m >= 0 else m - ci - 3.5, f"{m:.1f}",
                ha="left", fontsize=9)
    ax.set_xticks(x, [NAMES[p] for p in order], rotation=15, fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"Средняя суммарная награда за эпизод, {UNIT}")
    ax.set_title("Исходные условия: сравнение стратегий на 500 тестовых эпизодах\n"
                 "(чёрные усы: 95% ДИ по тестовым эпизодам для всех стратегий; "
                 "фиолетовые: 95% ДИ по seed для Q-learning)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "fig2_base_comparison.png")


def fig_robustness(summary, paired) -> None:
    names = [s.name for s in TEST_SCENARIOS]
    labels = [f"{COVERAGE[s.coverage]} {s.title}" for s in TEST_SCENARIOS]
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ax = axes[0]
    width = 0.17
    for k, fam in enumerate(MAIN_POLICIES):
        d = summary[summary["policy"] == fam].set_index("scenario").loc[names]
        ax.bar(np.arange(len(names)) + (k - 2) * width, d["reward_mean"], width,
               yerr=d["reward_ci95_episodes"], capsize=2, color=COLORS[fam], label=NAMES[fam])
    ax.set_xticks(np.arange(len(names)), labels, rotation=25, ha="right", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"Средняя суммарная награда, {UNIT}")
    ax.set_title("Тестовые сценарии (500 эпизодов на сценарий, усы = 95% ДИ по эпизодам)\n"
                 "[B] = условия обучения базового агента, [DR] = внутри диапазона рандомизации, "
                 "[OUT] = вне всех обучающих диапазонов")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    for k, fam in enumerate(("q_base", "q_dr")):
        d = paired[paired["policy"] == fam].set_index("scenario").loc[names]
        xs = np.arange(len(names)) + (k - 0.5) * 0.24
        ax.errorbar(xs, d["diff_vs_heuristic"], yerr=d["diff_ci95_episodes"], fmt="none",
                    ecolor=COLORS[fam], elinewidth=1, alpha=0.5, capsize=2)
        ax.errorbar(xs, d["diff_vs_heuristic"], yerr=d["diff_ci95_seeds"], fmt="o",
                    elinewidth=2.5, capsize=4, color=COLORS[fam], label=NAMES[fam])
    ax.axhline(0, color=COLORS["heuristic"], lw=1.5, ls="--", label="уровень эвристики")
    ax.set_xticks(np.arange(len(names)), labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel(f"Награда агента минус награда эвристики, {UNIT}")
    ax.set_title("Разность Q-learning и эвристики (парно, на одних и тех же эпизодах)\n"
                 "толстые усы: 95% ДИ по 5 seed обучения; тонкие: 95% ДИ по 500 тестовым эпизодам")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "fig3_robustness.png")


def fig_sweeps(summary) -> None:
    factors = [("initial_budget", f"Начальный бюджет на период, {UNIT}"),
               ("cost_scale", "Множитель стоимости подготовки, раз"),
               ("p_scale", "Множитель вероятности победы, раз")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, (factor, xlabel) in zip(axes, factors):
        d = summary[summary["group"] == f"sweep_{factor}"]
        for fam in MAIN_POLICIES:
            g = d[d["policy"] == fam].sort_values(factor)
            ax.plot(g[factor], g["reward_mean"], marker="o", ms=4, color=COLORS[fam],
                    ls="--" if fam == "oracle" else "-", label=NAMES[fam])
            if fam.startswith("q_"):
                ax.fill_between(g[factor], g["reward_mean"] - g["reward_seed_std"],
                                g["reward_mean"] + g["reward_seed_std"], color=COLORS[fam],
                                alpha=0.2)
        lo, hi = getattr(DR_RANGES, factor)
        ax.axvspan(lo, hi, color=COLORS["q_dr"], alpha=0.08, label="диапазон рандомизации")
        ax.axvline(getattr(BASE_CONDITIONS, factor), color=COLORS["q_base"], ls=":",
                   label="условия базового обучения")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(f"Средняя суммарная награда, {UNIT}")
        ax.set_title(f"Зависимость от фактора «{xlabel.split(',')[0]}»\n"
                     "(остальные условия -- исходные; полоса Q-learning = ±1 std по seed)")
        ax.axhline(0, color="black", lw=0.6)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left")
    _save(fig, "fig4_sweeps.png")


def fig_seed_spread(hist, per_seed) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax = axes[0]
    for cond in ("base", "dr"):
        fam = f"q_{cond}"
        for i, h in enumerate(hist[cond]):
            ep = [v["episode"] for v in h["validation"]]
            ax.plot(ep, [v["val_reward_mean"] for v in h["validation"]], color=COLORS[fam],
                    alpha=0.6, lw=1, label=NAMES[fam] if i == 0 else None)
    ax.set_title("Кривые обучения отдельных seed (валидация, исходные условия)")
    ax.set_xlabel("Обучающий эпизод")
    ax.set_ylabel(f"Средняя награда на 200 эпизодах, {UNIT}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = [s.name for s in TEST_SCENARIOS]
    heur = per_seed[per_seed["policy"] == "heuristic"].set_index("scenario")["reward_mean"]
    for k, fam in enumerate(("q_base", "q_dr")):
        d = per_seed[per_seed["policy"] == fam]
        for j, sc in enumerate(names):
            vals = d[d["scenario"] == sc]["reward_mean"] - heur[sc]
            ax.scatter(np.full(len(vals), j) + (k - 0.5) * 0.25, vals, color=COLORS[fam],
                       s=18, label=NAMES[fam] if j == 0 else None)
    ax.axhline(0, color=COLORS["heuristic"], ls="--", label="уровень эвристики")
    ax.set_xticks(np.arange(len(names)), names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel(f"Награда seed минус награда эвристики, {UNIT}")
    ax.set_title("Разброс по seed обучения на тестовых сценариях (точка = один seed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "fig5_seed_spread.png")


def fig_behavior(rate, path) -> None:
    fams = ["heuristic", "q_base", "q_dr", "oracle"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, stage in zip(axes[0], ("начало (>15)", "конец (<=5)")):
        d = rate[(rate["scenario"] == "base") & (rate["stage"] == stage)]
        bins = list(dict.fromkeys(d["ev_bin"]))
        for fam in fams:
            g = d[d["policy"] == fam].groupby("ev_bin", observed=True)
            # средневзвешенно по seed (для Q-агентов строки -- 5 seed)
            r = g.apply(lambda x: np.average(x["submit_rate"], weights=x["n"]),
                        include_groups=False).reindex(bins)
            ax.plot(bins, r.values * 100, marker="o", color=COLORS[fam], label=NAMES[fam],
                    ls="--" if fam == "oracle" else "-")
        ax.set_title(f"Доля подач среди доступных по бюджету тендеров\nисходные условия, стадия: {stage}")
        ax.set_xlabel(f"Ожидаемая прибыль тендера E[R], {UNIT}")
        ax.set_ylabel("Доля решений «подать», %")
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=8)
    for ax, sc in zip(axes[1], ("base", "budget_90")):
        d = path[path["scenario"] == sc]
        for fam in fams:
            g = d[d["policy"] == fam].groupby("t")["mean"].mean()
            ax.plot(g.index + 1, g.values, color=COLORS[fam], label=NAMES[fam],
                    ls="--" if fam == "oracle" else "-")
        title = {"base": "исходные условия (бюджет 40)", "budget_90": "бюджет 90 у.е. [OUT]"}[sc]
        ax.set_title(f"Средний остаток бюджета перед решением, {title}")
        ax.set_xlabel("Номер тендера в периоде")
        ax.set_ylabel(f"Остаток бюджета, {UNIT}")
        ax.grid(alpha=0.3)
    axes[1][0].legend(fontsize=8)
    _save(fig, "fig6_behavior.png")


def fig_example_episode(example) -> None:
    fams = ["heuristic", "q_base", "q_dr", "oracle"]
    first = example[example["policy"] == "heuristic"].sort_values("t")
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 1]})
    ax = axes[0]
    t = first["t"].values + 1
    ax.bar(t, first["ev"], color=np.where(first["ev"] > 0, "#a6d96a", "#f4a582"),
           edgecolor="gray", label="E[R] тендера")
    for k, fam in enumerate(fams):
        d = example[(example["policy"] == fam) & (example["action"] == 1)]
        y = first["ev"].max() + 3 + 3 * k
        lost, won = d[~d["won"]], d[d["won"]]
        ax.scatter(lost["t"] + 1, np.full(len(lost), y), marker="s", s=45,
                   facecolor="white", edgecolor=COLORS[fam], linewidth=1.8,
                   label=f"{NAMES[fam]}: подано {len(d)}, побед {len(won)}")
        ax.scatter(won["t"] + 1, np.full(len(won), y), marker="*", s=170,
                   color=COLORS[fam], edgecolor="black", zorder=3)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel(f"Ожидаемая прибыль E[R], {UNIT}")
    ax.set_title("Пример тестового эпизода (исходные условия, Q-агенты seed 0)\n"
                 "□ -- заявка подана и проиграна, ★ -- заявка подана и выиграна")
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    ax.grid(alpha=0.3)
    ax = axes[1]
    for fam, lw in zip(fams, (2, 4, 1.8, 1.8)):
        d = example[example["policy"] == fam].sort_values("t")
        after = d["budget_before"] - np.where(d["action"] == 1, d["cost"], 0.0)
        ax.step(np.r_[0, d["t"] + 1], np.r_[d["budget_before"].iloc[0], after], where="post",
                color=COLORS[fam], lw=lw, alpha=0.6 if fam == "q_base" else 1.0,
                label=f"{NAMES[fam]}: итог эпизода {d['reward'].sum():.1f} {UNIT}",
                ls="--" if fam == "oracle" else "-")
    ax.set_title("Остаток бюджета после решения по каждому тендеру")
    ax.set_xlabel("Номер тендера в периоде")
    ax.set_ylabel(f"Остаток бюджета, {UNIT}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    _save(fig, "fig7_example_episode.png")


# --------------------------------------------------------------------------
def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for v in r:
            if isinstance(v, (float, np.floating)):
                cells.append("—" if pd.isna(v) else f"{v:.2f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _num(x: float, sign: bool = False) -> str:
    if pd.isna(x):
        return "—"
    return (f"{x:+.2f}" if sign else f"{x:.2f}").replace("-", "−")


def _diff(mean: float, *ci95: float) -> str:
    """Разность жирным, если ни один из переданных 95% ДИ не содержит 0."""
    s = _num(mean, sign=True)
    return f"**{s}**" if all(abs(mean) > c for c in ci95) else s


def write_tables(summary, paired, agree) -> None:
    out = []
    base = summary[summary["scenario"] == "base"].copy()
    order = ["always_skip", "always_submit", "random", "heuristic", "q_base", "q_dr", "oracle"]
    base = base.set_index("policy").loc[order].reset_index()
    q = base["policy"].str.startswith("q_")
    t = pd.DataFrame({
        "Стратегия": base["policy"].map(NAMES),
        "Награда, у.е.": base["reward_mean"].map(_num),
        "95% ДИ по эпизодам": base["reward_ci95_episodes"].map(lambda c: f"± {c:.2f}"),
        "95% ДИ по seed": base["reward_ci95_seeds"].map(lambda c: "—" if pd.isna(c) else f"± {c:.2f}"),
        "Std по эпизодам": base["reward_std"],
        "Std по seed": base["reward_seed_std"].where(q).map(_num),
        "Маржа побед": base["revenue_mean"],
        "Штрафы": base["penalties_mean"],
        "Затраты": base["cost_mean"],
        "Заявок": base["submitted_mean"],
        "Побед": base["won_mean"],
        "Доля побед": base["success_rate"],
        "Бюджет исп.": base["budget_used_mean"],
        "Нарушения": base["violation_rate"],
    })
    out += ["## Исходные условия (500 тестовых эпизодов)", "", _md(t), ""]

    names = [s.name for s in TEST_SCENARIOS]
    piv = summary[summary["scenario"].isin(names)]
    rows, diffs, others = [], [], []
    for s in TEST_SCENARIOS:
        d = piv[piv["scenario"] == s.name].set_index("policy")
        pb = paired[(paired["scenario"] == s.name)].set_index("policy")
        label = f"{COVERAGE[s.coverage]} {s.title}"
        rows.append({
            "Сценарий": label,
            "Случайная": _num(d.loc["random", "reward_mean"]),
            "Эвристика": _num(d.loc["heuristic", "reward_mean"]),
            "Q base (± std по seed)": f"{_num(d.loc['q_base', 'reward_mean'])} ± {d.loc['q_base', 'reward_seed_std']:.2f}",
            "Q DR (± std по seed)": f"{_num(d.loc['q_dr', 'reward_mean'])} ± {d.loc['q_dr', 'reward_seed_std']:.2f}",
            "Оракул": _num(d.loc["oracle", "reward_mean"]),
        })
        row = {"Сценарий": label}
        for fam, short in (("q_base", "Q base"), ("q_dr", "Q DR")):
            r = pb.loc[fam]
            row |= {
                f"{short} − эвр.": _diff(r["diff_vs_heuristic"], r["diff_ci95_seeds"],
                                         r["diff_ci95_episodes"]),
                f"{short}: ДИ по seed": f"± {r['diff_ci95_seeds']:.2f}",
                f"{short}: ДИ по эпизодам": f"± {r['diff_ci95_episodes']:.2f}",
                f"{short}: seed > эвр.": f"{r['seeds_better_than_heuristic']}/{r['n_seeds']}",
            }
        diffs.append(row)
        r = pb.loc["q_dr"]
        others.append({
            "Сценарий": label,
            "Q DR − Q base": _diff(r["dr_minus_base"], r["dr_minus_base_ci95_seeds"],
                                   r["dr_minus_base_ci95_episodes"]),
            "DR − base: ДИ по seed (Уэлч)": f"± {r['dr_minus_base_ci95_seeds']:.2f}",
            "DR − base: ДИ по эпизодам": f"± {r['dr_minus_base_ci95_episodes']:.2f}",
            "Эвристика − оракул": _diff(r["heuristic_minus_oracle"],
                                        r["heuristic_minus_oracle_ci95_episodes"]),
            "Эвр. − оракул: ДИ по эпизодам": f"± {r['heuristic_minus_oracle_ci95_episodes']:.2f}",
        })
    out += ["## Тестовые сценарии: средняя награда, у.е. (Q: среднее ± std по 5 seed)", "",
            _md(pd.DataFrame(rows)), "",
            "## Тестовые сценарии: парная разность с эвристикой, у.е. (95% ДИ раздельно; "
            "жирным -- оба ДИ не содержат 0)", "", _md(pd.DataFrame(diffs)), "",
            "## Тестовые сценарии: разность DR − base и эвристика − оракул, у.е. "
            "(жирным -- все указанные ДИ не содержат 0)", "", _md(pd.DataFrame(others)), ""]

    rows = []
    for s in TEST_SCENARIOS:
        d = piv[piv["scenario"] == s.name].set_index("policy")
        pb = paired[(paired["scenario"] == s.name)].set_index("policy")
        rows.append({
            "Сценарий": f"{COVERAGE[s.coverage]} {s.title}",
            "Эвристика, % оракула": 100 * pb.loc["q_base", "heuristic_share_of_oracle"],
            "Q base, % оракула": 100 * pb.loc["q_base", "share_of_oracle"],
            "Q DR, % оракула": 100 * pb.loc["q_dr", "share_of_oracle"],
            "Заявок эвр./Qb/Qdr/ор.": "/".join(f"{d.loc[p, 'submitted_mean']:.1f}" for p in ("heuristic", "q_base", "q_dr", "oracle")),
            "Бюджет исп. эвр./Qb/Qdr/ор.": "/".join(f"{d.loc[p, 'budget_used_mean']:.2f}" for p in ("heuristic", "q_base", "q_dr", "oracle")),
            "Непосещ. сост. Qb, %": 100 * d.loc["q_base", "unseen_state_share"],
            "Непосещ. сост. Qdr, %": 100 * d.loc["q_dr", "unseen_state_share"],
        })
    out += ["## Тестовые сценарии: доля от оракула и поведение", "", _md(pd.DataFrame(rows)), ""]

    base_rows = {fam: summary[(summary["scenario"] == "base") & (summary["policy"] == fam)]
                 ["reward_mean"].item() for fam in MAIN_POLICIES}
    rows = []
    for s in TEST_SCENARIOS[1:]:
        d = piv[piv["scenario"] == s.name].set_index("policy")
        rows.append({"Сценарий": f"{COVERAGE[s.coverage]} {s.title}",
                     **{NAMES[f]: f"{100 * (d.loc[f, 'reward_mean'] / base_rows[f] - 1):+.1f}%"
                        for f in ("heuristic", "q_base", "q_dr", "oracle")}})
    out += ["## Изменение средней награды относительно исходных условий", "",
            _md(pd.DataFrame(rows)), ""]

    for factor in ("initial_budget", "cost_scale", "p_scale"):
        d = summary[summary["group"] == f"sweep_{factor}"]
        p = d.pivot_table(index=factor, columns="policy", values="reward_mean")[MAIN_POLICIES]
        p.columns = [NAMES[c] for c in p.columns]
        out += [f"## Развёртка по {factor} (средняя награда, у.е.)", "", _md(p.reset_index()), ""]

    a = (agree.groupby(["scenario", "policy"]).mean(numeric_only=True)
         .drop(columns="seed").reset_index())
    a["policy"] = a["policy"].map(NAMES)
    out += ["## Поведение (200 тестовых эпизодов; Q -- среднее по 5 seed)", "", _md(a), ""]
    (RESULTS_DIR / "summary_tables.md").write_text("\n".join(out), encoding="utf-8")
    print("  saved results/summary_tables.md")


def main() -> None:
    summary = pd.read_csv(RESULTS_DIR / "eval_summary.csv")
    per_seed = pd.read_csv(RESULTS_DIR / "eval_per_seed.csv")
    paired = pd.read_csv(RESULTS_DIR / "paired_vs_heuristic.csv")
    baselines = json.loads((RESULTS_DIR / "validation_baselines.json").read_text(encoding="utf-8"))
    hist = _load_histories()
    fig_learning_curves(hist, baselines)
    fig_base_comparison(summary, per_seed)
    fig_robustness(summary, paired)
    fig_sweeps(summary)
    fig_seed_spread(hist, per_seed)
    behavior_files = [RESULTS_DIR / f for f in ("behavior_submit_rate.csv",
                                                "behavior_budget_path.csv",
                                                "behavior_agreement.csv", "example_episode.csv")]
    if all(f.exists() for f in behavior_files):
        rate, path, agree, example = (pd.read_csv(f) for f in behavior_files)
        fig_behavior(rate, path)
        fig_example_episode(example)
        write_tables(summary, paired, agree)
    else:
        print("  behavior files missing: run `python -m src.behavior` first")


if __name__ == "__main__":
    main()
