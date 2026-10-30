from dataclasses import replace

import numpy as np
import pytest

from src.agent import QLearningAgent, StateDiscretizer, masked_argmax
from src.config import AGENT_SEEDS, POLICIES_DIR, QLearningConfig
from src.environment import TenderPortfolioEnv
from src.train import train_agent


def test_discretizer_range_and_size():
    disc = StateDiscretizer()
    assert disc.n_states == 3 * 4 * 3 * 3 * 5 * 3 * 2
    env = TenderPortfolioEnv()
    seen = set()
    for seed in range(50):
        obs, _ = env.reset(seed=seed)
        terminated = False
        while not terminated:
            s = disc(obs)
            assert 0 <= s < disc.n_states
            seen.add(s)
            obs, _, terminated, _, _ = env.step(1)
    assert len(seen) > 100


def test_discretizer_is_injective_on_components():
    disc = StateDiscretizer()
    a = np.array([20.0, 0.1, 5.0, 0.1, 40.0, 30.0, 1.0], dtype=np.float32)
    b = a.copy()
    b[6] = 0.0
    c = a.copy()
    c[0] = 90.0
    assert len({disc(a), disc(b), disc(c)}) == 3


def test_masked_argmax_respects_mask_and_ties():
    assert masked_argmax(np.array([0.0, 5.0]), [1, 0]) == 0
    assert masked_argmax(np.array([0.0, 5.0]), [1, 1]) == 1
    assert masked_argmax(np.array([0.0, 0.0]), [1, 1]) == 0   # не посещённое состояние -> отказ


def test_q_update_formula():
    cfg = replace(QLearningConfig(), alpha_mode="constant", alpha_const=0.5, gamma=0.9)
    agent = QLearningAgent(4, 2, cfg)
    agent.q[1] = [2.0, 10.0]
    td = agent.update(0, 1, r=3.0, s_next=1, terminated=False, next_mask=[1, 1])
    assert td == pytest.approx(3.0 + 0.9 * 10.0)
    assert agent.q[0, 1] == pytest.approx(0.5 * 12.0)
    # недопустимое в s' действие не участвует в max
    agent.update(2, 0, r=1.0, s_next=1, terminated=False, next_mask=[1, 0])
    assert agent.q[2, 0] == pytest.approx(0.5 * (1.0 + 0.9 * 2.0))
    # после терминального шага нет бутстрэпа
    agent.update(3, 1, r=-4.0, s_next=1, terminated=True, next_mask=[1, 1])
    assert agent.q[3, 1] == pytest.approx(-2.0)
    assert agent.visits[0, 1] == 1 and agent.visits.sum() == 3


def test_visit_learning_rate_decays():
    agent = QLearningAgent(1, 2, QLearningConfig())
    rates = []
    for _ in range(5):
        rates.append(agent.learning_rate(0, 0))
        agent.update(0, 0, 1.0, 0, True, [1, 1])
    assert rates[0] == 1.0 and all(x > y for x, y in zip(rates, rates[1:]))
    agent.visits[0, 0] = 10**9
    assert agent.learning_rate(0, 0) == QLearningConfig().alpha_min


def test_epsilon_schedule():
    cfg = replace(QLearningConfig(), n_episodes=1000)
    agent = QLearningAgent(1, 2, cfg)
    assert agent.epsilon(0) == cfg.eps_start
    assert agent.epsilon(400) == pytest.approx((cfg.eps_start + cfg.eps_end) / 2)
    assert agent.epsilon(999) == pytest.approx(cfg.eps_end)


def test_short_training_updates_q_and_roundtrip(tmp_path):
    cfg = replace(QLearningConfig(), n_episodes=300, eval_every=150, n_val_episodes=20)
    agent, hist = train_agent("base", 0, cfg, verbose=False)
    assert agent.visits.sum() == 300 * TenderPortfolioEnv().cfg.horizon
    assert np.count_nonzero(agent.q) > 50
    assert len(hist["validation"]) == 2
    path = tmp_path / "q.npz"
    agent.save(path, {"condition": "base"})
    loaded = QLearningAgent.load(path)
    np.testing.assert_array_equal(loaded.q, agent.q)
    np.testing.assert_array_equal(loaded.visits, agent.visits)


def test_train_cli_with_out_does_not_touch_main_results(tmp_path, monkeypatch):
    import sys

    from src import train as train_module
    main_results = tmp_path / "main_results"
    out_dir = tmp_path / "smoke"
    monkeypatch.setattr(train_module, "RESULTS_DIR", main_results)
    monkeypatch.setattr(sys, "argv", ["train", "--episodes", "50", "--seeds", "0",
                                      "--conditions", "base", "--out", str(out_dir)])
    train_module.main()
    assert (out_dir / "q_base_seed0.npz").exists()
    assert (out_dir / "validation_baselines.json").exists()
    assert not main_results.exists()


def test_published_policies_load_and_have_updates():
    disc = StateDiscretizer()
    horizon = TenderPortfolioEnv().cfg.horizon
    expected_updates = QLearningConfig().n_episodes * horizon
    for cond in ("base", "dr"):
        for seed in AGENT_SEEDS:
            agent = QLearningAgent.load(POLICIES_DIR / f"q_{cond}_seed{seed}.npz")
            assert agent.q.shape == (disc.n_states, 2)
            assert agent.visits.sum() == expected_updates
            assert np.count_nonzero(agent.q) > 100


def test_training_is_reproducible():
    cfg = replace(QLearningConfig(), n_episodes=200, eval_every=200, n_val_episodes=10)
    a1, _ = train_agent("dr", 1, cfg, verbose=False)
    a2, _ = train_agent("dr", 1, cfg, verbose=False)
    np.testing.assert_array_equal(a1.q, a2.q)
