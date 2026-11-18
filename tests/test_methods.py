import numpy as np
import pytest

from main.AdaCent import AdaCent, _linear_tail, k_center_with_history
from main.EnhancedAdaCent import EnhancedAdaCent
from main.EnhancedFullCent import EnhancedFullCent, k_center_with_history_values
from main.FullCent import FullCent


@pytest.mark.parametrize("history, expected", [
    ([(0, 0), (2, 0.5)], 1.25),
    ([(0, 0), (2, 0)], 0.0),
    ([(1, 4.0), (1, 5.0)], 5.0),
])
def test_linear_tail(history, expected):
    assert _linear_tail(history, T=5) == pytest.approx(expected)


def test_kcenter_does_not_repeat_points():
    pts = np.array([[0.0], [0.5], [1.0]])
    idx = k_center_with_history(pts, k=10, rng=np.random.default_rng(0))
    assert sorted(idx) == [0, 1, 2]


def test_kcenter_respects_previous_centers():
    pts = np.array([[0.0], [0.5], [1.0]])
    prev = np.array([[0.0], [0.5]])
    idx = k_center_with_history(pts, k=10, rng=np.random.default_rng(0), prev_centers=prev)
    assert idx == [2]


def test_value_aware_selector_exhausts_pool():
    pts = np.array([[0.0], [0.5], [1.0]])
    idx = k_center_with_history_values(pts, 10, np.random.default_rng(0), epsilon=0.1)
    assert len(idx) == len(set(idx)) == 3
    idx = k_center_with_history_values(
        pts, 10, np.random.default_rng(0), epsilon=0.1,
        prev_centers=pts.copy(), prev_vals=np.ones(3),
    )
    assert idx == []


@pytest.mark.parametrize("class_name", ["AdaCent", "EnhancedAdaCent", "EnhancedFullCent", "FullCent"])
def test_budget_and_termination(monkeypatch, class_name):
    import main.AdaCent as ac
    import main.EnhancedAdaCent as ea
    import main.EnhancedFullCent as ef
    import main.FullCent as fc

    modules = {"AdaCent": ac, "EnhancedAdaCent": ea, "EnhancedFullCent": ef, "FullCent": fc}
    module = modules[class_name]
    monkeypatch.setattr(module, "get_suite_dim", lambda _bench: 2)
    monkeypatch.setattr(module, "get_acc", lambda x, _task, _step, _bench, _T: 0.5)
    cls = getattr(module, class_name)
    kwargs = dict(n=4, k=8, T=2, B=50, seed=1)
    if "Enhanced" in class_name:
        kwargs["epsilon"] = 0.1
    result = cls(**kwargs).run(None, "fake")
    assert len(result) <= 50
    assert result
    assert len(result) <= 4 * 2
    assert [step for step, _ in result] == list(range(1, len(result) + 1))
    assert all(np.isfinite(score) for _, score in result)


def test_enhanced_selector_handles_zero_scores():
    alg = EnhancedAdaCent(n=3, k=2, T=2, B=6, epsilon=0.1)
    idx = alg._choose_next_center(
        np.array([2]), np.array([0, 1]),
        np.array([0, 0, 0]), 0.0,
        np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float),
    )
    assert idx == 2
