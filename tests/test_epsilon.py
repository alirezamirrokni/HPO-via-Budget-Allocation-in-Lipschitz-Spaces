import numpy as np
import pytest
from torch.quasirandom import SobolEngine

import main.utils as utils


def fake_acc(vec, **kwargs):
    return 0.2 + float(vec[0]) * 0.6


def test_epsilon_matches_exhaustive_pairs(monkeypatch):
    monkeypatch.setattr(utils, "get_suite_dim", lambda _bench: 2)
    monkeypatch.setattr(utils, "get_acc", fake_acc)
    estimate = utils.estimate_epsilon(None, "task", 15, 1, seed=7)
    pts = SobolEngine(2, scramble=True, seed=7).draw(15).numpy()
    vals = np.array([fake_acc(pt) for pt in pts], dtype=np.float32)
    i, j = np.triu_indices(15, 1)
    ratio = (1 - np.minimum(vals[i], vals[j]) / np.maximum(vals[i], vals[j])) / np.linalg.norm(pts[i] - pts[j], axis=1)
    expected = float(np.percentile(ratio[ratio > 0], 95))
    assert estimate == pytest.approx(expected, rel=1e-5)


def test_epsilon_rejects_constant_values(monkeypatch):
    monkeypatch.setattr(utils, "get_suite_dim", lambda _bench: 2)
    monkeypatch.setattr(utils, "get_acc", lambda *args, **kwargs: 1.0)
    with pytest.raises(ValueError, match="valid pairs"):
        utils.estimate_epsilon(None, "task", 4, 1, seed=0)
