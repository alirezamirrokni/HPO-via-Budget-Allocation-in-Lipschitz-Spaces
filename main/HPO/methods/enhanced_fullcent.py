from typing import List, Tuple

import numpy as np
from torch.quasirandom import SobolEngine

from ..benchmark import get_acc, get_suite_dim


def k_center_with_history_values(
    vecs: np.ndarray,
    k: int,
    rng: np.random.Generator,
    epsilon: float,
    prev_centers: np.ndarray = None,
    prev_vals: np.ndarray = None,
) -> List[int]:
    n_points = len(vecs)
    if n_points == 0 or k <= 0:
        return []

    available = np.ones(n_points, dtype=bool)
    nearest = np.full(n_points, np.inf)
    if prev_centers is not None and prev_centers.size:
        distances = np.linalg.norm(vecs[:, None, :] - prev_centers[None, :, :], axis=2)
        available[np.any(distances < 1e-7, axis=1)] = False
        values = np.asarray(prev_vals, dtype=float)
        if values.size != prev_centers.shape[0]:
            raise ValueError("Each previously selected center needs a value")
        modified = distances.copy()
        positive = values > 0
        if positive.any() and epsilon > 0:
            scales = max(values[positive].max(), 1e-8) / values[positive]
            warped = (
                distances[:, positive] * scales[None, :]
                - (scales[None, :] - 1.0) / epsilon
            )
            modified[:, positive] = warped
        nearest = modified.min(axis=1)

    centers = []
    while len(centers) < k and available.any():
        remaining = np.flatnonzero(available)
        idx = (
            int(rng.choice(remaining))
            if not centers
            else int(remaining[np.argmax(nearest[remaining])])
        )
        centers.append(idx)
        available[idx] = False
        nearest = np.minimum(nearest, np.linalg.norm(vecs - vecs[idx], axis=1))
    return centers


class EnhancedFullCent:
    def __init__(
        self,
        n: int,
        k: int,
        T: int,
        B: int,
        epsilon: float,
        seed: int = 0,
    ):
        self.n = n
        self.k = k
        self.T = T
        self.B = B
        self.epsilon = epsilon
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.last_num_used_configs = 0

    def run(self, bench, task_id: str) -> List[Tuple[int, float]]:
        dim = get_suite_dim(bench)
        vecs = (
            SobolEngine(
                dim,
                scramble=True,
                seed=self.seed,
            )
            .draw(self.n)
            .numpy()
        )

        prev_centers = np.empty((0, dim), dtype=float)
        prev_vals = np.empty((0,), dtype=float)

        total_spent = 0
        best_overall = -float("inf")
        trace: List[Tuple[int, float]] = []
        used_configs = 0

        while total_spent < self.B:
            idxs = k_center_with_history_values(
                vecs,
                self.k,
                self.rng,
                self.epsilon,
                prev_centers,
                prev_vals,
            )

            if not idxs:
                break

            new_centers = vecs[idxs]

            cands = [
                {
                    "vec": vec,
                    "spent": 0,
                    "value": None,
                }
                for vec in new_centers
            ]

            while any(c["spent"] < self.T for c in cands) and total_spent < self.B:
                for c in cands:
                    if c["spent"] >= self.T or total_spent >= self.B:
                        continue

                    if c["spent"] == 0:
                        used_configs += 1

                    c["spent"] += 1
                    total_spent += 1

                    acc = get_acc(c["vec"], task_id, c["spent"], bench, self.T)
                    c["value"] = acc

                    best_overall = max(best_overall, acc)
                    trace.append((total_spent, best_overall))

            vals = np.array(
                [c["value"] if c["value"] is not None else 0.0 for c in cands]
            )

            prev_centers = np.vstack([prev_centers, new_centers])
            prev_vals = np.concatenate([prev_vals, vals])

        self.last_num_used_configs = used_configs
        return trace
