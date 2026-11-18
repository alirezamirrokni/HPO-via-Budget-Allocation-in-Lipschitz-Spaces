from __future__ import annotations

from typing import List, Tuple

import numpy as np
from torch.quasirandom import SobolEngine

from ..benchmark import get_acc, get_suite_dim


class FullCent:
    def __init__(self, n: int, k: int, T: int, B: int, seed: int = 0):
        self.n = n
        self.k = k
        self.T = T
        self.B = B
        self.seed = seed
        self.last_num_used_configs = 0

    def run(self, bench, task_id: str) -> List[Tuple[int, float]]:
        vecs = (
            SobolEngine(dimension=get_suite_dim(bench), scramble=True, seed=self.seed)
            .draw(self.n)
            .numpy()
        )
        available = np.ones(self.n, dtype=bool)
        nearest = np.full(self.n, np.inf, dtype=float)
        rng = np.random.default_rng(self.seed)
        budget = 0
        best = -float("inf")
        trace: List[Tuple[int, float]] = []
        evaluated = 0

        while budget < self.B and evaluated < self.k and available.any():
            remaining = np.flatnonzero(available)
            index = (
                int(rng.choice(remaining))
                if evaluated == 0
                else int(remaining[np.argmax(nearest[remaining])])
            )
            available[index] = False
            evaluated += 1
            nearest = np.minimum(nearest, np.linalg.norm(vecs - vecs[index], axis=1))

            for step in range(1, self.T + 1):
                if budget >= self.B:
                    break
                value = get_acc(vecs[index], task_id, step, bench, self.T)
                budget += 1
                best = max(best, value)
                trace.append((budget, best))

        self.last_num_used_configs = evaluated
        return trace
