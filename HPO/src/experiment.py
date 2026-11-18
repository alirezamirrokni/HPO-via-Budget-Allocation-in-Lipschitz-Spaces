from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from yahpo_gym import benchmark_set

from . import other_algorithms
from .AdaCent import AdaCent
from .EnhancedAdaCent import EnhancedAdaCent
from .EnhancedFullCent import EnhancedFullCent
from .FullCent import FullCent
from .utils import estimate_epsilon, track_objective_function

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


CENTER_METHODS = {
    "AdaCent": AdaCent,
    "EnhancedAdaCent": EnhancedAdaCent,
    "FullCent": FullCent,
    "EnhancedFullCent": EnhancedFullCent,
}

ENHANCED_METHODS = {"EnhancedAdaCent", "EnhancedFullCent"}

BASELINES = {
    "Random Search": (other_algorithms.random_search, "trials", "n_trials"),
    "Hyperband": (other_algorithms.hyperband, "iters", "n_iterations"),
    "BOHB": (other_algorithms.bohb_search, "iters", "n_iterations"),
    "SMAC": (other_algorithms.smac_search, "trials", "n_trials"),
}


def trace_to_array(trace: List[Tuple[int, float]], horizon: int) -> np.ndarray:
    arr = np.empty(horizon, dtype=float)
    last, idx = 0.0, 0
    for b in range(1, horizon + 1):
        while idx < len(trace) and trace[idx][0] <= b:
            last = trace[idx][1]
            idx += 1
        arr[b - 1] = last
    return arr


def aggregate_results(
    runs: List[Dict[str, np.ndarray]],
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
]:
    algos = runs[0].keys()
    means, stds = {}, {}

    for algo in algos:
        traces = np.stack([d[algo] for d in runs], axis=0)

        means[algo] = np.nanmean(traces, axis=0)
        stds[algo] = np.nanstd(traces, axis=0)

    return means, stds


def plot_task(task_id, means, stds, suite_name, epsilon, plots_dir, num_seeds):
    horizon = len(next(iter(means.values())))
    budget = np.arange(1, horizon + 1)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, mean in means.items():
        (line,) = ax.plot(budget, mean, label=label)
        spread = stds[label]
        ax.fill_between(
            budget, mean - spread, mean + spread, color=line.get_color(), alpha=0.15
        )

    ax.set(
        xlim=(1, horizon),
        xlabel="Epoch budget spent"
        if suite_name == "lcbench"
        else "Fidelity steps spent",
        ylabel="Best validation accuracy (%)"
        if suite_name == "lcbench"
        else "Best validation accuracy",
        title=f"{suite_name} / {task_id} ({num_seeds} seeds)",
    )
    ax.legend(title=f"Estimated ε = {epsilon:.2f}" if epsilon is not None else None)
    ax.grid(alpha=0.25)
    out_path = plots_dir / f"{task_id}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out_path}")


def make_label(name: str, params: Dict) -> str:
    if "k" in params:
        return f"{name} (k={params['k']})"
    return name


def build_algorithm(name, params, seed, max_epoch, horizon, epsilon):
    options = {"n": 1000, "T": max_epoch, "B": horizon, "seed": seed, **params}
    if name in ENHANCED_METHODS:
        options["epsilon"] = epsilon
    return CENTER_METHODS[name](**options)


def run_external_algorithm(name, params, bench, task, seed):
    run, config_option, argument = BASELINES[name]
    options = {argument: params[config_option]} if config_option in params else {}
    return run(bench, task, seed=seed, **options)


def run_selected_algorithms(
    suite_name: str,
    algorithms: Dict[str, Dict],
    task_ids: List[str],
    seeds: List[int],
    max_epoch: int,
    horizon: int,
    outdir: Path,
    epsilon_samples: int = 10_000,
):
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir = outdir / "plots"
    avg_root = outdir / "avg_traces"
    plots_dir.mkdir(parents=True, exist_ok=True)
    avg_root.mkdir(parents=True, exist_ok=True)

    labels = [make_label(name, params) for name, params in algorithms.items()]
    estimate_task_epsilon = not ENHANCED_METHODS.isdisjoint(algorithms)

    overall_used_config_counts: Dict[str, List[int]] = {label: [] for label in labels}

    total_runs = len(task_ids) * len(seeds) * len(algorithms)
    completed_runs = 0

    for task_idx, task in enumerate(task_ids, start=1):
        bench = benchmark_set.BenchmarkSet(suite_name, instance=task, multithread=False)

        print(f"→ Running task {task} ({task_idx}/{len(task_ids)})")
        eps = None
        if estimate_task_epsilon:
            eps = estimate_epsilon(
                bench, task, epsilon_samples, max(1, int(0.1 * max_epoch)), max_epoch
            )
            print(f"  estimated epsilon: {eps:.6f}")

        all_results: List[Dict[str, np.ndarray]] = []
        used_config_counts: Dict[str, List[int]] = {label: [] for label in labels}

        for seed_idx, seed in enumerate(seeds, start=1):
            print(f"  Seed {seed} ({seed_idx}/{len(seeds)})")
            result: Dict[str, np.ndarray] = {}

            for algo_idx, (name, params) in enumerate(algorithms.items(), start=1):
                label = make_label(name, params)
                overall_idx = completed_runs + 1
                progress_pct = (
                    100.0 * overall_idx / total_runs if total_runs > 0 else 100.0
                )
                print(
                    f"    [{overall_idx}/{total_runs} | {progress_pct:6.2f}%] "
                    f"Running {label} (algorithm {algo_idx}/{len(algorithms)})"
                )

                with track_objective_function(bench) as tracker:
                    if name in CENTER_METHODS:
                        algorithm = build_algorithm(
                            name, params, seed, max_epoch, horizon, eps
                        )
                        trace = algorithm.run(bench, task)
                    else:
                        trace = run_external_algorithm(name, params, bench, task, seed)
                    used_configs = tracker.num_unique_configs

                used_config_counts[label].append(used_configs)
                overall_used_config_counts[label].append(used_configs)

                aligned = trace_to_array(trace, horizon)
                used_budget = trace[-1][0] if trace else 0
                result[label] = aligned

                completed_runs += 1
                print(
                    f"      done: budget used = {used_budget}, "
                    f"used configs = {used_configs}"
                )

            all_results.append(result)

        means, stds = aggregate_results(all_results)

        for label, mean_arr in means.items():
            std_arr = stds[label]
            algo_dir = avg_root / label.replace(" ", "_")
            algo_dir.mkdir(parents=True, exist_ok=True)
            path = algo_dir / f"{task}.npz"
            np.savez_compressed(path, mean=mean_arr, std=std_arr)
            print(f"✓ Saved average trace to {path}")

        print("Average number of used configs:")
        for label, counts in used_config_counts.items():
            print(f"  {label}: {np.mean(counts):.2f} ± {np.std(counts):.2f}")

        plot_task(task, means, stds, suite_name, eps, plots_dir, len(seeds))

    print("Overall average number of used configs across all tasks:")
    for label, counts in overall_used_config_counts.items():
        print(f"  {label}: {np.mean(counts):.2f} ± {np.std(counts):.2f}")

    zip_path = outdir / "avg_traces.zip"
    shutil.make_archive(
        str(zip_path.with_suffix("")), "zip", root_dir=outdir, base_dir="avg_traces"
    )
    print(f"✓ Created zip: {zip_path}")
