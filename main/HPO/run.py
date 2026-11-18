from __future__ import annotations

import argparse
import json
from pathlib import Path

from .experiment import run_selected_algorithms
from .benchmark import get_all_task_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YAHPO budget allocation experiments"
    )
    parser.add_argument("--config", type=Path, required=True)
    tasks = parser.add_mutually_exclusive_group(required=True)
    tasks.add_argument(
        "--task", action="append", help="YAHPO task ID. Repeat for multiple tasks."
    )
    tasks.add_argument(
        "--all-tasks", action="store_true", help="Run every task in the selected suite."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Random seeds (default: 0–29).",
    )
    parser.add_argument(
        "--methods", nargs="+", help="Method names from the configuration."
    )
    parser.add_argument(
        "--output", type=Path, help="Directory for generated traces and plots."
    )
    parser.add_argument(
        "--epsilon-samples", type=int, help="Override the Sobol sample count."
    )
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    parameters = cfg["algorithms"]
    methods = args.methods or cfg.get("methods", list(parameters))
    missing = set(methods) - parameters.keys()
    if missing:
        parser.error("Unknown methods: " + ", ".join(sorted(missing)))
    algorithms = {name: parameters[name] for name in methods}
    sample_count = (
        int(cfg.get("epsilon_samples", 10_000))
        if args.epsilon_samples is None
        else args.epsilon_samples
    )
    if sample_count < 2:
        parser.error("--epsilon-samples must be at least 2")

    suite = cfg["suite"]
    run_selected_algorithms(
        suite_name=suite,
        algorithms=algorithms,
        task_ids=get_all_task_ids(suite) if args.all_tasks else args.task,
        seeds=args.seeds if args.seeds is not None else list(range(30)),
        max_epoch=int(cfg["max_epoch"]),
        horizon=int(cfg["budget"]),
        outdir=args.output or Path("outputs") / suite,
        epsilon_samples=sample_count,
    )


if __name__ == "__main__":
    main()
