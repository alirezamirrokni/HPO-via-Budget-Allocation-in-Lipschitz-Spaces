from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.run import run_selected_algorithms
from main.utils import get_all_task_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YAHPO budget allocation experiments")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epsilon-samples", type=int)
    args = parser.parse_args()

    if bool(args.task) == args.all_tasks:
        parser.error("Specify one or more --task values or --all-tasks, but not both.")

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    algorithms = dict(cfg["algorithms"])
    if args.methods is not None:
        missing = set(args.methods) - algorithms.keys()
        if missing:
            parser.error("Unknown methods: " + ", ".join(sorted(missing)))
        algorithms = {k: v for k, v in algorithms.items() if k in args.methods}
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
        seeds=args.seeds,
        max_epoch=int(cfg["max_epoch"]),
        horizon=int(cfg["budget"]),
        hb_budget=int(cfg.get("hb_budget", cfg["budget"])),
        outdir=args.output or Path("outputs") / suite,
        epsilon_samples=sample_count,
    )


if __name__ == "__main__":
    main()
