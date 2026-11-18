# Multi-fidelity Bayesian optimization comparisons

This directory contains the multi-fidelity Bayesian optimization experiments accompanying [the paper](https://arxiv.org/pdf/2510.10605). It is a separate experimental stack from the YAHPO Gym evaluations under `main/`.

The Hydra configurations in `conf/algorithm/` cover AdaCent, Enhanced-AdaCent, rMFBO variants, Knowledge Gradient (KG), Max-value Entropy Search (MES), and random search. The problem configurations in `conf/problem/` cover LCBench, RPart, and XGBoost.

## Setup

Use a dedicated Python environment; Python 3.10 is recommended.

```bash
cd MFBO
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Examples

```bash
python main.py algorithm=adacent problem=lcbench algorithm.find_best=false
python main.py algorithm=enhanced_adacent problem=iaml_rpart algorithm.find_best=false
python main.py algorithm=rmfbo_pseudo problem=iaml_xgboost
```

The default experiment is defined in `conf/experiment.yaml`. Set `algorithm.find_best=true` to search the AdaCent parameter grid. To manage checkpoints, use `checkpoint.enabled=false` to disable them or `checkpoint.reset=true` to restart the experiment. Generated checkpoints, plots, and traces under `res/` are excluded from version control by the repository `.gitignore`.
