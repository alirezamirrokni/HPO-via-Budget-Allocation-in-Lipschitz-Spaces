# Multi-fidelity Bayesian optimization

This experiment suite uses Hydra to compare AdaCent and Enhanced-AdaCent with random search, multi-fidelity Knowledge Gradient (KG), Max-value Entropy Search (MES), and rMFBO variants. The configurations in [`conf/problem/`](conf/problem/) cover LCBench, `iaml_rpart`, and `iaml_xgboost`.

## Installation

Follow the [YAHPO data setup](../README.md#installation) from the repository root and keep `YAHPO_LOCAL_CONFIG` set to the absolute path of the configuration. Install the additional requirements from the repository root:

```bash
python -m pip install -r MFBO/requirements.txt
cd MFBO
```

## Running comparisons

```bash
python main.py problem=lcbench algorithm=adacent algorithm.find_best=false
python main.py problem=iaml_rpart algorithm=enhanced_adacent algorithm.find_best=false
python main.py problem=iaml_xgboost algorithm=rmfbo_pseudo
```

The defaults are in [`conf/experiment.yaml`](conf/experiment.yaml). Set `algorithm.find_best=true` to search the AdaCent parameter grid, `checkpoint.enabled=false` to disable checkpoints, or `checkpoint.reset=true` to start the selected run from scratch.

The runner writes numeric records to `res/records/`, checkpoints to `res/checkpoints/` and Hydra logs to `outputs/`. To summarize the records for methods you have run:

```bash
python plot_accuracy.py --problem LCBench --methods AdaCent EnhancedAdaCent
```

The plotter reports the best measured objective at the configured target fidelity, averaged across repetitions with standard-error bars. LCBench uses validation accuracy and the IAML scenarios use AUC.
