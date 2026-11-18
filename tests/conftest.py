import sys
import types


try:
    import yahpo_gym
except ModuleNotFoundError:
    yahpo_gym = types.ModuleType("yahpo_gym")

    class BenchmarkSet:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Install and configure YAHPO Gym to run benchmarks.")

    yahpo_gym.BenchmarkSet = BenchmarkSet
    yahpo_gym.benchmark_set = types.SimpleNamespace(BenchmarkSet=BenchmarkSet)
    sys.modules["yahpo_gym"] = yahpo_gym
