import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("suite", ["lcbench", "rbv2_rpart", "rbv2_aknn"])
def test_config_valid(suite):
    cfg = json.loads((ROOT / "configs" / f"{suite}.json").read_text())
    assert cfg["suite"] == suite
    assert cfg["max_epoch"] == 52
    assert cfg["budget"] >= cfg["max_epoch"]
    assert cfg["epsilon_samples"] >= 2
    assert {"AdaCent", "EnhancedAdaCent", "EnhancedFullCent"} <= cfg["algorithms"].keys()
