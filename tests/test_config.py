from pathlib import Path

from macro_quant.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_config_loads_model_version() -> None:
    config = load_config(ROOT / "configs")
    assert config.model_version == "v0.4-mvp"
    assert "R1_soft_landing_reinflation" in config.model_weights["regime_weights"]
    assert "Gold" in config.model_weights["asset_weights"]
