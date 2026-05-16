from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, field_validator


class AppSettings(BaseModel):
    project_root: Path
    config_dir: Path
    data_dir: Path
    reports_dir: Path
    db_path: Path

    @field_validator("project_root", "config_dir", "data_dir", "reports_dir", "db_path")
    @classmethod
    def expand_path(cls, value: Path) -> Path:
        return Path(value).expanduser().resolve()


class ConfigBundle(BaseModel):
    tickers: dict[str, Any]
    indicators: dict[str, Any]
    model_weights: dict[str, Any]
    narrative_topics: dict[str, Any]
    portfolio_rules: dict[str, Any]

    @property
    def model_version(self) -> str:
        return str(self.model_weights.get("model_version", "v0.unknown"))


def project_root_from(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path).expanduser().resolve()
    return Path(__file__).resolve().parents[3]


def load_settings(project_root: Path | None = None, db_path: Path | None = None) -> AppSettings:
    root = project_root_from(project_root)
    load_dotenv(root / ".env")
    data_dir = root / "data"
    reports_dir = root / "reports"
    environ = __import__("os").environ
    configured_db = db_path or Path(
        environ.get("MACROLENS_DB_PATH") or environ.get("MACRO_QUANT_DB_PATH") or data_dir / "macrolens.duckdb"
    )
    if not configured_db.is_absolute():
        configured_db = root / configured_db
    return AppSettings(
        project_root=root,
        config_dir=root / "configs",
        data_dir=data_dir,
        reports_dir=reports_dir,
        db_path=configured_db,
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}")
    return data


def load_config(config_dir: Path) -> ConfigBundle:
    config_dir = Path(config_dir)
    return ConfigBundle(
        tickers=load_yaml(config_dir / "tickers.yaml"),
        indicators=load_yaml(config_dir / "indicators.yaml"),
        model_weights=load_yaml(config_dir / "model_weights.yaml"),
        narrative_topics=load_yaml(config_dir / "narrative_topics.yaml"),
        portfolio_rules=load_yaml(config_dir / "portfolio_rules.yaml"),
    )


def ensure_runtime_dirs(settings: AppSettings) -> None:
    for directory in [
        settings.data_dir,
        settings.data_dir / "raw",
        settings.data_dir / "processed",
        settings.reports_dir,
        settings.reports_dir / "daily",
        settings.reports_dir / "weekly",
        settings.reports_dir / "backtests",
        settings.reports_dir / "simulations",
    ]:
        directory.mkdir(parents=True, exist_ok=True)
