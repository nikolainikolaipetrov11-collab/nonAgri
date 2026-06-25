from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_ENV_VAR = "NONAGRI_CONFIG"


class ProjectConfig:
    def __init__(self, config_path: str | os.PathLike[str] | None = None):
        configured_path = config_path or os.environ.get(CONFIG_ENV_VAR)
        self.config_path = Path(configured_path) if configured_path else PROJECT_ROOT / "config.toml"
        if not self.config_path.is_absolute():
            self.config_path = PROJECT_ROOT / self.config_path

        with self.config_path.open("rb") as fh:
            self._data = tomllib.load(fh)

    @property
    def root(self) -> Path:
        return PROJECT_ROOT

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self._data.get(section, {}).get(key, default)

    def path(self, section: str, key: str) -> Path:
        value = self.get(section, key)
        if value is None:
            raise KeyError(f"Missing config value: {section}.{key}")

        path = Path(value)
        return path if path.is_absolute() else self.root / path

    @property
    def date_raw_dir(self) -> Path:
        return self.path("paths", "date_raw_dir")

    @property
    def date_process_dir(self) -> Path:
        return self.path("paths", "date_process_dir")

    @property
    def date_out_dir(self) -> Path:
        return self.path("paths", "date_out_dir")

    @property
    def date_out_non_agri_dir(self) -> Path:
        return self.path("paths", "date_out_non_agri_dir")

    @property
    def parcel_shp(self) -> Path:
        return self.path("paths", "parcel_shp")

    @property
    def agri_parcels_shp(self) -> Path:
        return self.path("paths", "agri_parcels_shp")

    @property
    def raw_landuse_shp(self) -> Path:
        return self.path("paths", "raw_landuse_shp")

    @property
    def checkpoint_dir(self) -> Path:
        return self.path("paths", "checkpoint_dir")

    @property
    def evaluation_dir(self) -> Path:
        return self.path("paths", "evaluation_dir")

    @property
    def safe_grain_dir(self) -> Path:
        return self.path("paths", "safe_grain_dir")

    @property
    def non_grain_dir(self) -> Path:
        return self.path("paths", "non_grain_dir")

    @property
    def non_agri_dir(self) -> Path:
        return self.path("paths", "non_agri_dir")

    @property
    def human_check_dir(self) -> Path:
        return self.path("paths", "human_check_dir")

    def date_out(self, *parts: str) -> Path:
        return self.date_out_dir.joinpath(*parts)

    def date_out_non_agri(self, *parts: str) -> Path:
        return self.date_out_non_agri_dir.joinpath(*parts)

    def split_report(self, filename: str) -> Path:
        return self.date_out("split_reports", filename)


CONFIG = ProjectConfig()
