# Configuration file: Used to configure shared components
# File: src/utils/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Ghi cứng chuỗi kết nối để đảm bảo không lấy nhầm biến môi trường cũ
DATABASE_URL = "postgresql+psycopg2://root:secret@localhost:5432/videorag"


def load_section(config_path: str | Path | None, section: str, defaults: dict[str, Any]) -> dict[str, Any]:
    """Load a named config section from YAML while keeping built-in defaults."""
    cfg = dict(defaults)
    if config_path is None:
        return cfg

    config_file = Path(config_path)
    if not config_file.exists():
        return cfg

    with config_file.open("r", encoding="utf-8") as f:
        file_cfg = yaml.safe_load(f) or {}

    if not isinstance(file_cfg, dict):
        return cfg

    section_cfg = file_cfg.get(section, {})
    if isinstance(section_cfg, dict):
        cfg.update(section_cfg)
    return cfg


def resolve_crop_path(crops_root: str | Path, crop_path: str | Path) -> Path:
    """Resolve a crop entry to an absolute path under the crop root.

    ``crop_path`` is stored relative to the crop root (for example
    ``{video_id}/{track_id}/f000123.jpg``), so it must not depend on the current
    working directory when the pipeline is run from a different folder.
    """
    crop_rel = Path(crop_path)
    if crop_rel.is_absolute():
        return crop_rel
    return Path(crops_root) / crop_rel