"""Runtime settings, read from environment variables with safe defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    data_dir: str
    backend: str
    app_token: str | None
    max_photos: int = 5
    max_upload_bytes: int = 20 * 1024 * 1024
    max_pixels: int = 40_000_000
    analysis_max_side: int = 256
    voxel_max_side: int = 48
    default_longest_mm: float = 100.0
    build_volume_mm: tuple[float, float, float] = (220.0, 220.0, 250.0)
    overhang_max_deg: float = 45.0
    face_budget: int = 300_000
    jobs_per_window: int = 10
    rate_window_s: int = 600


def load_settings() -> Settings:
    """Build settings from the environment.

    P3D_DATA_DIR  where jobs are stored (default: ./data)
    P3D_BACKEND   reconstruction backend (default: stub)
    P3D_TOKEN     if set, every /api request must send "Authorization: Bearer <token>"
    """
    return Settings(
        data_dir=os.environ.get("P3D_DATA_DIR", os.path.join(os.getcwd(), "data")),
        backend=os.environ.get("P3D_BACKEND", "stub"),
        app_token=os.environ.get("P3D_TOKEN") or None,
    )
