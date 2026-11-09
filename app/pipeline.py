"""The conversion pipeline: segment -> reconstruct -> scale -> clean -> check -> export."""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .config import Settings
from .exporters import write_3mf, write_stl_binary
from .imaging import SegmentationError, segment_object
from .meshops import compact, extents, orient_outward, place_on_plate, weld, remove_degenerate
from .printcheck import run_print_checks
from .reconstruct import ReconstructionError, get_backend


class PipelineError(Exception):
    """A stage failed in a way the user can act on."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


def run_pipeline(
    job_dir: Path,
    settings: Settings,
    reference_width_mm: float | None,
    progress: Callable[[str], None],
) -> dict:
    photos = sorted((job_dir / "photos").glob("photo-*.png"))
    if not photos:
        raise PipelineError("segment", "No photos were uploaded.")

    progress("segment")
    try:
        with Image.open(photos[0]) as img:
            image = img.convert("RGB")
        mask, _threshold = segment_object(image, settings.analysis_max_side)
    except SegmentationError as exc:
        raise PipelineError("segment", str(exc)) from exc
    Image.fromarray(mask.astype(np.uint8) * 255).save(job_dir / "mask.png")

    progress("reconstruct")
    try:
        backend = get_backend(settings.backend, settings.voxel_max_side)
        raw = backend.reconstruct(mask)
    except ReconstructionError as exc:
        raise PipelineError("reconstruct", str(exc)) from exc
    if len(raw.faces) == 0:
        raise PipelineError("reconstruct", "Reconstruction produced no geometry.")

    progress("scale")
    notes = list(raw.notes)
    size = extents(raw.vertices)
    if reference_width_mm:
        if size[0] <= 0:
            raise PipelineError("scale", "The reconstructed object has no width to calibrate against.")
        factor = reference_width_mm / size[0]
        scale_info = {
            "source": "reference_width",
            "reference_width_mm": reference_width_mm,
            "factor": round(float(factor), 6),
            "accuracy": "Width matches your reference. Depth and height are estimated from one photo.",
        }
    else:
        factor = settings.default_longest_mm / float(size.max())
        scale_info = {
            "source": "normalised",
            "factor": round(float(factor), 6),
            "accuracy": "Not calibrated. The size is a placeholder. Enter a reference width for real dimensions.",
        }
        notes.append("No reference width was given, so the model is unscaled.")
    vertices = raw.vertices * factor

    progress("clean")
    vertices, faces = weld(vertices, raw.faces)
    faces = remove_degenerate(vertices, faces)
    vertices, faces = compact(vertices, faces)
    if len(faces) == 0:
        raise PipelineError("clean", "Mesh cleanup removed every face.")
    faces = orient_outward(vertices, faces)
    vertices = place_on_plate(vertices)

    progress("check")
    report = run_print_checks(
        vertices,
        faces,
        settings.build_volume_mm,
        settings.overhang_max_deg,
        settings.face_budget,
    )

    progress("export")
    write_stl_binary(job_dir / "mesh.stl", vertices, faces)
    write_3mf(job_dir / "model.3mf", vertices, faces)

    return {
        "backend": raw.backend,
        "scale": scale_info,
        "report": report,
        "notes": notes,
    }
