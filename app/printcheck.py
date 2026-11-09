"""Print readiness checks. Each check returns pass, warn, fail or not_checked."""
from __future__ import annotations

import numpy as np

from .meshops import edge_report, extents, overhang_fraction, signed_volume

OVERHANG_WARN_FRACTION = 0.02


def _check(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def run_print_checks(
    vertices: np.ndarray,
    faces: np.ndarray,
    build_volume_mm: tuple[float, float, float],
    overhang_max_deg: float,
    face_budget: int,
) -> dict:
    checks: list[dict] = []

    edges = edge_report(faces)
    open_edges = edges["boundary_edges"]
    checks.append(_check(
        "watertight",
        "pass" if open_edges == 0 else "fail",
        "Closed surface." if open_edges == 0 else f"{open_edges} open edges. The slicer may not print this cleanly.",
    ))

    bad_edges = edges["non_manifold_edges"]
    checks.append(_check(
        "manifold",
        "pass" if bad_edges == 0 else "fail",
        "Each edge is shared by exactly two faces." if bad_edges == 0
        else f"{bad_edges} edges are shared by more than two faces.",
    ))

    dims = extents(vertices)
    limits = np.array(build_volume_mm, dtype=np.float64)
    if np.all(dims <= limits):
        checks.append(_check("build_volume", "pass", "Fits the build volume as placed."))
    elif np.all(np.sort(dims) <= np.sort(limits)):
        checks.append(_check("build_volume", "warn", "Fits only if rotated on the build plate."))
    else:
        checks.append(_check("build_volume", "fail", "Larger than the build volume in every orientation."))

    fraction, _ = overhang_fraction(vertices, faces, overhang_max_deg)
    overhang_status = "warn" if fraction > OVERHANG_WARN_FRACTION else "pass"
    checks.append(_check(
        "overhangs",
        overhang_status,
        f"{fraction * 100:.1f}% of surface area overhangs more than {overhang_max_deg:g}° from vertical. "
        "Supports may be needed.",
    ))

    if len(faces) > face_budget:
        checks.append(_check("face_count", "warn",
                             f"{len(faces)} triangles exceeds the {face_budget} budget. Decimation is not in this build."))
    else:
        checks.append(_check("face_count", "pass", f"{len(faces)} triangles."))

    checks.append(_check("wall_thickness", "not_checked",
                         "Wall thickness analysis is not included in this build."))

    statuses = {c["status"] for c in checks}
    overall = "fail" if "fail" in statuses else "warn" if "warn" in statuses else "pass"
    return {
        "status": overall,
        "checks": checks,
        "dimensions_mm": [round(float(d), 3) for d in dims],
        "volume_mm3": round(abs(signed_volume(vertices, faces)), 2) if len(faces) else 0.0,
        "triangles": int(len(faces)),
        "edges": edges,
    }
