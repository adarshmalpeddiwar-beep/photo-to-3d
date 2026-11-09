"""Mesh cleanup, measurement and placement helpers (NumPy only).

Conventions: vertices are (n, 3) floats in millimetres after scaling, faces are (m, 3)
int indices, triangles wind counter-clockwise seen from outside, and Z is up.
"""
from __future__ import annotations

import numpy as np


def weld(vertices: np.ndarray, faces: np.ndarray, decimals: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Merge vertices that sit at the same position (after rounding)."""
    rounded = np.round(np.asarray(vertices, dtype=np.float64), decimals)
    unique, inverse = np.unique(rounded, axis=0, return_inverse=True)
    new_faces = inverse.reshape(-1)[np.asarray(faces, dtype=np.int64)]
    return unique, new_faces


def remove_degenerate(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Drop triangles that repeat a vertex or have (almost) zero area."""
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        return faces.reshape(0, 3)
    distinct = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
    tri = vertices[faces]
    area2 = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    return faces[distinct & (area2 > 1e-12)]


def compact(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove vertices that no face uses and renumber the faces."""
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return vertices[used], remap[faces]


def edge_report(faces: np.ndarray) -> dict:
    """Count boundary edges (used by one face) and non-manifold edges (used by more than two)."""
    if len(faces) == 0:
        return {"boundary_edges": 0, "non_manifold_edges": 0, "edge_count": 0}
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "boundary_edges": int((counts == 1).sum()),
        "non_manifold_edges": int((counts > 2).sum()),
        "edge_count": int(len(counts)),
    }


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Volume of a closed mesh. Positive when the faces point outward."""
    tri = vertices[faces]
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


def orient_outward(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Flip every face if the mesh is inside-out."""
    if signed_volume(vertices, faces) < 0:
        return faces[:, ::-1].copy()
    return faces


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit normals and areas for each triangle."""
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(cross, axis=1)
    safe = np.where(length > 0, length, 1.0)
    return cross / safe[:, None], length / 2.0


def extents(vertices: np.ndarray) -> np.ndarray:
    """Size of the bounding box along x, y and z."""
    return vertices.max(axis=0) - vertices.min(axis=0)


def place_on_plate(vertices: np.ndarray) -> np.ndarray:
    """Centre the model on X and Y, and rest it on Z = 0."""
    v = vertices.copy()
    mins = vertices.min(axis=0)
    maxs = vertices.max(axis=0)
    v[:, 0] -= (mins[0] + maxs[0]) / 2.0
    v[:, 1] -= (mins[1] + maxs[1]) / 2.0
    v[:, 2] -= mins[2]
    return v


def overhang_fraction(vertices: np.ndarray, faces: np.ndarray, max_deg: float) -> tuple[float, int]:
    """Share of surface area that overhangs more than max_deg from vertical.

    A downward-facing face is flagged when its normal is steeper than the limit:
    n_z < -sin(max_deg). A flat ceiling (n_z = -1) is always flagged.
    Returns (area fraction, number of flagged triangles).
    """
    normals, areas = face_normals(vertices, faces)
    total = float(areas.sum())
    if total <= 0:
        return 0.0, 0
    flagged = normals[:, 2] < -np.sin(np.radians(max_deg))
    return float(areas[flagged].sum() / total), int(flagged.sum())
