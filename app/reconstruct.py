"""Pluggable 3D reconstruction backends.

Only the silhouette stub is implemented. It extrudes the photo outline into voxels
so the whole pipeline (scaling, cleanup, checks, export) can run end to end.
Its shape is NOT a real model of the object. A generative or photogrammetry backend
plugs in as another ReconstructionBackend subclass.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from PIL import Image


class ReconstructionError(RuntimeError):
    """A backend could not produce geometry."""


@dataclass
class RawMesh:
    vertices: np.ndarray  # (n, 3) float, in backend units
    faces: np.ndarray     # (m, 3) int
    unit: str             # e.g. "voxel"; scaled to millimetres later
    backend: str
    notes: list[str] = field(default_factory=list)


class ReconstructionBackend(ABC):
    name: str

    @abstractmethod
    def reconstruct(self, mask: np.ndarray) -> RawMesh:
        """Build a mesh from a boolean silhouette mask (row 0 is the top of the photo)."""


class SilhouetteStubBackend(ReconstructionBackend):
    """Placeholder: extrude the silhouette into voxels, then mesh the exposed faces."""

    name = "silhouette-stub"

    def __init__(self, voxel_max_side: int = 48):
        self.voxel_max_side = voxel_max_side

    def reconstruct(self, mask: np.ndarray) -> RawMesh:
        grid = _resample_mask(mask, self.voxel_max_side)
        grid = grid[::-1]  # image rows run top to bottom; the model's y axis runs up
        rows, cols = np.any(grid, axis=1), np.any(grid, axis=0)
        if not grid.any():
            raise ReconstructionError("The silhouette is empty.")
        grid = grid[rows][:, cols]  # crop to the object so the guessed depth follows the object
        width, height = grid.shape[1], grid.shape[0]
        depth = max(1, min(width, height))  # guessed: a single photo cannot show depth
        voxels = np.repeat(grid.T[:, :, None], depth, axis=2)
        vertices, faces = voxels_to_mesh(voxels)
        notes = [
            "Placeholder reconstruction: the photo outline is extruded. "
            "Depth is guessed and the shape is not a model of the object.",
        ]
        return RawMesh(vertices, faces, unit="voxel", backend=self.name, notes=notes)


def _resample_mask(mask: np.ndarray, max_side: int) -> np.ndarray:
    height, width = mask.shape
    scale = max_side / max(height, width)
    new_h = max(1, round(height * scale))
    new_w = max(1, round(width * scale))
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    return np.asarray(img) >= 128


def _shift(voxels: np.ndarray, axis: int, sign: int) -> np.ndarray:
    """neighbour[i] = voxels[i + sign] along axis, with zeros outside the grid."""
    out = np.zeros_like(voxels)
    src = [slice(None)] * 3
    dst = [slice(None)] * 3
    if sign > 0:
        dst[axis], src[axis] = slice(0, -1), slice(1, None)
    else:
        dst[axis], src[axis] = slice(1, None), slice(0, -1)
    out[tuple(dst)] = voxels[tuple(src)]
    return out


def voxels_to_mesh(voxels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mesh the exposed faces of a boolean voxel grid. Triangles face outward.

    Vertices sit on integer grid corners. A voxel at index (i, j, k) covers [i, i+1] etc.
    """
    voxels = voxels.astype(bool)
    chunks: list[np.ndarray] = []
    for axis in range(3):
        a1, a2 = (axis + 1) % 3, (axis + 2) % 3  # (a1, a2, axis) is right-handed
        for sign in (1, -1):
            exposed = voxels & ~_shift(voxels, axis, sign)
            idx = np.argwhere(exposed)
            if idx.size == 0:
                continue
            n = len(idx)
            plane = idx[:, axis] + (1 if sign > 0 else 0)
            quad = np.zeros((n, 4, 3), dtype=np.int64)
            for k, (du, dv) in enumerate(((0, 0), (1, 0), (1, 1), (0, 1))):
                quad[:, k, axis] = plane
                quad[:, k, a1] = idx[:, a1] + du
                quad[:, k, a2] = idx[:, a2] + dv
            if sign < 0:
                quad = quad[:, [0, 3, 2, 1], :]  # flip winding so normals point out
            chunks.append(quad[:, [0, 1, 2], :])
            chunks.append(quad[:, [0, 2, 3], :])
    if not chunks:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    tris = np.concatenate(chunks, axis=0)
    vertices, inverse = np.unique(tris.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    return vertices.astype(np.float64), faces.astype(np.int64)


def get_backend(name: str, voxel_max_side: int = 48) -> ReconstructionBackend:
    if name == "stub":
        return SilhouetteStubBackend(voxel_max_side)
    if name == "photogrammetry":
        raise ReconstructionError("The photogrammetry backend is not implemented in this build.")
    raise ReconstructionError(f"Unknown reconstruction backend '{name}'.")
