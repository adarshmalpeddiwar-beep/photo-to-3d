"""Read and write STL (binary) and 3MF files. Standard library and NumPy only."""
from __future__ import annotations

import struct
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from .meshops import face_normals, weld

STL_DTYPE = np.dtype([("normal", "<f4", (3,)), ("v", "<f4", (3, 3)), ("attr", "<u2")])
MODEL_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
_CONTENT_TYPES = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
    "</Types>"
)
_RELS = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
    'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>'
    "</Relationships>"
)


def write_stl_binary(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    data = np.zeros(len(faces), dtype=STL_DTYPE)
    if len(faces):
        normals, _ = face_normals(vertices, faces)
        data["normal"] = normals.astype(np.float32)
        data["v"] = vertices[faces].astype(np.float32)
    header = b"photo3d binary STL, millimetres".ljust(80, b" ")
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(struct.pack("<I", len(faces)))
        fh.write(data.tobytes())


def read_stl_binary(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(path).read_bytes()
    if len(raw) < 84:
        raise ValueError("File is too short to be a binary STL.")
    count = struct.unpack_from("<I", raw, 80)[0]
    if len(raw) != 84 + count * 50:
        raise ValueError("Not a binary STL, or the file is truncated.")
    data = np.frombuffer(raw, dtype=STL_DTYPE, count=count, offset=84)
    vertices = data["v"].astype(np.float64).reshape(-1, 3)
    faces = np.arange(3 * count, dtype=np.int64).reshape(-1, 3)
    return weld(vertices, faces)


def write_3mf(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<model unit="millimeter" xml:lang="en-US" xmlns="{MODEL_NS}">',
        "<resources>",
        '<object id="1" type="model">',
        "<mesh>",
        "<vertices>",
    ]
    lines += [f'<vertex x="{x:.6f}" y="{y:.6f}" z="{z:.6f}"/>' for x, y, z in vertices]
    lines += ["</vertices>", "<triangles>"]
    lines += [f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in faces]
    lines += ["</triangles>", "</mesh>", "</object>", "</resources>",
              '<build><item objectid="1"/></build>', "</model>"]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", _CONTENT_TYPES)
        zf.writestr("_rels/.rels", _RELS)
        zf.writestr("3D/3dmodel.model", "\n".join(lines))


def read_3mf(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ns = {"m": MODEL_NS}
    with zipfile.ZipFile(path) as zf:
        root = ET.fromstring(zf.read("3D/3dmodel.model"))
    vertices = np.array(
        [(float(v.get("x")), float(v.get("y")), float(v.get("z"))) for v in root.iterfind(".//m:vertex", ns)],
        dtype=np.float64,
    ).reshape(-1, 3)
    faces = np.array(
        [(int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))) for t in root.iterfind(".//m:triangle", ns)],
        dtype=np.int64,
    ).reshape(-1, 3)
    return vertices, faces
