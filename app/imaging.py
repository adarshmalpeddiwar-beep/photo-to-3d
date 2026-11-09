"""Photo validation, EXIF-safe decoding and silhouette segmentation (NumPy and Pillow only)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps

ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class PhotoError(ValueError):
    """The uploaded file is not a usable photo."""


class SegmentationError(ValueError):
    """The object could not be separated from the background."""


def decode_photo(raw: bytes, max_pixels: int) -> Image.Image:
    """Validate an uploaded photo and return it as an upright RGB image.

    The caller saves the result as PNG, which drops EXIF metadata such as GPS tags.
    """
    try:
        probe = Image.open(io.BytesIO(raw))
        fmt = probe.format
        width, height = probe.size
    except Exception as exc:  # Pillow raises several exception types
        raise PhotoError("Could not read this file as an image.") from exc

    if fmt not in ALLOWED_FORMATS:
        raise PhotoError(f"Unsupported format {fmt or 'unknown'}. Use JPEG, PNG or WebP.")
    if width * height > max_pixels:
        raise PhotoError("Image is too large. Resize it to about 40 megapixels or less.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as exc:
        raise PhotoError("The image is damaged and could not be decoded.") from exc

    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def otsu_threshold(values: np.ndarray) -> float:
    """Return the Otsu threshold for a 1D array of values."""
    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2.0
    p = hist / max(hist.sum(), 1)
    w0 = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_total = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_total * w0 - mu) ** 2 / (w0 * (1.0 - w0))
    between = np.nan_to_num(between, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return float(centers[int(np.argmax(between))])


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 4-connected region of True pixels."""
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: list[tuple[int, int]] = []
    for y0, x0 in zip(*np.nonzero(mask)):
        y0, x0 = int(y0), int(x0)
        if visited[y0, x0]:
            continue
        visited[y0, x0] = True
        stack = [(y0, x0)]
        region: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            region.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        if len(region) > len(best):
            best = region
    out = np.zeros_like(mask, dtype=bool)
    if best:
        ys, xs = zip(*best)
        out[list(ys), list(xs)] = True
    return out


def segment_object(image: Image.Image, max_side: int = 256) -> tuple[np.ndarray, float]:
    """Return a boolean silhouette mask (at analysis resolution) and the threshold used.

    Assumes the object contrasts with a roughly uniform background. Shadows and
    cluttered backgrounds will produce poor masks; the UI tells users to use a plain background.
    """
    small = image.copy()
    small.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    arr = np.asarray(small, dtype=np.float32)

    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    background = np.median(border, axis=0)
    distance = np.linalg.norm(arr - background, axis=2)

    threshold = otsu_threshold(distance.ravel())
    mask = largest_component(distance > threshold)

    coverage = float(mask.mean())
    if coverage < 0.02:
        raise SegmentationError("Could not find an object. Try a plainer background and even lighting.")
    if coverage > 0.9:
        raise SegmentationError("The object fills almost the whole photo, or the background is not uniform.")
    return mask, threshold
