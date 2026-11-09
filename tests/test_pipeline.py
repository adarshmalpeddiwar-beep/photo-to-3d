import io
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.config import Settings
from app.exporters import read_3mf, read_stl_binary
from app.imaging import PhotoError, decode_photo, largest_component, otsu_threshold, segment_object
from app.jobs import JobError, JobService
from app.meshops import extents, signed_volume
from app.pipeline import PipelineError, run_pipeline


def make_settings(tmp_path):
    return Settings(data_dir=str(tmp_path), backend="stub", app_token=None)


def synthetic_photo(size=(400, 300)) -> Image.Image:
    """Dark object on a light, uniform background."""
    img = Image.new("RGB", size, (235, 235, 235))
    draw = ImageDraw.Draw(img)
    w, h = size
    draw.rectangle([w * 0.25, h * 0.3, w * 0.75, h * 0.7], fill=(40, 60, 90))
    return img


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----- imaging ---------------------------------------------------------------

def test_decode_photo_rejects_non_images():
    with pytest.raises(PhotoError):
        decode_photo(b"this is not an image", max_pixels=1_000_000)


def test_decode_photo_rejects_oversized_images():
    with pytest.raises(PhotoError):
        decode_photo(png_bytes(synthetic_photo((400, 300))), max_pixels=1000)


def test_decode_photo_returns_rgb():
    img = decode_photo(png_bytes(synthetic_photo()), max_pixels=1_000_000)
    assert img.mode == "RGB"
    assert img.size == (400, 300)


def test_largest_component_keeps_biggest_blob():
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:3, 1:3] = True      # 4 pixels
    mask[5:9, 5:9] = True      # 16 pixels
    out = largest_component(mask)
    assert out.sum() == 16
    assert not out[1, 1]


def test_otsu_separates_two_clusters():
    values = np.concatenate([np.full(500, 5.0), np.full(500, 200.0)])
    thr = otsu_threshold(values)
    assert 5.0 < thr < 200.0


def test_segment_object_finds_the_rectangle():
    mask, _ = segment_object(synthetic_photo(), max_side=256)
    assert 0.1 < mask.mean() < 0.5


# ----- pipeline ----------------------------------------------------------------

def write_photo(job_dir, img):
    (job_dir / "photos").mkdir(parents=True, exist_ok=True)
    img.save(job_dir / "photos" / "photo-1.png", format="PNG")


def test_pipeline_with_reference_width_matches_that_width(tmp_path):
    settings = make_settings(tmp_path)
    job_dir = tmp_path / "job"
    write_photo(job_dir, synthetic_photo())
    result = run_pipeline(job_dir, settings, 50.0, lambda stage: None)

    assert result["scale"]["source"] == "reference_width"
    v, f = read_stl_binary(job_dir / "mesh.stl")
    assert extents(v)[0] == pytest.approx(50.0, rel=1e-4)
    assert result["report"]["status"] in {"pass", "warn"}
    assert result["report"]["checks"][0]["status"] == "pass"  # watertight
    assert (job_dir / "model.3mf").exists()
    assert (job_dir / "mask.png").exists()


def test_pipeline_without_reference_is_normalised_and_flagged(tmp_path):
    settings = make_settings(tmp_path)
    job_dir = tmp_path / "job"
    write_photo(job_dir, synthetic_photo())
    result = run_pipeline(job_dir, settings, None, lambda stage: None)

    assert result["scale"]["source"] == "normalised"
    v, _ = read_stl_binary(job_dir / "mesh.stl")
    assert extents(v).max() == pytest.approx(settings.default_longest_mm, rel=1e-4)
    assert any("unscaled" in note for note in result["notes"])


def test_pipeline_exports_match_each_other(tmp_path):
    settings = make_settings(tmp_path)
    job_dir = tmp_path / "job"
    write_photo(job_dir, synthetic_photo())
    run_pipeline(job_dir, settings, 60.0, lambda stage: None)
    sv, sf = read_stl_binary(job_dir / "mesh.stl")
    mv, mf = read_3mf(job_dir / "model.3mf")
    assert extents(sv) == pytest.approx(extents(mv), rel=1e-4)
    assert signed_volume(sv, sf) == pytest.approx(signed_volume(mv, mf), rel=1e-3)


def test_pipeline_reports_progress_in_order(tmp_path):
    settings = make_settings(tmp_path)
    job_dir = tmp_path / "job"
    write_photo(job_dir, synthetic_photo())
    seen = []
    run_pipeline(job_dir, settings, None, seen.append)
    assert seen == ["segment", "reconstruct", "scale", "clean", "check", "export"]


def test_pipeline_fails_clearly_on_blank_photo(tmp_path):
    settings = make_settings(tmp_path)
    job_dir = tmp_path / "job"
    write_photo(job_dir, Image.new("RGB", (200, 200), (128, 128, 128)))
    with pytest.raises(PipelineError) as info:
        run_pipeline(job_dir, settings, None, lambda stage: None)
    assert info.value.stage == "segment"


# ----- service -----------------------------------------------------------------

def test_service_full_flow_and_validation(tmp_path):
    settings = make_settings(tmp_path)
    service = JobService(settings)

    with pytest.raises(JobError) as bad_ref:
        service.create(reference_width_mm=-5)
    assert bad_ref.value.status == 400

    job = service.create(reference_width_mm=40, name="  Bracket  ")
    assert job["name"] == "Bracket"
    assert job["status"] == "draft"

    with pytest.raises(JobError):
        service.start(job["id"])  # no photos yet

    service.add_photo(job["id"], png_bytes(synthetic_photo()))
    started = service.start(job["id"])
    assert started["status"] == "queued"

    import time
    for _ in range(200):
        current = service.get(job["id"])
        if current["status"] in {"done", "failed"}:
            break
        time.sleep(0.05)
    assert current["status"] == "done", current.get("error")
    assert current["report"]["status"] in {"pass", "warn"}
    assert service.artifact_path(job["id"], "mesh.stl").is_file()

    with pytest.raises(JobError):
        service.artifact_path(job["id"], "../../etc/passwd")
    with pytest.raises(JobError):
        service.get("not-a-real-id")


def test_service_rate_limits_starts(tmp_path):
    settings = Settings(data_dir=str(tmp_path), backend="stub", app_token=None, jobs_per_window=1)
    service = JobService(settings)
    first = service.create()
    service.add_photo(first["id"], png_bytes(synthetic_photo()))
    service.start(first["id"])
    second = service.create()
    service.add_photo(second["id"], png_bytes(synthetic_photo()))
    with pytest.raises(JobError) as info:
        service.start(second["id"])
    assert info.value.status == 429


def test_stub_depth_follows_object_not_photo(tmp_path):
    # A wide, short object: depth should follow its smaller side, not the photo's size.
    from app.reconstruct import SilhouetteStubBackend
    mask = np.zeros((300, 400), dtype=bool)
    mask[100:200, 100:300] = True  # 200 x 100 px block, 2:1
    raw = SilhouetteStubBackend(48).reconstruct(mask)
    size = extents(raw.vertices)
    assert size[0] == pytest.approx(2 * size[1], rel=0.05)  # width is twice the height
    assert size[2] == pytest.approx(size[1], rel=0.05)      # depth follows the smaller side
