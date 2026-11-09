import numpy as np
import pytest

from app.exporters import read_3mf, read_stl_binary, write_3mf, write_stl_binary
from app.meshops import (
    compact,
    edge_report,
    extents,
    orient_outward,
    overhang_fraction,
    place_on_plate,
    remove_degenerate,
    signed_volume,
    weld,
)
from app.printcheck import run_print_checks
from app.reconstruct import voxels_to_mesh


def box_voxels(nx, ny, nz):
    return np.ones((nx, ny, nz), dtype=bool)


def test_single_voxel_is_a_closed_unit_cube():
    v, f = voxels_to_mesh(box_voxels(1, 1, 1))
    assert len(v) == 8
    assert len(f) == 12
    report = edge_report(f)
    assert report["boundary_edges"] == 0
    assert report["non_manifold_edges"] == 0
    assert signed_volume(v, f) == pytest.approx(1.0)


def test_box_volume_and_extents_after_voxel_meshing():
    v, f = voxels_to_mesh(box_voxels(2, 3, 4))
    assert signed_volume(v, f) == pytest.approx(24.0)
    assert extents(v).tolist() == [2.0, 3.0, 4.0]


def test_voxel_faces_point_outward():
    v, f = voxels_to_mesh(box_voxels(2, 2, 2))
    assert signed_volume(v, f) > 0
    centre = v.mean(axis=0)
    tri = v[f]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    outward = np.einsum("ij,ij->i", normals, tri.mean(axis=1) - centre)
    assert np.all(outward > 0)


def test_two_touching_voxels_are_watertight():
    grid = np.zeros((2, 1, 1), dtype=bool)
    grid[:, 0, 0] = True
    v, f = voxels_to_mesh(grid)
    assert edge_report(f)["boundary_edges"] == 0
    assert signed_volume(v, f) == pytest.approx(2.0)


def test_weld_merges_duplicate_vertices():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    f = np.array([[0, 1, 2], [3, 4, 5]])
    wv, wf = weld(v, f)
    assert len(wv) == 3
    assert np.allclose(wv[wf[0]], v[f[0]])
    assert np.allclose(wv[wf[1]], v[f[1]])
    assert len(set(wf[0].tolist())) == 3


def test_remove_degenerate_drops_zero_area_faces():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0]], dtype=float)
    f = np.array([[0, 1, 2], [0, 1, 3]])  # second is collinear
    assert remove_degenerate(v, f).tolist() == [[0, 1, 2]]


def test_compact_drops_unused_vertices():
    v = np.arange(12, dtype=float).reshape(4, 3)
    f = np.array([[0, 1, 3]])
    cv, cf = compact(v, f)
    assert len(cv) == 3
    assert cf.tolist() == [[0, 1, 2]]


def test_orient_outward_flips_inside_out_mesh():
    v, f = voxels_to_mesh(box_voxels(1, 1, 1))
    flipped = f[:, ::-1]
    assert signed_volume(v, flipped) < 0
    assert signed_volume(v, orient_outward(v, flipped)) > 0


def test_place_on_plate_rests_model_on_z_zero_and_centres_xy():
    v = np.array([[10, 10, 5], [12, 14, 5], [11, 12, 9]], dtype=float)
    p = place_on_plate(v)
    assert p[:, 2].min() == pytest.approx(0.0)
    assert (p[:, 0].min() + p[:, 0].max()) / 2 == pytest.approx(0.0)
    assert (p[:, 1].min() + p[:, 1].max()) / 2 == pytest.approx(0.0)


def test_overhang_fraction_flags_only_the_bottom_of_a_cube():
    v, f = voxels_to_mesh(box_voxels(1, 1, 1))
    fraction, count = overhang_fraction(v, f, 45.0)
    assert count == 2  # the two triangles of the bottom face
    assert fraction == pytest.approx(1 / 6)


def test_stl_round_trip_preserves_geometry(tmp_path):
    v, f = voxels_to_mesh(box_voxels(2, 3, 4))
    path = tmp_path / "box.stl"
    write_stl_binary(path, v, f)
    rv, rf = read_stl_binary(path)
    assert extents(rv).tolist() == [2.0, 3.0, 4.0]
    assert signed_volume(rv, rf) == pytest.approx(24.0)


def test_3mf_round_trip_preserves_geometry(tmp_path):
    v, f = voxels_to_mesh(box_voxels(2, 2, 2))
    path = tmp_path / "box.3mf"
    write_3mf(path, v, f)
    rv, rf = read_3mf(path)
    assert len(rv) == len(v)
    assert len(rf) == len(f)
    assert signed_volume(rv, rf) == pytest.approx(8.0)


def test_print_checks_pass_for_a_small_closed_box():
    v, f = voxels_to_mesh(box_voxels(20, 20, 20))
    v = place_on_plate(v)
    f = orient_outward(v, f)
    report = run_print_checks(v, f, (220, 220, 250), 45.0, 300_000)
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["watertight"] == "pass"
    assert statuses["manifold"] == "pass"
    assert statuses["build_volume"] == "pass"
    assert statuses["wall_thickness"] == "not_checked"


def test_print_checks_fail_when_larger_than_build_volume():
    v, f = voxels_to_mesh(box_voxels(300, 10, 10))
    report = run_print_checks(v, f, (220, 220, 250), 45.0, 300_000)
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["build_volume"] == "fail"
    assert report["status"] == "fail"


def test_print_checks_warn_when_only_rotation_fits():
    # 240 mm long, 100 mm wide and tall: too long for the plate as placed, fits when laid on its side
    v, f = voxels_to_mesh(box_voxels(240, 100, 100))
    report = run_print_checks(v, f, (220, 220, 250), 45.0, 300_000)
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["build_volume"] == "warn"


def test_open_mesh_fails_watertight_check():
    v, f = voxels_to_mesh(box_voxels(1, 1, 1))
    report = run_print_checks(v, f[:-2], (220, 220, 250), 45.0, 300_000)
    statuses = {c["name"]: c["status"] for c in report["checks"]}
    assert statuses["watertight"] == "fail"
