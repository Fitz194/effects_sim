"""Build 2 renderer: off-screen smoke tests. Skipped when the [viz] extra is not installed."""

from pathlib import Path

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")
pv.OFF_SCREEN = True

from effects.cli import main  # noqa: E402
from effects.viz import scene3d  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    main(
        [
            "run",
            str(ROOT / "scenarios" / "synthetic_city.yaml"),
            "--samples",
            "4",
            "--out",
            str(out),
        ]
    )
    return out


def test_load_run(run_dir):
    run = scene3d.load_run(run_dir)
    assert run.fields["population"].shape == (len(run.centres),) * 2
    d = run.damage_index()
    assert np.all((d >= 0) & (d <= 4))


def test_buildings_mesh_counts(run_dir):
    run = scene3d.load_run(run_dir)
    m = scene3d.buildings_mesh(run.buildings)
    assert m.n_points == 8 * len(run.buildings)
    assert m.n_cells == 5 * len(run.buildings)


def test_ground_mesh_cell_order_matches_field(run_dir):
    run = scene3d.load_run(run_dir)
    g = scene3d.ground_mesh(run, "overpressure")
    n = len(run.centres)
    # cell (row=j, col=i) centre must sit at (centres[i], centres[j])
    j, i = 3, n - 5
    cid = j * n + i
    c = g.cell_centers().points[cid]
    assert c[0] == pytest.approx(run.centres[i])
    assert c[1] == pytest.approx(run.centres[j])


@pytest.mark.parametrize("layer", list(scene3d.LAYERS))
def test_screenshot_renders_nonblank(run_dir, tmp_path, layer):
    png = tmp_path / f"{layer}.png"
    scene3d.render_screenshot(
        scene3d.load_run(run_dir), png, layer=layer, window_size=(640, 400)
    )
    from PIL import Image

    img = np.asarray(Image.open(png).convert("L"), dtype=float)
    assert img.std() > 10  # not a flat frame


def test_time_stepping_reveals_buildings(run_dir):
    sc = scene3d.Scene(
        scene3d.load_run(run_dir), off_screen=True, window_size=(320, 200), show_front=True
    )
    sc.set_time(0.0)
    early = (sc.bmesh.cell_data["display"] >= 0).mean()
    sc.set_time(sc.t_max)
    late = (sc.bmesh.cell_data["display"] >= 0).mean()
    sc.close()
    assert early < 0.01 and late == pytest.approx(1.0)


def test_cli_view_outputs(run_dir, tmp_path):
    main(
        [
            "view",
            str(run_dir),
            "--screenshot",
            str(tmp_path / "a.png"),
            "--animate",
            str(tmp_path / "a.gif"),
            "--frames",
            "3",
            "--export-vtk",
            str(tmp_path / "vtk"),
        ]
    )
    for f in ("a.png", "a.gif", "vtk/buildings.vtp", "vtk/ground.vti"):
        assert (tmp_path / f).exists()
