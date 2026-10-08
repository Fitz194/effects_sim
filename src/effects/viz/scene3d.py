"""Build 2: 3D rendering of a completed run (PyVista / VTK).

Read-only: everything comes from a `effects run` output directory (fields.npz, buildings.csv,
summary.json). Nothing here changes results.

Scene contents
- Ground plane coloured by a chosen field (fatalities, overpressure, population, thermal fluence).
- Buildings extruded as boxes (footprint area -> square side, height from data), coloured by
  expected damage state E[DS] = sum_i i * P(DS = i), 0 = none ... 4 = collapse.
- Overpressure rings (1, 5, 20 psi) with labels, and the burst point.
- Optional shock front: a dome whose ground radius follows the integrated arrival time.
  Buildings stay grey until the front reaches them, then show their damage colour.

Buildings are boxes because buildings.csv stores centroid, footprint area and height only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from effects.units import PSI

# Ordinal ramp for a dark background: undamaged recedes (dim slate), severity gets hotter.
STATE_COLOURS = ["#5f6a78", "#f3d36b", "#f0993e", "#d8512b", "#8e1c22"]  # none -> collapse
PRE_SHOCK = "#a9b1bb"
BG_TOP, BG_BOTTOM = "#1c2431", "#3b4757"
TEXT = "#f2f2f2"
RING_PSI = (1, 5, 20)
BUILDING_BAR = "damage state  0 none  1 light  2 moderate  3 severe  4 collapse"
# Ground ramps run dark -> light so low values sit quietly under the dark UI and text stays legible.
GROUND_CMAPS = {
    "slate": ["#1f2733", "#24425f", "#3b74a3", "#8fc0e0", "#e3f2fa"],
    "violet": ["#211f33", "#3d2f63", "#6c4fa0", "#b39ad6", "#efe6fb"],
}

LAYERS = {
    # name: (fields.npz key, colormap, scalar-bar title, per-km2 density?)
    "population": ("population", "slate", "log10 people / km2", True),
    "fatalities": ("fatal_mean", "violet", "log10 expected fatalities / km2", True),
    "overpressure": ("overpressure_pa", "slate", "log10 peak overpressure (psi)", False),
    "fluence": ("fluence_j_m2", "violet", "log10 thermal fluence (cal/cm2)", False),
}


@dataclass
class RunData:
    centres: np.ndarray
    fields: dict[str, np.ndarray]
    buildings: pd.DataFrame
    summary: dict

    @property
    def cell_m(self) -> float:
        return float(self.centres[1] - self.centres[0])

    @property
    def half_extent(self) -> float:
        return float(self.centres[-1] + self.cell_m / 2)

    @property
    def state_columns(self) -> list[str]:
        return [c for c in self.buildings.columns if c.startswith("p_")]

    def damage_index(self) -> np.ndarray:
        cols = self.state_columns
        if not cols:
            return np.zeros(len(self.buildings))
        p = self.buildings[cols].to_numpy()
        return p @ np.arange(len(cols))

    def blast(self):
        from effects.sources import make_blast

        s = self.summary["source"]
        return make_blast(
            s.get("blast_model", "brode"),
            s["yield_kt"],
            s["burst_height_m"],
            s.get("blast_fraction", 0.5),
            s.get("reflection_factor", 1.8),
            self.summary.get("atmosphere", {}).get("p0_pa", 101_325.0),
        )


def load_run(out_dir: str | Path) -> RunData:
    out = Path(out_dir)
    missing = [f for f in ("fields.npz", "buildings.csv", "summary.json") if not (out / f).exists()]
    if missing:
        raise FileNotFoundError(f"{out} is missing {missing}; run `effects run` first")
    with np.load(out / "fields.npz") as z:
        fields = {k: z[k] for k in z.files}
    summary = json.loads((out / "summary.json").read_text())
    if "source" not in summary:
        raise ValueError("summary.json has no 'source' block; re-run with this version")
    return RunData(fields.pop("centres_m"), fields, pd.read_csv(out / "buildings.csv"), summary)


# ---------------------------------------------------------------- geometry


def buildings_mesh(b: pd.DataFrame, z_scale: float = 1.0, min_side: float = 4.0):
    """All buildings as one PolyData of boxes (5 faces each, no floor)."""
    import pyvista as pv

    n = len(b)
    if n == 0:
        return pv.PolyData()
    x, y = b["x"].to_numpy(), b["y"].to_numpy()
    half = np.maximum(np.sqrt(b["footprint_m2"].to_numpy()), min_side) / 2
    h = np.maximum(b["height_m"].to_numpy(), 2.0) * z_scale
    dx = np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    dy = np.array([-1, -1, 1, 1, -1, -1, 1, 1])
    top = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    pts = np.empty((n, 8, 3))
    pts[..., 0] = x[:, None] + dx * half[:, None]
    pts[..., 1] = y[:, None] + dy * half[:, None]
    pts[..., 2] = top * h[:, None]
    quads = np.array([[4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]])
    idx = quads[None, :, :] + 8 * np.arange(n)[:, None, None]  # (n, 5, 4)
    faces = np.concatenate([np.full((n, 5, 1), 4), idx], axis=-1).ravel()
    mesh = pv.PolyData(pts.reshape(-1, 3), faces)
    mesh.cell_data["building_id"] = np.repeat(np.arange(n), 5)
    return mesh


def ground_mesh(run: RunData, layer: str):
    """ImageData plane with log10 of the chosen layer as cell scalars (NaN where empty)."""
    import pyvista as pv

    key, _, _, density = LAYERS[layer]
    data = run.fields[key].astype(float).copy()
    if density:
        data *= 1e6 / run.cell_m**2
    if layer == "overpressure":
        data = data / PSI
    if layer == "fluence":
        data = data / 41_840.0
    with np.errstate(divide="ignore"):
        logv = np.where(data > 0, np.log10(np.maximum(data, 1e-30)), np.nan)
    if layer == "fatalities":
        logv = np.where(data > 1e-2, logv, np.nan)
    n = len(run.centres)
    g = pv.ImageData(
        dimensions=(n + 1, n + 1, 1),
        spacing=(run.cell_m, run.cell_m, 1.0),
        origin=(-run.half_extent, -run.half_extent, 0.0),
    )
    g.cell_data["value"] = logv.ravel(order="C")
    return g


def ring(radius: float, z: float = 2.0, n: int = 256):
    import pyvista as pv

    th = np.linspace(0, 2 * np.pi, n)
    pts = np.c_[radius * np.cos(th), radius * np.sin(th), np.full(n, z)]
    return pv.lines_from_points(pts)


def dome(radius: float, vertical_scale: float = 1.0):
    import pyvista as pv

    s = pv.Sphere(radius=max(radius, 1.0), theta_resolution=72, phi_resolution=36, end_phi=90.0)
    s.points[:, 2] *= vertical_scale
    return s


# ---------------------------------------------------------------- scene


class Scene:
    """Holds the plotter and the mutable pieces needed for time stepping."""

    def __init__(
        self,
        run: RunData,
        layer: str = "population",
        z_scale: float = 1.5,
        off_screen: bool = True,
        window_size=(1600, 1000),
        show_front: bool = False,
    ):
        import pyvista as pv
        from matplotlib.colors import LinearSegmentedColormap

        from effects.sources.shock import arrival_time_table

        if layer not in LAYERS:
            raise ValueError(f"layer must be one of {list(LAYERS)}")
        self.run = run
        self.blast = run.blast()
        self.z_scale = z_scale
        self.pl = pv.Plotter(off_screen=off_screen, window_size=list(window_size))
        pl = self.pl
        pl.set_background(BG_BOTTOM, top=BG_TOP)

        # ground
        g = ground_mesh(run, layer)
        vals = g.cell_data["value"]
        finite = vals[np.isfinite(vals)]
        clim = (np.percentile(finite, 2), np.percentile(finite, 99.5)) if finite.size else (0, 1)
        _, cmap_name, title, _ = LAYERS[layer]
        cmap = LinearSegmentedColormap.from_list(cmap_name, GROUND_CMAPS[cmap_name])
        pl.add_mesh(
            g,
            scalars="value",
            cmap=cmap,
            clim=clim,
            nan_color="#262d38",
            lighting=False,
            show_scalar_bar=True,
            scalar_bar_args=_bar_args(title, 0.05),
        )
        self._bar_title(title, 0.05)

        # buildings
        self.dmg = run.damage_index()
        r_bld = np.hypot(run.buildings["x"].to_numpy(), run.buildings["y"].to_numpy())
        self.bmesh = buildings_mesh(run.buildings, z_scale)
        self.cell_bid = (
            self.bmesh.cell_data["building_id"] if self.bmesh.n_cells else np.zeros(0, int)
        )
        r_max = np.sqrt(2) * run.half_extent
        self.r_tab, self.t_tab = arrival_time_table(self.blast, r_max)
        self.t_bld = np.interp(r_bld, self.r_tab, self.t_tab)
        if self.bmesh.n_cells:
            self.bmesh.cell_data["display"] = self.dmg[self.cell_bid]
            state_cmap = LinearSegmentedColormap.from_list("states", STATE_COLOURS)
            pl.add_mesh(
                self.bmesh,
                scalars="display",
                cmap=state_cmap,
                clim=(0, 4),
                below_color=PRE_SHOCK,
                smooth_shading=False,
                ambient=0.25,
                scalar_bar_args={**_bar_args(BUILDING_BAR, 0.55), "below_label": " "},
            )
            self._bar_title(BUILDING_BAR, 0.55)

        # rings and burst point
        h = self.blast.burst_height_m
        for psi in RING_PSI:
            r = self.blast.range_for_overpressure(psi * PSI)
            if 0 < r < run.half_extent:
                pl.add_mesh(ring(r), color="#ffd27a", line_width=2.5, lighting=False)
                pl.add_point_labels(
                    np.array([[r * 0.7071, r * 0.7071, 30.0]]),
                    [f"{psi} psi"],
                    font_size=14,
                    text_color=TEXT,
                    shape=None,
                    show_points=False,
                    always_visible=True,
                )
        pl.add_mesh(
            pv.Sphere(radius=max(run.half_extent / 150, 15), center=(0, 0, max(h, 20))),
            color="#fff3c4",
            lighting=False,
        )

        # shock front
        self.front_dome = self.front_ring = None
        if show_front:
            self.front_dome = dome(1.0)
            self.front_ring = ring(1.0, z=3.0)
            pl.add_mesh(self.front_dome, color="#ffffff", opacity=0.12, lighting=False)
            pl.add_mesh(self.front_ring, color="#ffffff", line_width=3, lighting=False)

        title_txt = (
            f"{run.summary.get('name', '')}   {run.summary['source']['yield_kt']:g} kt, "
            f"burst height {h:g} m, {self.blast.__class__.__name__}"
        )
        pl.add_text(title_txt, position="upper_left", font_size=11, color=TEXT)
        f = run.summary.get("fatalities", {})
        if f:
            pl.add_text(
                f"fatalities mean {f['mean']:,.0f}  (90%: {f['p05']:,.0f} - {f['p95']:,.0f})",
                position="upper_right",
                font_size=10,
                color=TEXT,
            )
        self.time_text = pl.add_text(
            " ", position=(0.004, 0.935), viewport=True, font_size=11, color=TEXT, shadow=True
        )
        self._default_camera()

    def _bar_title(self, text: str, x: float):
        self.pl.add_text(
            text, position=(x, 0.095), viewport=True, font_size=9, color=TEXT, shadow=True
        )

    def _default_camera(self):
        e = self.run.half_extent
        self.pl.camera_position = [(-0.75 * e, -1.05 * e, 0.62 * e), (0, -0.08 * e, 0), (0, 0, 1)]
        self.pl.camera.view_angle = 30

    @property
    def t_max(self) -> float:
        return float(self.t_tab[-1])

    def set_time(self, t: float):
        """Advance the shock front to time t (s) and reveal damage behind it."""
        if self.bmesh.n_cells:
            arrived = self.t_bld[self.cell_bid] <= t
            self.bmesh.cell_data["display"] = np.where(arrived, self.dmg[self.cell_bid], -1.0)
        if self.front_dome is not None:
            r = float(np.interp(t, self.t_tab, self.r_tab))
            self.front_dome.copy_from(dome(r, vertical_scale=0.6))
            self.front_ring.copy_from(ring(r, z=3.0))
        self.time_text.SetInput(f"t = {t:6.2f} s after burst   (grey buildings: not yet reached)")

    def screenshot(self, path):
        self.pl.screenshot(str(path))

    def close(self):
        self.pl.close()


def _bar_args(title: str, x: float) -> dict:
    # Titles are drawn separately (see Scene._bar_title) so they never collide with tick labels.
    return dict(
        title=title,
        position_x=x,
        position_y=0.03,
        width=0.38,
        height=0.05,
        color=TEXT,
        title_font_size=1,
        label_font_size=11,
        n_labels=5,
        fmt="%.1f",
    )


# ---------------------------------------------------------------- outputs


def render_screenshot(
    run: RunData, path, layer="population", z_scale=1.5, window_size=(1600, 1000)
):
    sc = Scene(run, layer, z_scale, off_screen=True, window_size=window_size)
    sc.screenshot(path)
    sc.close()


def render_animation(
    run: RunData, path, layer="population", z_scale=1.5, frames=60, window_size=(1280, 800), fps=15
):
    """Shock-front animation as .gif (needs imageio) or .mp4 (needs imageio-ffmpeg)."""
    sc = Scene(run, layer, z_scale, off_screen=True, window_size=window_size, show_front=True)
    # time axis: until the 1 psi radius (or grid edge) is reached
    r_end = min(sc.blast.range_for_overpressure(1 * PSI) * 1.1, run.half_extent)
    t_end = float(np.interp(r_end, sc.r_tab, sc.t_tab))
    path = str(path)
    if path.endswith(".mp4"):
        sc.pl.open_movie(path, framerate=fps)
    else:
        sc.pl.open_gif(path, fps=fps)
    for t in np.concatenate([np.linspace(0, t_end, frames), np.full(fps, t_end)]):
        sc.set_time(t)
        sc.pl.write_frame()
    sc.close()


def export_vtk(run: RunData, out_dir, z_scale=1.0) -> list[Path]:
    """Write ParaView-ready files: buildings.vtp (per-building data) and ground.vti (all fields)."""
    import pyvista as pv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bm = buildings_mesh(run.buildings, z_scale)
    written = []
    if bm.n_cells:
        bid = bm.cell_data["building_id"]
        bm.cell_data["damage_index"] = run.damage_index()[bid]
        for c in run.state_columns:
            bm.cell_data[c] = run.buildings[c].to_numpy()[bid]
        bm.save(out / "buildings.vtp")
        written.append(out / "buildings.vtp")
    n = len(run.centres)
    g = pv.ImageData(
        dimensions=(n + 1, n + 1, 1),
        spacing=(run.cell_m, run.cell_m, 1.0),
        origin=(-run.half_extent, -run.half_extent, 0.0),
    )
    for k, v in run.fields.items():
        g.cell_data[k] = v.astype(float).ravel(order="C")
    g.save(out / "ground.vti")
    written.append(out / "ground.vti")
    return written


def interactive(run: RunData, layer="population", z_scale=1.5):
    """Open a window with a time slider for the shock front. Close the window to exit."""
    sc = Scene(run, layer, z_scale, off_screen=False, show_front=True)
    sc.set_time(sc.t_max)
    sc.pl.add_slider_widget(
        sc.set_time,
        [0.0, sc.t_max],
        value=sc.t_max,
        title="time after burst (s)",
        pointa=(0.55, 0.92),
        pointb=(0.95, 0.92),
        color=TEXT,
        style="modern",
    )
    sc.pl.add_key_event("r", sc._default_camera)
    sc.pl.show()
