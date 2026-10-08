"""Matplotlib 2D outputs. Magnitudes use single-hue sequential maps; damage states are ordinal so
they use one hue light->dark. Overpressure contours are drawn in neutral ink and labelled directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402

from effects.montecarlo import Results  # noqa: E402
from effects.sources import ThermalSource  # noqa: E402
from effects.units import cal_cm2_to_j_m2, pa_to_psi, psi_to_pa  # noqa: E402

INK = "#2b2b2b"
MUTED = "#6b6b6b"
GRID = "#e3e3e3"
MAP_PSI_LEVELS = (1, 5, 20)  # fewer rings on maps so labels don't collide
STATE_COLOURS = ["#eef2f7", "#bcd0e6", "#7ea6cf", "#3f73ae", "#173e73"]  # none -> collapse


def _style(ax):
    ax.tick_params(colors=MUTED, labelsize=8)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def _km(ax):
    ax.set_xlabel("East of ground zero (km)")
    ax.set_ylabel("North of ground zero (km)")
    ax.set_aspect("equal")


def _rings(ax, res: Results):
    x = res.env.grid.centres / 1e3
    psi = pa_to_psi(res.overpressure_nominal)
    levels = [lv for lv in MAP_PSI_LEVELS if psi.min() < lv < psi.max()]
    if levels:
        cs = ax.contour(x, x, psi, levels=levels, colors=INK, linewidths=1.0)
        ax.clabel(cs, fmt=lambda v: f"{v:g} psi", fontsize=7, colors=INK)


def overpressure_vs_range(blast, thermal: ThermalSource, max_range_m: float, path):
    r = np.linspace(0, max_range_m, 800)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    ax.semilogy(r / 1e3, pa_to_psi(blast.overpressure(r)), color="#3f73ae", lw=2)
    ax.set_title("Peak overpressure", loc="left", fontsize=10)
    ax.set_xlabel("Ground range (km)")
    ax.set_ylabel("psi")
    ax.grid(True, which="both", color=GRID, lw=0.5)
    _style(ax)
    ax = axes[1]
    ax.semilogy(r / 1e3, thermal.fluence(r) / 41_840.0, color="#b5542d", lw=2)
    ax.set_title("Thermal fluence (unshielded)", loc="left", fontsize=10)
    ax.set_xlabel("Ground range (km)")
    ax.set_ylabel("cal/cm²")
    ax.grid(True, which="both", color=GRID, lw=0.5)
    _style(ax)
    fig.suptitle(
        f"{blast.yield_kt:g} kt, burst height {blast.burst_height_m:g} m",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def population_map(res: Results, path):
    g = res.env.grid
    fig, ax = plt.subplots(figsize=(7, 6))
    dens = res.env.population / (g.cell_m**2 / 1e6)
    dens = np.where(dens > 0, dens, np.nan)
    im = ax.imshow(
        dens,
        origin="lower",
        extent=np.array(g.extent) / 1e3,
        cmap="Greys",
        norm=LogNorm(vmin=max(np.nanmin(dens), 1), vmax=np.nanmax(dens)),
    )
    fig.colorbar(im, ax=ax, label="people / km²", shrink=0.8)
    _rings(ax, res)
    ax.set_title("Population density with overpressure contours", loc="left", fontsize=10)
    _km(ax)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fatality_map(res: Results, path):
    g = res.env.grid
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    per_km2 = 1e6 / g.cell_m**2
    for ax, data, title in (
        (axes[0], res.fatal_mean * per_km2, "Expected fatalities per km² (mean)"),
        (axes[1], res.fatal_std * per_km2, "Fatalities per km², std. dev. across samples"),
    ):
        d = np.where(data > 1e-3, data, np.nan)
        vmax = np.nanmax(d) if np.isfinite(d).any() else 1.0
        im = ax.imshow(
            d,
            origin="lower",
            extent=np.array(g.extent) / 1e3,
            cmap="Reds",
            norm=LogNorm(vmin=max(vmax * 1e-4, 1e-3), vmax=vmax),
        )
        fig.colorbar(im, ax=ax, shrink=0.8)
        _rings(ax, res)
        ax.set_title(title, loc="left", fontsize=10)
        _km(ax)
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def building_damage_map(res: Results, path):
    b = res.env.buildings
    if not len(b):
        return
    labels = ("none",) + res.states
    likely = res.building_state_probs.argmax(axis=1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.set_facecolor("white")
    cmap = ListedColormap(STATE_COLOURS[: len(labels)])
    order = np.argsort(likely)  # draw severe on top
    size = np.clip(np.sqrt(b["footprint_m2"].to_numpy()) / 8, 1, 12)
    ax.scatter(
        b["x"].to_numpy()[order] / 1e3,
        b["y"].to_numpy()[order] / 1e3,
        s=size[order],
        c=likely[order],
        cmap=cmap,
        vmin=-0.5,
        vmax=len(labels) - 0.5,
        linewidths=0,
    )
    _rings(ax, res)
    handles = [
        plt.Line2D([], [], ls="", marker="o", ms=7, mfc=STATE_COLOURS[i], mec=GRID, label=labels[i])
        for i in range(len(labels))
    ]
    ax.legend(
        handles=handles,
        title="Most likely state",
        fontsize=8,
        title_fontsize=8,
        loc="upper right",
        frameon=True,
    )
    ext = np.array(res.env.grid.extent) / 1e3
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_title("Building damage (most likely state, mean over samples)", loc="left", fontsize=10)
    _km(ax)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def totals_histogram(res: Results, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, data, title, col in (
        (axes[0], res.fatalities, "Total fatalities", "#a33a2a"),
        (axes[1], res.injuries, "Total injuries", "#c28a2c"),
    ):
        ax.hist(data, bins=40, color=col, edgecolor="white", linewidth=0.5)
        for q, ls in ((5, ":"), (50, "-"), (95, ":")):
            v = np.percentile(data, q)
            ax.axvline(v, color=INK, ls=ls, lw=1)
            ax.annotate(
                f"P{q}\n{v:,.0f}",
                (v, ax.get_ylim()[1] * 0.92),
                fontsize=7,
                color=INK,
                ha="left",
                xytext=(3, 0),
                textcoords="offset points",
            )
        ax.set_title(f"{title} across {len(data)} Monte Carlo samples", loc="left", fontsize=10)
        ax.set_xlabel("people")
        ax.set_ylabel("samples")
        _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_all(res: Results, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    from effects.scenario import nominal_sources

    blast, therm = nominal_sources(res.scenario)
    max_r = max(
        blast.range_for_overpressure(psi_to_pa(0.5)),
        therm.range_for_fluence(cal_cm2_to_j_m2(1.0)),
        1_000.0,
    )
    paths = {
        "range_curves.png": lambda p: overpressure_vs_range(blast, therm, max_r, p),
        "population.png": lambda p: population_map(res, p),
        "fatalities.png": lambda p: fatality_map(res, p),
        "buildings.png": lambda p: building_damage_map(res, p),
        "totals.png": lambda p: totals_histogram(res, p),
    }
    written = []
    for name, fn in paths.items():
        p = out_dir / name
        fn(p)
        if p.exists():
            written.append(p)
    return written
