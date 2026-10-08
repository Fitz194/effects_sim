"""Synthetic environments for offline runs and tests.

Population follows a monocentric exponential density model (Clark 1951):
    rho(r) = rho_0 * exp(-r / L)
Buildings are sampled with the same spatial density; class depends on distance from the centre
(dense RC/steel core, masonry/wood suburbs, industrial scattered).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from effects.data.environment import Environment, Grid


def synthetic_city(
    grid: Grid,
    class_names: tuple[str, ...],
    default_mix: np.ndarray,
    total_population: float = 300_000,
    scale_length_m: float = 2_500.0,
    n_buildings: int = 20_000,
    centre_offset_m: tuple[float, float] = (0.0, 0.0),
    seed: int = 0,
) -> Environment:
    rng = np.random.default_rng(seed)
    x, y = grid.mesh()
    r = np.hypot(x - centre_offset_m[0], y - centre_offset_m[1])
    density = np.exp(-r / scale_length_m)
    pop = total_population * density / density.sum()

    # Buildings: sample radius from Gamma(2, L) (exponential areal density -> r*exp(-r/L))
    rb = rng.gamma(2.0, scale_length_m, n_buildings)
    th = rng.uniform(0, 2 * np.pi, n_buildings)
    bx = centre_offset_m[0] + rb * np.cos(th)
    by = centre_offset_m[1] + rb * np.sin(th)

    names = list(class_names)
    core = rb < 0.5 * scale_length_m
    probs_core = _mix(names, {"reinforced_concrete": 0.45, "steel_frame": 0.35, "masonry": 0.2})
    probs_sub = _mix(
        names,
        {"masonry": 0.6, "wood_frame": 0.2, "light_industrial": 0.1, "reinforced_concrete": 0.1},
    )
    cls = np.where(
        core,
        rng.choice(names, n_buildings, p=probs_core),
        rng.choice(names, n_buildings, p=probs_sub),
    )
    tall = np.isin(cls, ["reinforced_concrete", "steel_frame"])
    height = np.where(tall, rng.uniform(12, 60, n_buildings), rng.uniform(5, 10, n_buildings))
    footprint = np.where(
        tall, rng.uniform(400, 2500, n_buildings), rng.uniform(50, 150, n_buildings)
    )
    footprint = np.where(cls == "light_industrial", rng.uniform(800, 5000, n_buildings), footprint)

    buildings = pd.DataFrame(
        {"x": bx, "y": by, "cls": cls, "footprint_m2": footprint, "height_m": height}
    )
    _, _, ok = grid.cell_index(bx, by)
    return Environment(
        grid=grid,
        population=pop,
        buildings=buildings[ok].reset_index(drop=True),
        class_names=tuple(class_names),
        default_mix=np.asarray(default_mix, dtype=float),
        meta={"kind": "synthetic", "seed": seed},
    )


def _mix(names: list[str], weights: dict[str, float]) -> np.ndarray:
    p = np.array([weights.get(n, 0.0) for n in names], dtype=float)
    if p.sum() == 0:
        p[:] = 1.0
    return p / p.sum()
