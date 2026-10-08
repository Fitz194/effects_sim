"""Environment container: a local metric grid centred on ground zero, population, buildings.

Coordinates are local metres (x east, y north) with ground zero at (0, 0).
Real-world loaders project into a local UTM CRS and shift to this frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Grid:
    half_width_m: float
    cell_m: float

    @property
    def n(self) -> int:
        return int(np.ceil(2 * self.half_width_m / self.cell_m))

    @property
    def centres(self) -> np.ndarray:
        """1D cell-centre coordinates (m), same for x and y."""
        return -self.half_width_m + self.cell_m * (np.arange(self.n) + 0.5)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        e = self.n * self.cell_m / 2
        return (-e, e, -e, e)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        c = self.centres
        return np.meshgrid(c, c)  # X[j, i], Y[j, i]; row j = y

    def ground_range(self) -> np.ndarray:
        x, y = self.mesh()
        return np.hypot(x, y)

    def cell_index(self, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, col, valid) for points in local metres."""
        e = self.n * self.cell_m / 2
        col = np.floor((np.asarray(x) + e) / self.cell_m).astype(int)
        row = np.floor((np.asarray(y) + e) / self.cell_m).astype(int)
        valid = (col >= 0) & (col < self.n) & (row >= 0) & (row < self.n)
        return row, col, valid


BUILDING_COLUMNS = ["x", "y", "cls", "footprint_m2", "height_m"]


@dataclass
class Environment:
    grid: Grid
    population: np.ndarray  # (n, n) people per cell
    buildings: pd.DataFrame  # columns BUILDING_COLUMNS
    class_names: tuple[str, ...]
    default_mix: np.ndarray  # (n_classes,) fallback building-class mix
    meta: dict = field(default_factory=dict)

    def class_mix(self) -> np.ndarray:
        """(n, n, n_classes) footprint-area-weighted class mix per cell.

        Cells without buildings use default_mix. This is what indoor population is assumed to
        occupy.
        """
        n, k = self.grid.n, len(self.class_names)
        mix = np.zeros((n, n, k))
        b = self.buildings
        if len(b):
            row, col, ok = self.grid.cell_index(b["x"].to_numpy(), b["y"].to_numpy())
            cls_idx = pd.Categorical(b["cls"], categories=self.class_names).codes
            ok &= cls_idx >= 0
            area = b["footprint_m2"].to_numpy() * np.maximum(b["height_m"].to_numpy() / 3.0, 1.0)
            np.add.at(mix, (row[ok], col[ok], cls_idx[ok]), area[ok])
        tot = mix.sum(axis=-1, keepdims=True)
        default = np.broadcast_to(self.default_mix / self.default_mix.sum(), mix.shape)
        return np.where(tot > 0, mix / np.where(tot > 0, tot, 1.0), default)
