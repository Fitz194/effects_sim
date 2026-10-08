"""Lognormal fragility curves for building damage states.

P(DS >= ds | p) = Phi( ln(p / theta_ds) / beta_ds )

Overpressure-only curves are appropriate for long-duration (large-yield) blast, where loading is
quasi-static and peak overpressure governs. For short-duration loads, pressure-impulse (P-I)
curves would be needed; that is a documented extension point.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources

import numpy as np
import yaml
from scipy.special import ndtr


@dataclass(frozen=True)
class LognormalCurve:
    median: float  # Pa
    beta: float

    def prob(self, x, median_scale=1.0):
        x = np.maximum(np.asarray(x, dtype=float), 1e-12)
        return ndtr(np.log(x / (self.median * median_scale)) / self.beta)


@dataclass(frozen=True)
class FragilitySet:
    """Ordered damage-state exceedance curves for one building class."""

    name: str
    states: tuple[str, ...]  # increasing severity, excludes "none"
    curves: tuple[LognormalCurve, ...]
    description: str = ""
    # Drag-sensitive structures (Glasstone & Dolan 1977, Sec. 5.137-5.138) are damaged by dynamic
    # pressure, so an overpressure-only curve is only valid near the yield it was derived for.
    drag_sensitive: bool = False
    reference_yield_kt: float | None = None

    def exceedance(self, p, median_scale=1.0):
        """Array (..., n_states): P(DS >= state). Forced monotone non-increasing in severity."""
        ex = np.stack([c.prob(p, median_scale) for c in self.curves], axis=-1)
        return np.minimum.accumulate(ex, axis=-1)

    def state_probs(self, p, median_scale=1.0):
        """Array (..., n_states + 1): P(DS == state), with index 0 = 'none'. Sums to 1."""
        ex = self.exceedance(p, median_scale)
        ones = np.ones(ex.shape[:-1] + (1,))
        zeros = np.zeros(ex.shape[:-1] + (1,))
        full = np.concatenate([ones, ex, zeros], axis=-1)
        return full[..., :-1] - full[..., 1:]


def load_fragility_library(path=None) -> dict[str, FragilitySet]:
    """Load fragility sets from YAML. Defaults to the packaged library (provenance in the YAML)."""
    if path is None:
        text = resources.files("effects.fragility").joinpath("default_fragility.yaml").read_text()
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()
    raw = yaml.safe_load(text)
    states = tuple(raw["states"])
    lib = {}
    for name, spec in raw["classes"].items():
        curves = tuple(
            LognormalCurve(median=float(spec["median_kpa"][s]) * 1e3, beta=float(spec["beta"][s]))
            for s in states
        )
        lib[name] = FragilitySet(
            name,
            states,
            curves,
            spec.get("description", ""),
            bool(spec.get("drag_sensitive", False)),
            float(spec["reference_yield_kt"]) if spec.get("reference_yield_kt") else None,
        )
    return lib


YIELD_WARNING_FACTOR = 3.0


def yield_warnings(sets, yield_kt: float) -> list[str]:
    """Warn for each drag-sensitive class whose curves were derived at a very different yield."""
    out = []
    for fs in sets:
        ref = fs.reference_yield_kt
        if (
            fs.drag_sensitive
            and ref
            and not (ref / YIELD_WARNING_FACTOR <= yield_kt <= ref * YIELD_WARNING_FACTOR)
        ):
            out.append(
                f"WARNING: building class '{fs.name}' is drag-sensitive and its fragility curves "
                f"were derived at {ref:g} kt; the scenario yield is {yield_kt:g} kt, so its "
                "damage is likely mis-estimated (Glasstone & Dolan 1977, Sec. 5.137-5.138)."
            )
    return out
