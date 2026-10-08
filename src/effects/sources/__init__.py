"""Source models: blast overpressure/impulse, thermal fluence, impact energy.

Pure functions, SI units, no I/O. Each model documents its reference and validity range.
"""

from effects.sources.blast import NuclearBlast, impact_energy_kt, kg_impulse, kg_overpressure
from effects.sources.nuclear_fits import BrodeBlast, dna_free_air_1kt_pa
from effects.sources.thermal import ThermalSource

BLAST_MODELS = ("brode", "kinney_graham")


def make_blast(
    model: str,
    yield_kt: float,
    burst_height_m: float,
    blast_fraction: float = 0.5,
    reflection_factor: float = 1.8,
    p0: float = 101_325.0,
):
    """Construct a blast model exposing overpressure(r) and range_for_overpressure(p).

    blast_fraction and reflection_factor only apply to the TNT-equivalent (kinney_graham) model.
    """
    if model == "brode":
        return BrodeBlast(yield_kt, burst_height_m, p0)
    if model == "kinney_graham":
        return NuclearBlast(yield_kt, burst_height_m, blast_fraction, reflection_factor, p0)
    raise ValueError(f"unknown blast model {model!r}; choose from {BLAST_MODELS}")


__all__ = [
    "BLAST_MODELS",
    "BrodeBlast",
    "NuclearBlast",
    "ThermalSource",
    "dna_free_air_1kt_pa",
    "impact_energy_kt",
    "kg_impulse",
    "kg_overpressure",
    "make_blast",
]
