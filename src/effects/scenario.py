"""Scenario schema (YAML) and environment construction."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, Field, model_validator

from effects.data import Environment, Grid, synthetic_city
from effects.fragility import load_fragility_library


class SourceSpec(BaseModel):
    type: Literal["nuclear"] = "nuclear"
    yield_kt: float = Field(gt=0)
    burst_height_m: float = Field(default=0.0, ge=0)
    # brode: nuclear height-of-burst fit (default). kinney_graham: TNT-equivalent, surface-valid.
    blast_model: Literal["brode", "kinney_graham"] = "brode"
    blast_fraction: float = Field(default=0.5, gt=0, le=1)  # kinney_graham only
    thermal_fraction: float = Field(default=0.35, ge=0, le=1)
    reflection_factor: float = Field(default=1.8, ge=1, le=2)  # kinney_graham only


class AtmosphereSpec(BaseModel):
    p0_pa: float = 101_325.0
    visibility_m: float = Field(default=20_000.0, gt=0)


class SyntheticSpec(BaseModel):
    total_population: float = 300_000
    scale_length_m: float = 2_500.0
    n_buildings: int = 20_000
    centre_offset_m: tuple[float, float] = (0.0, 0.0)
    seed: int = 0


class RealSpec(BaseModel):
    lat: float
    lon: float
    population_raster: str | None = None
    osm_buildings: bool = True


class EnvironmentSpec(BaseModel):
    type: Literal["synthetic", "real"] = "synthetic"
    half_width_m: float = Field(default=8_000.0, gt=0)
    cell_m: float = Field(default=50.0, gt=0)
    fragility_library: str | None = None
    default_class_mix: dict[str, float] = {
        "masonry": 0.6,
        "wood_frame": 0.1,
        "reinforced_concrete": 0.15,
        "steel_frame": 0.1,
        "light_industrial": 0.05,
    }
    synthetic: SyntheticSpec = SyntheticSpec()
    real: RealSpec | None = None

    @model_validator(mode="after")
    def _check(self):
        if self.type == "real" and self.real is None:
            raise ValueError("environment.type 'real' needs an environment.real block")
        if (2 * self.half_width_m / self.cell_m) > 2000:
            raise ValueError("grid too large (>2000 cells per side); increase cell_m")
        return self


class UncertaintySpec(BaseModel):
    """Distributions sampled once per Monte Carlo realisation."""

    yield_sigma_ln: float = Field(default=0.1, ge=0)
    blast_fraction_range: tuple[float, float] | None = (0.45, 0.55)
    reflection_factor_range: tuple[float, float] | None = (1.6, 2.0)
    thermal_fraction_range: tuple[float, float] | None = (0.30, 0.40)
    visibility_range_m: tuple[float, float] | None = (10_000.0, 30_000.0)
    fragility_median_sigma_ln: float = Field(default=0.2, ge=0)  # epistemic, per class
    population_sigma_ln: float = Field(default=0.1, ge=0)
    indoor_fraction_range: tuple[float, float] | None = (0.75, 0.95)
    # Model-form error on peak overpressure (Brode states ~10% accuracy)
    blast_model_sigma_ln: float = Field(default=0.1, ge=0)


class MonteCarloSpec(BaseModel):
    n_samples: int = Field(default=500, ge=1)
    seed: int = 42
    uncertainty: UncertaintySpec = UncertaintySpec()


class ExposureSpec(BaseModel):
    indoor_fraction: float = Field(default=0.85, ge=0, le=1)


class OutputSpec(BaseModel):
    dir: str = "outputs"


class Scenario(BaseModel):
    name: str
    source: SourceSpec
    atmosphere: AtmosphereSpec = AtmosphereSpec()
    environment: EnvironmentSpec = EnvironmentSpec()
    exposure: ExposureSpec = ExposureSpec()
    montecarlo: MonteCarloSpec = MonteCarloSpec()
    output: OutputSpec = OutputSpec()

    @classmethod
    def from_yaml(cls, path: str | Path) -> Scenario:
        with open(path, encoding="utf-8") as f:
            return cls.model_validate(yaml.safe_load(f))


def build_environment(sc: Scenario) -> tuple[Environment, dict]:
    """Construct the environment and load the fragility library for a scenario."""
    lib = load_fragility_library(sc.environment.fragility_library)
    names = tuple(lib.keys())
    unknown = set(sc.environment.default_class_mix) - set(names)
    if unknown:
        raise ValueError(f"default_class_mix has classes not in fragility library: {unknown}")
    mix = np.array([sc.environment.default_class_mix.get(n, 0.0) for n in names])
    grid = Grid(sc.environment.half_width_m, sc.environment.cell_m)
    if sc.environment.type == "synthetic":
        s = sc.environment.synthetic
        env = synthetic_city(
            grid,
            names,
            mix,
            s.total_population,
            s.scale_length_m,
            s.n_buildings,
            s.centre_offset_m,
            s.seed,
        )
    else:
        from effects.data.geo import real_environment

        r = sc.environment.real
        if r is None:
            raise ValueError("environment.type is 'real' but the 'real' block is missing")
        env = real_environment(r.lat, r.lon, grid, names, mix, r.population_raster, r.osm_buildings)
    return env, lib


def nominal_sources(sc: Scenario):
    """(blast, thermal) models at nominal scenario inputs."""
    from effects.sources import ThermalSource, make_blast

    s, atm = sc.source, sc.atmosphere
    blast = make_blast(
        s.blast_model,
        s.yield_kt,
        s.burst_height_m,
        s.blast_fraction,
        s.reflection_factor,
        atm.p0_pa,
    )
    thermal = ThermalSource(s.yield_kt, s.burst_height_m, s.thermal_fraction, atm.visibility_m)
    return blast, thermal
