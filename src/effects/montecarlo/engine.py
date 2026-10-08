"""Monte Carlo consequence engine.

Each realisation samples the uncertain inputs once (yield, partitions, reflection, visibility,
per-class fragility median shift, population scale, indoor fraction, blast model-form factor),
then evaluates the full
deterministic chain on the grid and on every building. Per-cell results are accumulated as running
mean/variance (Welford), so memory does not grow with n_samples. Per-sample totals are kept for
distributions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from effects.data import Environment
from effects.fragility import CasualtyModel, FragilitySet
from effects.scenario import Scenario
from effects.sources import ThermalSource, make_blast


@dataclass
class Results:
    scenario: Scenario
    env: Environment
    states: tuple[str, ...]
    # per-sample totals
    fatalities: np.ndarray
    injuries: np.ndarray
    sampled_inputs: dict[str, np.ndarray]
    # per-cell maps
    overpressure_nominal: np.ndarray  # Pa, nominal inputs
    fluence_nominal: np.ndarray  # J/m^2, nominal inputs
    fatal_mean: np.ndarray  # people per cell
    fatal_std: np.ndarray
    injured_mean: np.ndarray
    # per-building: mean state probabilities (n_buildings, n_states + 1), incl. 'none'
    building_state_probs: np.ndarray

    def summary(self) -> dict:
        def stats(a):
            return {
                "mean": float(a.mean()),
                "p05": float(np.percentile(a, 5)),
                "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)),
            }

        bsp = self.building_state_probs
        expected_counts = {s: float(bsp[:, i].sum()) for i, s in enumerate(("none",) + self.states)}
        return {
            "name": self.scenario.name,
            "n_samples": int(len(self.fatalities)),
            "population_in_grid": float(self.env.population.sum()),
            "n_buildings": int(len(self.env.buildings)),
            "fatalities": stats(self.fatalities),
            "injuries": stats(self.injuries),
            "expected_buildings_by_state": expected_counts,
        }


def _uniform(rng, rng_spec, nominal):
    return rng.uniform(*rng_spec) if rng_spec else nominal


def run(
    sc: Scenario,
    env: Environment,
    library: dict[str, FragilitySet],
    casualty: CasualtyModel | None = None,
    progress: bool = False,
) -> Results:
    casualty = casualty or CasualtyModel()
    mc, unc, src = sc.montecarlo, sc.montecarlo.uncertainty, sc.source
    rng = np.random.default_rng(mc.seed)

    sets = [library[c] for c in env.class_names]
    states = sets[0].states
    ns = len(states) + 1
    k = len(sets)

    r_cells = env.grid.ground_range()
    mix = env.class_mix()  # (n, n, k)
    b = env.buildings
    r_bld = np.hypot(b["x"].to_numpy(), b["y"].to_numpy()) if len(b) else np.zeros(0)
    b_cls = (
        np.array([env.class_names.index(c) for c in b["cls"]], dtype=int)
        if len(b)
        else np.zeros(0, dtype=int)
    )

    def sources(yield_kt, bf, rf, tf, vis):
        blast = make_blast(
            src.blast_model, yield_kt, src.burst_height_m, bf, rf, sc.atmosphere.p0_pa
        )
        therm = ThermalSource(yield_kt, src.burst_height_m, tf, vis)
        return blast, therm

    nb, nt = sources(
        src.yield_kt,
        src.blast_fraction,
        src.reflection_factor,
        src.thermal_fraction,
        sc.atmosphere.visibility_m,
    )
    p_nom, q_nom = nb.overpressure(r_cells), nt.fluence(r_cells)

    n = mc.n_samples
    fat_tot, inj_tot = np.zeros(n), np.zeros(n)
    inputs = {
        key: np.zeros(n)
        for key in (
            "yield_kt",
            "blast_fraction",
            "reflection_factor",
            "thermal_fraction",
            "visibility_m",
            "population_scale",
            "indoor_fraction",
            "blast_model_factor",
        )
    }
    mean_f = np.zeros_like(r_cells)
    m2_f = np.zeros_like(r_cells)
    mean_i = np.zeros_like(r_cells)
    bsp_sum = np.zeros((len(b), ns))

    for s in range(n):
        y = src.yield_kt * np.exp(unc.yield_sigma_ln * rng.standard_normal())
        bf = _uniform(rng, unc.blast_fraction_range, src.blast_fraction)
        rf = _uniform(rng, unc.reflection_factor_range, src.reflection_factor)
        tf = _uniform(rng, unc.thermal_fraction_range, src.thermal_fraction)
        vis = _uniform(rng, unc.visibility_range_m, sc.atmosphere.visibility_m)
        pscale = np.exp(unc.population_sigma_ln * rng.standard_normal())
        fin = _uniform(rng, unc.indoor_fraction_range, sc.exposure.indoor_fraction)
        med_scale = np.exp(unc.fragility_median_sigma_ln * rng.standard_normal(k))
        pfac = np.exp(unc.blast_model_sigma_ln * rng.standard_normal())
        for key, val in zip(inputs, (y, bf, rf, tf, vis, pscale, fin, pfac), strict=True):
            inputs[key][s] = val

        blast, therm = sources(y, bf, rf, tf, vis)
        p, q = blast.overpressure(r_cells) * pfac, therm.fluence(r_cells)

        cell_sp = np.zeros(r_cells.shape + (ns,))
        for c, fs in enumerate(sets):
            cell_sp += mix[..., c : c + 1] * fs.state_probs(p, med_scale[c])
        pf, pi = casualty.rates(cell_sp, p, q, fin, yield_kt=y)
        pop = env.population * pscale
        fat, inj = pf * pop, pi * pop

        fat_tot[s], inj_tot[s] = fat.sum(), inj.sum()
        delta = fat - mean_f
        mean_f += delta / (s + 1)
        m2_f += delta * (fat - mean_f)
        mean_i += (inj - mean_i) / (s + 1)

        if len(b):
            pb = blast.overpressure(r_bld) * pfac
            for c, fs in enumerate(sets):
                sel = b_cls == c
                if sel.any():
                    bsp_sum[sel] += fs.state_probs(pb[sel], med_scale[c])

        if progress and (s + 1) % max(1, n // 10) == 0:
            print(f"  sample {s + 1}/{n}")

    std_f = np.sqrt(m2_f / max(n - 1, 1))
    return Results(
        scenario=sc,
        env=env,
        states=states,
        fatalities=fat_tot,
        injuries=inj_tot,
        sampled_inputs=inputs,
        overpressure_nominal=p_nom,
        fluence_nominal=q_nom,
        fatal_mean=mean_f,
        fatal_std=std_f,
        injured_mean=mean_i,
        building_state_probs=bsp_sum / n,
    )
