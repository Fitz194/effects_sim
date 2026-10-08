"""Command line entry point.

python -m effects.cli calc --yield-kt 15 --burst-height-m 600
python -m effects.cli run scenarios/synthetic_city.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from effects.sources import BLAST_MODELS, ThermalSource, make_blast
from effects.units import cal_cm2_to_j_m2, psi_to_pa

PSI_THRESHOLDS = (20, 10, 5, 3, 1)
CAL_THRESHOLDS = (10, 5, 3)
AIRBURST_WARNING = (
    "WARNING: kinney_graham model with burst height > 0. It has no Mach-stem / height-of-burst "
    "enhancement, so air-burst overpressure ranges are underestimated (roughly 1.5-2x in range "
    "near the optimum burst height). Use blast_model 'brode' for air bursts."
)


def blast_warnings(model: str, burst_height_m: float) -> list[str]:
    return [AIRBURST_WARNING] if model == "kinney_graham" and burst_height_m > 0 else []


def radii(blast, thermal: ThermalSource) -> dict:
    return {
        "overpressure_psi_km": {
            str(p): round(blast.range_for_overpressure(psi_to_pa(p)) / 1e3, 3)
            for p in PSI_THRESHOLDS
        },
        "thermal_cal_cm2_km": {
            str(q): round(thermal.range_for_fluence(cal_cm2_to_j_m2(q)) / 1e3, 3)
            for q in CAL_THRESHOLDS
        },
    }


def cmd_calc(a) -> None:
    blast = make_blast(
        a.blast_model, a.yield_kt, a.burst_height_m, a.blast_fraction, a.reflection_factor
    )
    therm = ThermalSource(a.yield_kt, a.burst_height_m, a.thermal_fraction, a.visibility_m)
    r = radii(blast, therm)
    print(
        f"Source: {a.yield_kt:g} kt, burst height {a.burst_height_m:g} m, "
        f"blast model {a.blast_model}"
    )
    print("\nPeak overpressure    ground range")
    for p, km in r["overpressure_psi_km"].items():
        print(f"  {p:>3} psi ({psi_to_pa(float(p)) / 1e3:6.1f} kPa)   {km:8.2f} km")
    print("\nThermal fluence      ground range")
    for q, km in r["thermal_cal_cm2_km"].items():
        print(f"  {q:>3} cal/cm²           {km:8.2f} km")
    for w in blast_warnings(a.blast_model, a.burst_height_m):
        print("\n" + w)
    if a.plot:
        from effects.viz.plots2d import overpressure_vs_range

        max_r = max(blast.range_for_overpressure(psi_to_pa(0.5)), 1_000.0)
        overpressure_vs_range(blast, therm, max_r, a.plot)
        print(f"\nPlot written to {a.plot}")


def cmd_run(a) -> None:
    from effects.montecarlo import run
    from effects.scenario import Scenario, build_environment, nominal_sources
    from effects.viz.plots2d import write_all

    sc = Scenario.from_yaml(a.scenario)
    if a.samples:
        sc.montecarlo.n_samples = a.samples
    out = Path(a.out or sc.output.dir)
    print(f"Scenario '{sc.name}': building environment ({sc.environment.type})...")
    t0 = time.perf_counter()
    env, lib = build_environment(sc)
    print(
        f"  grid {env.grid.n}x{env.grid.n} @ {env.grid.cell_m:g} m, "
        f"population {env.population.sum():,.0f}, buildings {len(env.buildings):,}"
    )
    print(f"Running {sc.montecarlo.n_samples} Monte Carlo samples...")
    res = run(sc, env, lib, progress=True)

    out.mkdir(parents=True, exist_ok=True)
    s = res.summary()
    src = sc.source
    s["source"] = src.model_dump()
    s["atmosphere"] = sc.atmosphere.model_dump()
    s["blast_model"] = src.blast_model
    s["radii_nominal"] = radii(*nominal_sources(sc))
    s["warnings"] = blast_warnings(src.blast_model, src.burst_height_m)
    s["runtime_s"] = round(time.perf_counter() - t0, 2)
    (out / "summary.json").write_text(json.dumps(s, indent=2))

    b = env.buildings.copy()
    for i, name in enumerate(("none",) + res.states):
        b[f"p_{name}"] = res.building_state_probs[:, i]
    b.to_csv(out / "buildings.csv", index=False)
    np.savez_compressed(
        out / "fields.npz",
        centres_m=env.grid.centres,
        population=env.population,
        overpressure_pa=res.overpressure_nominal,
        fluence_j_m2=res.fluence_nominal,
        fatal_mean=res.fatal_mean,
        fatal_std=res.fatal_std,
        injured_mean=res.injured_mean,
    )
    plots = write_all(res, out)

    f, i = s["fatalities"], s["injuries"]
    print(f"\nFatalities  mean {f['mean']:,.0f}   90% interval {f['p05']:,.0f} - {f['p95']:,.0f}")
    print(f"Injuries    mean {i['mean']:,.0f}   90% interval {i['p05']:,.0f} - {i['p95']:,.0f}")
    for w in s["warnings"]:
        print("\n" + w)
    print(f"\nWrote summary.json, buildings.csv, fields.npz and {len(plots)} plots to {out}/")


def cmd_view(a) -> None:
    from effects.viz import scene3d

    run = scene3d.load_run(a.run_dir)
    did = False
    if a.screenshot:
        scene3d.render_screenshot(run, a.screenshot, a.layer, a.z_scale)
        print(f"Screenshot written to {a.screenshot}")
        did = True
    if a.animate:
        scene3d.render_animation(run, a.animate, a.layer, a.z_scale, frames=a.frames)
        print(f"Animation written to {a.animate}")
        did = True
    if a.export_vtk:
        for p in scene3d.export_vtk(run, a.export_vtk):
            print(f"Wrote {p}")
        did = True
    if not did:
        print("Opening 3D window (drag to orbit, scroll to zoom, 'r' resets the camera)...")
        scene3d.interactive(run, a.layer, a.z_scale)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="effects")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("calc", help="single-source radii calculator")
    c.add_argument("--yield-kt", type=float, required=True)
    c.add_argument("--burst-height-m", type=float, default=0.0)
    c.add_argument("--blast-model", choices=BLAST_MODELS, default="brode")
    c.add_argument("--blast-fraction", type=float, default=0.5)
    c.add_argument("--reflection-factor", type=float, default=1.8)
    c.add_argument("--thermal-fraction", type=float, default=0.35)
    c.add_argument("--visibility-m", type=float, default=20_000.0)
    c.add_argument("--plot", type=str, default=None, help="write range-curve PNG to this path")
    c.set_defaults(func=cmd_calc)

    r = sub.add_parser("run", help="run a scenario YAML through the Monte Carlo engine")
    r.add_argument("scenario")
    r.add_argument("--samples", type=int, default=None, help="override n_samples")
    r.add_argument("--out", type=str, default=None, help="override output directory")
    r.set_defaults(func=cmd_run)

    v = sub.add_parser("view", help="3D view of a completed run (needs the [viz] extra)")
    v.add_argument("run_dir", help="output directory written by `effects run`")
    v.add_argument(
        "--layer",
        choices=("population", "fatalities", "overpressure", "fluence"),
        default="population",
        help="field draped on the ground",
    )
    v.add_argument("--z-scale", type=float, default=1.5, help="vertical exaggeration of buildings")
    v.add_argument("--screenshot", help="write a PNG instead of opening a window")
    v.add_argument("--animate", help="write a shock-front animation (.gif or .mp4)")
    v.add_argument("--frames", type=int, default=60)
    v.add_argument("--export-vtk", help="write buildings.vtp + ground.vti for ParaView")
    v.set_defaults(func=cmd_view)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
