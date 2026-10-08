# Design

## Goal

A desktop tool that takes a scenario (source, location, environment) and produces damage-state
probabilities for buildings and casualty estimates per cell, with uncertainty, then renders them.
Models come from the published open literature. Accuracy is claimed only where validated.

## Data flow

```
scenario (YAML) -> source model -> field(x, y): overpressure, impulse, thermal fluence
                                        |
environment (population grid, buildings) +
                                        v
                    vulnerability (fragility curves, casualty model)
                                        |
                                        v
                    Monte Carlo wrapper -> distributions
                                        |
                                        v
                       viz (2D maps, later 3D)
```

## Build 1: analytic engine (implemented)

| Module | What it does | Reference |
|---|---|---|
| `sources/nuclear_fits.py` | **Default blast model.** Brode height-of-burst fit (Mach-stem included), Kinney-Graham tail below 2 psi; DNA 1 kt free-air standard for validation | Brode 1986; DNA; constants via MIT-licensed `glasstone` |
| `sources/blast.py` | Kinney-Graham free-air overpressure, impulse, duration; TNT-equivalent nuclear model (`blast_model: kinney_graham`) | Kinney & Graham 1985; Glasstone & Dolan 1977 |
| `sources/shock.py` | Rankine-Hugoniot shock speed and ground arrival time (for animation) | |
| `sources/thermal.py` | Point-source thermal fluence with exponential transmittance, L = visibility / 2 | Glasstone & Dolan 1977 ch. 7; transmittance fitted to Fig. 7.42 |
| `fragility/curves.py` | Lognormal damage-state curves per building class, loaded from YAML | HAZUS-style form |
| `fragility/default_fragility.yaml` | 5 building classes x 4 damage states, each tagged with a `basis` | Medians **derived** from Glasstone & Dolan Fig. 5.140 and Tables 5.139a/b; betas **assumed** (0.3) |
| `fragility/human.py` | Indoor casualties via damage state, outdoor via direct blast + burns | Glasstone & Dolan Tables 12.21, 12.38 and Fig. 12.65 (see file for which values are sourced, derived or assumed) |
| `data/environment.py` | Local metric grid centred on ground zero; per-cell building-class mix | |
| `data/synthetic.py` | Monocentric exponential-density city + sampled buildings (offline) | Clark 1951 |
| `data/geo.py` | WorldPop/GHSL GeoTIFF reprojection (sum-preserving), OSM buildings via osmnx, cached | |
| `scenario.py` | Pydantic-validated YAML schema | |
| `montecarlo/engine.py` | Samples inputs per realisation; Welford running mean/std per cell; per-building mean state probabilities | |
| `viz/plots2d.py` | Range curves, population + rings, fatality mean/std maps, building damage map, totals histograms | |
| `cli.py` | `calc` (radii table) and `run` (full scenario) | |

### Uncertainty sampled per realisation

Yield (lognormal), blast/thermal partition, reflection factor, visibility, a per-class lognormal
shift on fragility medians (epistemic), population scale, indoor fraction, and a lognormal
model-form factor on peak overpressure (`blast_model_sigma_ln`, default 0.1, from Brode's stated
~10% accuracy).

### Validation status

| Check | Status |
|---|---|
| Cube-root scaling, monotonicity, far-field decay, range-solver inversion | Passing (unit tests) |
| Thermal inverse-square and energy conservation | Passing |
| Monte Carlo mean exceedance vs closed-form lognormal convolution | Passing (`tests/validation`) |
| Monte Carlo convergence between independent seeds | Passing |
| Kinney-Graham surface burst vs Brode (1-20 psi, 1 kt-1 Mt) | Passing: within 10% at 5-20 psi, KG ~17% low at 1-2 psi (tolerance 20%) |
| Kinney-Graham free-air (blast fraction 0.5) vs DNA 1 kt standard, 100 m-5 km | Passing: within ~25% (tolerance 30%) |
| Brode optimum burst height for 5 psi, 15 kt vs ~2,530 ft from Glasstone HOB curves (cited by Wellerstein 2013) | Passing: 700 m / 2,300 ft, ~9% low (tolerance 15%) |
| Brode tail continuity and positivity | Passing |
| Blast ranges vs transcribed Glasstone figures | Optional: harness in `tests/validation/data/reference_points.csv` |
| Brode vs Glasstone Fig. 3.73 (1 kt surface-burst intercepts at 100-1000 psi; 100 kt worked example, 50 psi) | Passing: within 20% (`test_glasstone_blast_points.py`); the worked example is 7% low |
| Thermal model vs 9 points read from Glasstone Fig. 7.42 (3-50 cal/cm², 1.5 kt-9 Mt, 12-mile visibility) incl. the book's "about 7 miles" worked example | Passing within 15% (`test_thermal_fig_7_42.py`). The transmittance length was *fitted* to 22 points from this figure, so this checks the fit, not an independent source |
| Building fragility medians vs Glasstone: wood-frame reproduces the book's worked example (29,000 ft severe, 33,000 ft moderate at 1 Mt) within 15%; light damage 1 psi; class ordering | Passing (`test_fragility_anchors.py`). The nomogram was read by pixel measurement and converted with this repo's Brode model, so errors compound: expect ±25% in overpressure. Betas are assumed |
| Casualty parameters equal Tables 12.21 and 12.38 and the Fig. 12.65 lines they were read from | Passing (`test_fragility_anchors.py`). This confirms transcription only; see limitations 8-10 for how far the values can be trusted |

Example (Brode): 15 kt gives 5 psi at 1.15 km for a surface burst and 1.62 km at 600 m burst height.

### Known limitations

1. ~~No height-of-burst model~~ Fixed by the Brode fit (default). The `kinney_graham` option still
   lacks HOB effects; the CLI warns if it is used with burst height > 0. Brode assumes sea-level
   ambient and an ideal (non-dusty) surface; pressure is scaled linearly with p0 for other altitudes.
2. Flat ground; no terrain or building shielding (blast or thermal).
3. Thermal transmittance is a single fitted exponential (L = visibility / 2) that absorbs scattered-light buildup. It reproduces Fig. 7.42 (12-mile visibility, burst height 200 W^0.4 ft) to about 4% RMS in range but linear scaling with visibility, other burst heights and ranges beyond about 20 miles are untested.
4. Burn thresholds now rise with yield (Fig. 12.65 lines). Burn *lethality* is an assumption (the 50% third-degree line); the book gives no burn mortality curve, so burn deaths are an upper bound. Skin pigmentation, clothing and shielding are not modelled.
5. Overpressure-only fragility: valid for long-duration loading. Short-duration loads need P-I curves. **Drag-sensitive classes** (`reinforced_concrete`, `steel_frame`, `light_industrial`; Glasstone Types 6-12) are damaged by dynamic pressure, so their curves hold only near the reference yield (15 kt). At 1 Mt the same book nomogram gives 35-45% lower overpressure for these types. The CLI warns when the yield differs from the reference by more than 3x.
6. Fragility medians come from *reading a printed nomogram* (Fig. 5.140) and converting distance to overpressure with the Brode fit. Glasstone gives its own accuracy as ±20% in distance. Its "moderate" and "severe" levels are only 10-30% apart in overpressure, so this model's moderate, severe and collapse curves nearly coincide (the "severe" state is interpolated). Betas (0.3) are assumed. Glasstone Type 3 (multistory brick apartment) stands in for UK brick housing, and Type 5 for timber houses. UK construction is not distinguished.
7. Prompt radiation and fallout are not modelled.
8. **Indoor casualty rates come from Japan.** Table 12.21 (RC buildings, 0.3-0.75 mile from ground zero) includes deaths from early radiation and burns, and Glasstone says the numbers "cannot be used to estimate casualties from the degree of structural damage". They are applied here verbatim because they are the only primary tabulation, with "severe" and "collapse" both taking the "severe damage" row (88% killed). Light/moderate rows (8% and 14% killed) probably overstate casualties far from ground zero, and the RC-building rates are applied to all building classes.
9. **Direct blast uses incident, not effective, overpressure.** Table 12.38 is for effective pressure (incident plus an orientation-dependent dynamic-pressure term), so blast injury close in is underestimated. The injury median (17 psi) is derived and its beta assumed; lethality beta (0.18) is derived by reading the table's threshold and 100% values as the 1% and 99% points.
10. **Correction of an earlier version.** An earlier commit anchored house collapse at 5 psi from secondary sources (OTA 1979 and a Wikipedia summary). The primary text indicates about 3-3.5 psi for severe damage to wood-frame houses (and about 5 psi for brick), so that anchor was too high for timber houses. It has been replaced by the derivation above.
11. Reading error: the nomogram and the Fig. 7.42 / 12.65 values were read from scanned page images, so each carries a few percent of reading error on top of the book's own tolerances.

## Build 2: 3D rendering (implemented)

`viz/scene3d.py`, driven by `effects view <run_dir>`. Read-only view onto a run's `fields.npz`,
`buildings.csv` and `summary.json`; it does not change any results.

- Ground plane draped with population, fatalities, overpressure or thermal fluence (log scale).
- Buildings extruded as boxes, coloured by expected damage state E[DS].
- 1 / 5 / 20 psi rings with labels; burst point at true height.
- Shock-front dome and ring driven by the integrated arrival time; buildings stay grey until the
  front reaches them.
- Outputs: interactive window with a time slider, PNG screenshot, GIF/MP4 animation, and
  ParaView files (`buildings.vtp` with per-building state probabilities, `ground.vti` with every
  field).

Limitations: buildings are boxes (CSV holds centroid, area, height only); the shock dome is a
visual approximation, not a computed front; arrival time ignores the curved incident path near
ground zero for air bursts.

## Build 3: 2D compressible solver

- Finite-volume Euler equations, HLLC Riemann solver, MUSCL reconstruction.
- Plan-view mode with buildings as solid obstacles (reflection, street channelling).
- Axisymmetric (r, z) mode for validation against the Sedov-Taylor analytic solution, and to
  cross-check the Brode height-of-burst fit.
- Numba on CPU first (`pip install -e ".[solver]"`); GPU only if needed.

## Conventions

- SI units throughout. Convert only at I/O boundaries.
- Core functions are pure and vectorised over NumPy arrays.
- Every model function documents its reference and validity range.
- Randomness takes an explicit seed / `numpy.random.Generator` for reproducibility.
