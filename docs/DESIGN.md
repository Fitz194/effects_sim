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
| `sources/thermal.py` | Point-source thermal fluence with exponential transmittance | Glasstone & Dolan 1977 ch. 7 (simplified) |
| `fragility/curves.py` | Lognormal damage-state curves per building class, loaded from YAML | HAZUS-style form |
| `fragility/default_fragility.yaml` | 5 building classes x 4 damage states, each tagged with a `basis` | Residential collapse medians **anchored** to published statements; everything else **illustrative placeholders** |
| `fragility/human.py` | Indoor casualties via damage state, outdoor via direct blast + burns | **Illustrative placeholders** |
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
| Residential collapse median vs OTA (1979) and Glasstone & Dolan summaries: collapse at about 5 psi, nearly all destroyed by about 8 psi | Passing (`tests/validation/test_fragility_anchors.py`). Anchors come from secondary summaries, beta is assumed, so this is a consistency check, not a fit to data |
| Other building classes, all betas, all casualty rates | **Not yet done**: placeholders (see the `basis` tag in the YAML) |

Example (Brode): 15 kt gives 5 psi at 1.15 km for a surface burst and 1.62 km at 600 m burst height.

### Known limitations

1. ~~No height-of-burst model~~ Fixed by the Brode fit (default). The `kinney_graham` option still
   lacks HOB effects; the CLI warns if it is used with burst height > 0. Brode assumes sea-level
   ambient and an ideal (non-dusty) surface; pressure is scaled linearly with p0 for other altitudes.
2. Flat ground; no terrain or building shielding (blast or thermal).
3. Thermal transmittance is a single exponential; no scattered-light buildup. Order-of-magnitude.
4. Burn thresholds are yield-independent (in reality they rise with yield).
5. Overpressure-only fragility: valid for long-duration loading. Short-duration loads need P-I curves.
6. Only the residential (`wood_frame`, `masonry`) collapse medians are anchored. All other fragility parameters, all betas and all casualty rates are placeholders. The anchors are secondary summaries of Glasstone & Dolan; the primary tables (ch. 5 damage summaries, ch. 12 blast injury) still need transcribing into `tests/validation/data/reference_points.csv` with page numbers.
7. Prompt radiation and fallout are not modelled.
8. **Indoor casualties look too low.** With houses collapsing at about 5 psi, the default model gives about 13% indoor fatality at 5 psi and plateaus at 25% at any overpressure (the collapse fatality rate). OTA (1979) uses civil-defence assumptions of roughly 50% mean lethality at 5-6 psi, which it calls relatively conservative. The two are not reconciled, and the OTA statements are ambiguous about standing versus lying occupants, so the rates were left alone rather than tuned to one secondary number. HAZUS indoor casualty rates were considered and not adopted: they are earthquake rates, give far lower fatalities, and are not valid for blast. Needs primary casualty data (Glasstone & Dolan ch. 12 and civil-defence studies).

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
