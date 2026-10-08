# Changelog

## Unreleased

- **Housekeeping:** added MIT `LICENSE`; `.gitignore` now also covers `venv/`, `env/` and `.env`;
  stopped tracking `__pycache__`, `*.egg-info` and `outputs/` (all were already listed in `.gitignore`).
- **Fragility (breaking change to results):** all five building classes now take their medians from
  Glasstone & Dolan (1977) Fig. 5.140 and Tables 5.139a/b (page references in the YAML `reference`
  field), replacing the placeholders. Wood-frame severe damage is now about 3.5 psi (was 40 kPa =
  5.8 psi collapse), brick about 5 psi, and the RC/steel/industrial frame classes are much weaker
  than the old placeholders (RC collapse 99 kPa, was 150 kPa). Betas are 0.3 (assumed). Corrects an
  earlier "5 psi" house-collapse anchor taken from secondary sources. Drag-sensitive classes carry a
  reference yield (15 kt) and the CLI warns outside 3x.
- **Casualties:** indoor rates from Table 12.21, direct-blast curves from Table 12.38, burns
  yield-dependent from Fig. 12.65 (`CasualtyModel.rates` now takes `yield_kt`). Fatalities for the
  15 kt demo scenario rise from about 18,500 to about 54,000. See `docs/DESIGN.md` limitations 8-10.
- **Thermal:** transmittance length is now visibility / 2 (was visibility), fitted to Glasstone
  Fig. 7.42. Thermal radii are smaller than in 0.2.0 (3 cal/cm² at 3.2 km for 15 kt, 20 km visibility).
- **Validation:** new tests against Glasstone Fig. 3.73 (blast), Fig. 7.42 (thermal), and the
  fragility/casualty derivations (`tests/validation/`).
- **Docs:** `docs/DESIGN.md` now records the indoor-casualty gap as a known limitation.

## 0.2.0 - Build 2

- **Blast model:** new default `brode` (Brode 1986 height-of-burst fit, Mach-stem included) with a
  Kinney-Graham tail below 2 psi. Air-burst ranges are no longer underestimated. The old model is
  still available as `blast_model: kinney_graham`.
- **Validation:** Kinney-Graham vs Brode (surface), vs DNA 1 kt free-air standard, and Brode's
  optimum burst height vs Glasstone (via Wellerstein 2013). See `docs/DESIGN.md`.
- **Uncertainty:** model-form factor on peak overpressure (`blast_model_sigma_ln`, default 0.1).
- **3D:** `effects view` with interactive time slider, screenshots, GIF/MP4 shock-front
  animation, ParaView export.
- `summary.json` now records the source and atmosphere blocks (needed by `effects view`).
- New scenario `scenarios/synthetic_city_airburst.yaml`.
- Example renders in `docs/images/` (make the animation with `effects view ... --animate`).

## 0.1.0 - Build 1

Analytic engine: source models, fragility, data layer, Monte Carlo, 2D plots, CLI.
