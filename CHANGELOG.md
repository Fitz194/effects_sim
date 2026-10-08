# Changelog

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
