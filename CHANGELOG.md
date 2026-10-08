# Changelog

## Unreleased

- **Housekeeping:** added MIT `LICENSE`; `.gitignore` now also covers `venv/`, `env/` and `.env`;
  stopped tracking `__pycache__`, `*.egg-info` and `outputs/` (all were already listed in `.gitignore`).
- **Fragility:** residential classes (`wood_frame`, `masonry`) now have a collapse median of 5 psi
  (34.5 kPa), anchored to published statements (see the YAML `reference` field). Their lower damage
  states were scaled with it. Other classes are unchanged. Every class now carries a `basis` tag.
  Expect more damage at 3-8 psi than in 0.2.0. Example renders in `docs/images/` predate this change.
- **Validation:** `tests/validation/test_fragility_anchors.py` guards the residential anchors.
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
