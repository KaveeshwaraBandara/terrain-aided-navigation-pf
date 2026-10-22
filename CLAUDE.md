# CLAUDE.md

Context for AI coding assistants working in this repo.

## Project
University mini project (EN4594 Autonomous Systems): implement a state estimation filter
for an application. Chosen: **particle filter terrain-aided navigation** for a GPS-denied
UAV over real Copernicus GLO-30 terrain of Sri Lanka, with EKF and dead-reckoning baselines.
The report is graded on correctness, clarity, depth of analysis, filter suitability and creativity.

## Conventions
- State `x = [px, py, bx, by]` (UTM 44N metres; INS velocity bias m/s). True velocity = INS − bias.
- Every estimator implements `tan.filters.base.Estimator` (`initialize`, `predict`, `update`, `estimate`).
- Particle operations are vectorised NumPy — no per-particle Python loops.
- Weights live in log space (`ParticleFilter.log_weights`).
- All parameters come from `configs/default.yaml`; scripts accept `--set a.b=value` overrides.
- Randomness only through `np.random.Generator` passed in (reproducible Monte Carlo).
- Figures use `matplotlib.figure.Figure` (no pyplot); fixed colour per estimator in `plotting.COLORS`.

## Commands
- `make setup`, `make test`, `make lint`, `make format`, `make run`, `make data`

## Status
- Done: DEM loading/interpolation/reprojection, simulation, dead reckoning, metrics, plotting,
  experiment runner, DEM download script, tests, CI.
- TODO (student implements): resampling functions, ParticleFilter.predict/update/estimate,
  EKF predict/update. Tests for these skip until implemented.
