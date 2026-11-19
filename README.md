# Terrain-Aided Navigation with a Particle Filter

[![CI](https://github.com/KaveeshwaraBandara/terrain-aided-navigation-pf/actions/workflows/ci.yml/badge.svg)](https://github.com/KaveeshwaraBandara/terrain-aided-navigation-pf/actions/workflows/ci.yml)

Particle-filter terrain-aided navigation (TAN) for a GPS-denied UAV, flown in simulation
over **real Copernicus GLO-30 terrain of Sri Lanka**, with EKF and INS dead-reckoning
baselines. State estimation mini project — EN4594 Autonomous Systems.

## The problem

When GNSS is lost, a UAV's inertial navigation drifts without bound. It still carries a
barometric altimeter (altitude above sea level) and a radar altimeter (height above the
ground). Their difference is the terrain height directly below the aircraft; comparing a
sequence of those measurements with a stored elevation map pins down where it is.

The terrain likelihood is nonlinear and often **multimodal** — many places can share the
same height — which is why a particle filter is the natural estimator and a linearised
EKF is not.

## Model

State `x = [px, py, bx, by]` — position (m, UTM 44N) and INS velocity bias (m/s).

| | Equation |
|---|---|
| Motion | `p_k = p_{k-1} + dt (u_k − b_{k-1}) + w_p`, `b_k = b_{k-1} + w_b` (u = INS velocity) |
| Measurement | `z_k = baro_k − radar_k = h(p_k) + v_k`, h = bilinear DEM height |

| Estimator | Role | Status |
|---|---|---|
| `ParticleFilter` | SIR (bootstrap) particle filter — main contribution; options: Student-t likelihood, measurement-offset state, divergence monitor | done |
| `RaoBlackwellizedPF` | Marginalised PF: particles carry position, a Kalman filter per particle carries the bias | done |
| `ExtendedKalmanFilter` | Linearised baseline (DEM slope as Jacobian) | done |
| `DeadReckoning` | INS only, no terrain — lower bar | done |

## Study regions

Three ≈ 40 km × 40 km boxes of Copernicus GLO-30, chosen by a pre-registered roughness rule
(`scripts/survey_regions.py`, see `data/README.md`):

| Key | Area | 1 km roughness | Fisher density (10⁻³ m⁻²) |
|---|---|---|---|
| `hill_country` | Central highlands, Nuwara Eliya | 64.5 m | 12.9 |
| `kandy_rolling` | Mid-country around Kandy | 37.7 m | 6.9 |
| `north_central_plains` | Dry-zone plains, Anuradhapura | 3.5 m | 0.12 |
| `synthetic` | Random sum-of-Gaussians hills | 62.7 m | 5.6 |

(From `report/tables/terrain_stats.csv`.)

## Repository layout

```
configs/
  default.yaml      all model / filter parameters
  regions.yaml      the three study boxes (written by scripts/survey_regions.py)
  experiments/      e1..e7 (+ smoke) experiment definitions, _common.yaml shared settings
src/tan/
  data.py           OpenTopography download (key never printed), reprojection, crop, manifest
  dem.py            DEM container, vectorised bilinear height + slope, reprojection
  terrain.py        roughness, slope, Fisher information (per region and along a flight)
  simulate.py       the world: random flights, INS, altimeters, map error; common random numbers
  filters/          particle filter, RBPF, EKF, dead reckoning, resampling schemes
  pcrb.py           posterior Cramér–Rao bound with the true noise model
  metrics.py        convergence, false lock, RMSE, ANEES, Wilson / bootstrap, McNemar
  runner.py         parallel, resumable Monte Carlo runner
  analysis/         saved results → report figures, tables and LaTeX macros
scripts/            survey_regions, download_dem, terrain_stats, sweep, make_figures,
                    benchmark, validation, extract_listings, run_experiment (quick look)
tests/              pytest suite — incl. PF / RBPF vs exact Kalman filter on planar terrain
data/               manifest.json (checksums); rasters are downloaded, not committed
report/             latex/ (report), figures/, tables/, generated/ (macros), findings.md,
                    WALKTHROUGH.md
```

## Setup and full reproduction

Requires Python ≥ 3.10 (Ubuntu 24.04 ships 3.12; install `python3-venv` if `venv` is missing).

```bash
make setup                 # creates .venv and installs the package + dev tools
make test                  # unit tests (~5 s)
make smoke                 # whole pipeline on synthetic terrain, no download needed
```

Real terrain needs a free [OpenTopography](https://opentopography.org) API key:

```bash
cp .env.example .env       # paste your key into OPENTOPO_API_KEY (never commit .env)
make data                  # 4 API calls: island survey + three regions → data/processed/
make benchmark             # runtime per filter step (idle machine)
make experiments           # E1–E7, resumable, ~1 h on a 10-core laptop
make figures               # every figure / table / LaTeX number from saved results
make report                # PDF if latexmk exists; otherwise upload report/ to Overleaf
```

`make quick` runs every experiment with 3 runs and short flights (smoke test). One experiment:
`python scripts/sweep.py --experiment e3 [--quick] [--jobs 10]` then
`python scripts/make_figures.py --experiment e3`. Any parameter of the quick-look script can be
overridden: `python scripts/run_experiment.py --set particle_filter.n_particles=1000`.

**LaTeX report:** `report/latex/main.tex` reads figures, tables and numbers from
`report/{figures,tables,generated}/`. Without a local TeX installation, upload the whole
`report/` folder to Overleaf and compile `latex/main.tex` (pdfLaTeX + BibTeX, standard packages).

## Results

![Convergence and false lock vs initial uncertainty](report/figures/e1_rates.png)

**E1 — hill country, 300 random flights per cell:** converged runs (false-lock runs in brackets)
against the initial position uncertainty σ₀ (`report/tables/e1_convergence.csv`):

| σ₀ | EKF | PF, N = 5 000 | PF, N = 20 000 |
|---|---|---|---|
| 100 m | 85 % (15 %) | 100 % (0 %) | 100 % (0 %) |
| 500 m | 28 % (70 %) | 95 % (3 %) | 98 % (1 %) |
| 1000 m | 14 % (84 %) | 72 % (25 %) | 92 % (6 %) |
| 2000 m | 6 % (93 %) | 40 % (54 %) | 70 % (29 %) |

- On real 30 m terrain the EKF's linearisation fails earlier than on smooth synthetic hills
  (σ₀ = 500 m: 28 % vs 68 % convergence under the same protocol), and this ungated EKF then ends
  worse than dead reckoning.
- Across terrain types rougher terrain lowers the PF's error (pooled Spearman −0.63) and raises the
  EKF's (+0.29); between flights within one region the relation is weak.
- False lock is the PF's failure mode; at σ₀ = 2 km a divergence monitor (no ground truth) plus
  roughening raise convergence from 40 % to 96 % with 5 000 particles
  (`report/tables/e6_test.csv`).
- A Student-t likelihood keeps the PF robust to radar outliers; once locked on, the
  Rao-Blackwellised PF estimates the INS bias much better than the plain PF; in the tracking phase
  PF 20 000 stays ≈ 2× above the posterior Cramér–Rao bound, almost entirely because of its
  inflated process noise.

Full report: `report/latex/` (Overleaf-ready); running lab notebook: `report/findings.md`;
guide to the code, figures and likely questions: `report/WALKTHROUGH.md`.

## Roadmap

- [x] Resampling: ESS, multinomial, systematic, stratified, residual (`filters/resampling.py`)
- [x] Particle filter: predict, update, estimate, roughening (`filters/particle_filter.py`)
- [x] EKF baseline: predict, update (`filters/ekf.py`)
- [x] Real terrain: three regions chosen by a pre-registered rule, downloaded with checksums
- [x] Experiment 1 — PF vs EKF vs dead reckoning, increasing initial uncertainty
- [x] Experiment 2 — terrain informativeness per flight
- [x] Experiment 3 — particle count, resampling scheme, ESS threshold, roughening
- [x] Experiment 4 — robustness: altimeter outliers (Student-t), map error / offset, INS grade
- [x] Experiment 5 — consistency (ANEES) and the posterior Cramér–Rao bound
- [x] Experiment 6 — false-lock detection and recovery
- [x] Experiment 7 — Rao-Blackwellised PF
- [x] Report (`report/latex/`; running results in `report/findings.md`)

## Data attribution

Elevation data: produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus
Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA;
all rights reserved. Accessed through OpenTopography.

## References

- F. Gustafsson et al., "Particle filters for positioning, navigation, and tracking,"
  *IEEE Trans. Signal Processing*, vol. 50, no. 2, 2002.
- F. Gustafsson, "Particle filter theory and practice with positioning applications,"
  *IEEE Aerospace and Electronic Systems Magazine*, 2010.
- N. Bergman, L. Ljung, F. Gustafsson, "Terrain navigation using Bayesian statistics,"
  *IEEE Control Systems Magazine*, 1999.
- J. D. Hol, T. B. Schön, F. Gustafsson, "On resampling algorithms for particle filters,"
  *IEEE Nonlinear Statistical Signal Processing Workshop*, 2006.

## License

MIT — see [LICENSE](LICENSE).
