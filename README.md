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
| `ParticleFilter` | SIR (bootstrap) particle filter — main contribution | done |
| `ExtendedKalmanFilter` | Linearised baseline (DEM slope as Jacobian) | done |
| `DeadReckoning` | INS only, no terrain — lower bar | done |

## Study regions

| Key | Area | Terrain |
|---|---|---|
| `hill_country` | Central highlands, Nuwara Eliya | rugged — informative |
| `kandy_rolling` | Around Kandy | moderate |
| `north_central_plains` | Near Anuradhapura | flat — uninformative |
| `synthetic` | Random sum-of-Gaussians hills | offline testing, no download |

## Repository layout

```
configs/            default.yaml (all parameters), regions.yaml (bounding boxes)
src/tan/
  dem.py            DEM container, vectorised bilinear height + slope, reprojection
  simulate.py       ground-truth flight + INS / altimeter simulation
  filters/          base interface, particle filter, resampling, EKF, dead reckoning
  experiment.py     build DEM + filters, single and Monte Carlo runs
  metrics.py        RMSE, NEES, convergence time
  plotting.py       report figures (trajectories, error, ESS, particle snapshots)
scripts/            download_dem.py, run_experiment.py
tests/              pytest suite — incl. PF vs exact Kalman filter on planar terrain
report/             report outline and figures
```

## Setup

Requires Python ≥ 3.10 (Ubuntu 24.04 ships 3.12; install `python3-venv` if `venv` is missing).

```bash
make setup                 # creates .venv and installs the package + dev tools
source .venv/bin/activate
make test                  # 35 tests, ~3 s
make run                   # synthetic terrain → results/<timestamp>_synthetic/
```

Real terrain needs a free [OpenTopography](https://opentopography.org) API key:

```bash
cp .env.example .env       # paste your key into OPENTOPO_API_KEY
make data                  # downloads + reprojects every region in configs/regions.yaml
python scripts/run_experiment.py --region hill_country --runs 50
```

Any parameter can be overridden from the command line:

```bash
python scripts/run_experiment.py --set particle_filter.n_particles=1000 init.position_sigma=2000
```

## Roadmap

- [x] Resampling: ESS, multinomial, systematic, stratified, residual (`filters/resampling.py`)
- [x] Particle filter: predict, update, estimate, roughening (`filters/particle_filter.py`)
- [x] EKF baseline: predict, update (`filters/ekf.py`)
- [ ] Download the three regions and sanity-check the hillshades
- [ ] Experiment 1 — PF vs EKF vs dead reckoning, increasing initial uncertainty
- [ ] Experiment 2 — hill country vs rolling vs flat terrain
- [ ] Experiment 3 — particle count vs RMSE vs runtime; resampling scheme comparison
- [ ] Experiment 4 — robustness: altimeter outliers (Student-t likelihood), canopy bias
- [ ] Experiment 5 — Monte Carlo consistency (NEES), particle-cloud figures
- [ ] Stretch — Rao-Blackwellised PF (Kalman filter per particle for the bias states)
- [ ] Report (`report/outline.md`; running results in `report/findings.md`)

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
