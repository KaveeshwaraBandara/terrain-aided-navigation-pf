# Findings log

Running record of results and observations, to draw on when writing the report.
Newest first. Every number says which terrain, how many runs and which settings produced it.

---

## 2026-11-04 — Filters implemented and validated (synthetic terrain)

Setup unless stated otherwise: `configs/default.yaml` — synthetic terrain (60 Gaussian hills,
600 m relief, 20 km × 20 km, 30 m grid), 300 s flight at 40 m/s, σ_z = 3.6 m,
PF with 5000 particles, systematic resampling at N_eff < 0.5 N, no roughening.

### Correctness: PF reproduces the exact Kalman filter on planar terrain
On a tilted plane h(p) = 0.05 px + 0.03 py + 200 the measurement is linear, so the EKF is
the exact Kalman filter (exact posterior). PF with 20 000 particles vs KF, initial error
σ₀ = 100 m, 5 seeds, worst case over the whole 300 s flight:

| | max \|PF mean − KF mean\| / KF σ | PF/KF position-covariance trace ratio |
|---|---|---|
| position | 0.02 – 0.09 | 0.91 – 1.25 |
| bias | 0.04 – 0.15 | — |

Kept as a regression test: `tests/test_filters.py::test_pf_matches_kalman_filter_on_planar_terrain`.
Good evidence for the "correctness of implementation" criterion.

### Baseline comparison, σ₀ = 500 m, 20 Monte Carlo runs (`make run` with `--runs 20`)

| Estimator | RMSE over flight | Final error (mean ± std) | Converged (stays < 50 m) | Mean convergence time |
|---|---|---|---|---|
| Dead reckoning | 692 m | 604 ± 305 m | 0 % | — |
| Particle filter | 44 m | 7.7 ± 5.1 m | 100 % | 41 s |
| EKF | 87 m | 5.3 ± 3.1 m | 100 % | 49 s |

The median PF error falls to ~15 m within ~15 s; the EKF takes ~50 s. After both converge
they track at ~5–20 m. The PF's lower RMSE comes from the faster acquisition, not better tracking.

### Initial-uncertainty sweep (ad-hoc check, 12 runs per row; Experiment 1 will redo this properly)

| σ₀ | PF 5000: converged / median final | EKF: converged / median final | PF 20 000: converged / median final |
|---|---|---|---|
| 250 m | 100 % / 3.3 m | 100 % / 3.5 m | 100 % / 2.8 m |
| 500 m | 100 % / 5.5 m | 100 % / 3.8 m | 100 % / 3.5 m |
| 1000 m | 100 % / 5.7 m | 100 % / 6.1 m | 100 % / 6.4 m |
| 2000 m | 75 % / 16.0 m | 50 % / 340 m | 100 % / 8.3 m |

Takeaways:
- Up to σ₀ = 1 km the EKF converges as reliably as the PF on this terrain, only slower.
  The synthetic hills are smooth (widths ~200 m – 2.4 km), which flatters linearisation.
  **Hypothesis to test on real terrain:** the rougher 30 m DEM will widen the gap.
- At σ₀ = 2 km the EKF fails half the time; the PF needs enough particles (20 000 → 100 %).
- Once converged, the EKF with a locally smooth surface is about as accurate as the PF; the
  5000-particle PF is slightly noisier (Monte Carlo error), 20 000 particles closes the gap.

### Failure mode: false lock (σ₀ = 2 km, 5000 particles, 3 of 12 runs failed)
- Failed runs had the largest initial guess errors (2.7, 3.1, 3.9 km). Few particles started near
  the truth, and the cloud locked onto a different patch of terrain whose profile matched.
- Signature: final error 3–8 km while the cloud std was only 0.2–3 m (confidently wrong);
  resampled on ~270 of 300 steps; minimum N_eff ≈ 1 particle.
- Healthy runs: ~40–55 resamples, final cloud std ~10–12 m, error ~1–18 m.
- **Discussion / creativity angle:** this signature is observable without ground truth
  (persistent N_eff collapse + resampling every step + measurements that never fit). A divergence
  monitor that re-spreads or re-initialises the cloud is a credible extension.

### Roughening check (σ₀ = 2 km, 5000 particles, 12 runs)
| K | 0 | 0.1 | 0.2 |
|---|---|---|---|
| converged | 75 % | 83 % | 58 % |

Not a reliable fix on its own at this sample size; more particles worked. Revisit in Experiment 3.

### Modelling decision: off-map particles get zero weight
Particles outside the DEM used to read the clamped edge height (fake flat terrain). They now get
zero likelihood, since the map covers the mission area by design. No measurable effect in the
runs above (those particles already had negligible weight), but it is the correct model.
Test: `test_off_map_particles_get_zero_weight`.
