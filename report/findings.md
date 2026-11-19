# Findings log

Running record of results and observations, to draw on when writing the report.
Newest first. Every number says which terrain, how many runs and which settings produced it.

## Machine (all runtimes in the report refer to this)

| | |
|---|---|
| CPU | 13th Gen Intel Core i7-13620H — 10 cores (6 performance + 4 efficiency), 16 threads, max 4.9 GHz, 24 MiB L3 |
| RAM | 16 GB (15.3 GiB usable) |
| OS | Ubuntu 24.04.5 LTS, Linux 7.0.0 |
| Python / libraries | Python 3.12.3, NumPy 2.5.4, SciPy 1.18.1, Matplotlib 3.11.2, rasterio 1.5.2 |

Laptop, mains power. Per-step runtimes are measured single-process with BLAS threads = 1
(`scripts/benchmark.py`), never inside the parallel sweeps: on this hybrid CPU a worker's
per-flight time rises from 0.27 s alone to ~1.05 s when 15 workers share memory bandwidth.

---

## 2026-11-19 — Experiment design decisions, fixed before any real-terrain run

- **Definitions** (A5): converged = error < 50 m throughout at least the final 100 s;
  false lock = final error > 200 m with 2σ radius < 50 m. A run can be neither (late or
  incomplete acquisition, or an excursion > 50 m while crossing flat ground). On a 200 s smoke
  run on synthetic terrain, most PF runs counted as "not converged" ended with 5–12 m error:
  "not converged" does not mean "lost", and the report must say so.
- **E4** runs on the hill-country and Kandy regions only: a robustness test needs a nominal
  filter that works; the flat region is covered by E1/E2.
- **E4 remedies**: σ_z inflated ×2 (7.2 m) and ×4 (14.4 m), and the offset-state PF (prior std
  20 m on the offset). Map-error field std 2.4 m = absolute LE90 4 m / 1.645 (Gaussian
  assumption); correlation length 1 km is an assumption.
- **E6 selection rule** (dev seeds, seed root 1000): among window L ∈ {5, 10, 20} and
  per-window false-alarm probability p ∈ {0.01, 0.001}, choose the highest pooled convergence
  rate at σ₀ = 2000 m among settings whose false-alarm rate at σ₀ = 250 m (runs with ≥ 1
  alarm) is ≤ 5 %; ties → longer window. Re-spread fraction fixed at 0.5. The chosen setting is
  then evaluated once on the main seeds (M = 300).
- **Illustrative run for the visuals**: the lowest-index E1 hill-country run at σ₀ = 2000 m
  where PF (N = 20 000) converged with convergence time ≥ 60 s and the EKF did not.
- **Colours**: one validated colour per estimator everywhere; σ_z-inflated PFs reuse PF blue
  with other line styles (same estimator, other parameter). The EKF is omitted from the E4
  map-error *figure* (kept in its table) because its orange and the offset-state PF's red fail
  the colour-separation check (ΔE 7.1 < 15).

---

## 2026-11-19 — Work order, and a correction to plan amendment A2

- The OpenTopography key was not yet available, so M2 infrastructure and the M3 estimators
  that need no real terrain (PCRB, RBPF, divergence monitor) were built and unit-tested on
  synthetic terrain first. M1's download steps run as soon as the key is in `.env`.
- **A2 correction.** The review claimed that averaging the terrain slope over *different*
  random flights gives no valid bound. That was wrong for this design: flights are placed
  uniformly at random, so given the initial guess the true start is distributed exactly as the
  filters' prior N(guess, σ₀²), and the Tichavský recursion with the expectation over the
  Monte Carlo ensemble bounds the ensemble MSE. Both versions are computed: the ensemble bound
  (against E1/E5 ensemble RMSE) and the per-flight bound for fixed flight plans (a local bound
  for the tracking phase).
- Verified in tests: the PCRB recursion equals the Kalman covariance for a linear model and
  equals the EKF covariance on planar terrain; the derived true process noise Q matches the
  simulator's empirical noise covariance (within 6 %); the RBPF reproduces the exact Kalman
  filter (position and bias) on planar terrain.
- Note on the synthetic reference terrain: its *elevation range* is 1 970 m (overlapping
  hills add up), although `synthetic.relief` is 600 m per hill.

---

## 2026-11-19 — Pre-registered region-selection rule (written before the survey ran)

Goal: three ~40 km × 40 km boxes of contrasting terrain, chosen by terrain statistics only
(never by filter results), so the terrain comparison cannot be tuned to a narrative.

- Survey data: one COP90 request covering Sri Lanka (S 5.85, N 9.90, W 79.50, E 82.00).
- Candidates: 0.36° × 0.36° boxes with corners on a 0.02° grid that contain the class's anchor
  town at least 0.06° inside every edge. Anchors (approximate town centres):
  Nuwara Eliya 6.950 N 80.789 E, Kandy 7.291 N 80.634 E, Anuradhapura 8.311 N 80.404 E.
- Score: mean 1 km roughness (std of elevation in a ≈1 km moving window), over land cells.
- Rule: rugged = highest roughness around Nuwara Eliya; flat = lowest roughness around
  Anuradhapura; moderate = around Kandy, overlapping neither, roughness closest in log terms to
  the geometric mean of the other two. Every box must have < 1 % no-data/sea.
- Implemented in `scripts/survey_regions.py`; all candidates and scores go to
  `report/tables/region_selection.csv`.

Other decisions today:
- The key's presence is checked by the download script itself, which never prints it.
- The LaTeX report will be Overleaf-ready only (owner's choice, no local TeX installation).

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
