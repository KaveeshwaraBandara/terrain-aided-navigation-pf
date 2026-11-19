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

## 2026-11-19 — M2 complete: `make quick` runs end to end on the real regions

- All seven experiments run in `--quick` mode (3 runs, 150 s flights) and every analysis writes
  its figures/tables (under `results/<exp>_quick/report/`, never `report/`).
- Smoke-level sanity only (3 runs, not evidence): on the hill-country DEM the EKF already fails
  at σ₀ = 250 m in 2 of 3 runs while the PF converges, unlike on the synthetic map where the EKF
  held up to 1 km; on the plains no filter converges reliably even at σ₀ = 100 m. Corridor
  altitude gives radar ranges up to ≈ 0.7 km in the quick runs.
- Bug found and fixed by the quick run: bootstrap bounds can exceed the point estimate by a
  rounding error, which matplotlib rejects as negative error bars (clipped at 0 now).

---

## 2026-11-19 — M1: real terrain downloaded, regions chosen by the pre-registered rule

- OpenTopography key works. Survey: one COP90 request over Sri Lanka (33.9 MB). The rule picked
  (0.36° boxes, ≈ 39.8 km × 39.8 km after reprojection; 0 % no-data/sea in all three):

  | Region | Box (°N / °E) | 1 km roughness on COP90 (candidates) |
  |---|---|---|
  | hill_country | 6.74–7.10 / 80.72–81.08 | 65.2 m (max of 144, range 56.4–65.2) |
  | kandy_rolling | 7.22–7.58 / 80.34–80.70 | 37.8 m (min of 144, range 37.8–59.4) |
  | north_central_plains | 8.24–8.60 / 80.14–80.50 | 3.4 m (min of 144, range 3.4–5.9) |

  The "moderate" target (geometric mean of the other two) was 14.9 m; **no box around Kandy is
  that smooth**, so the least rough one was chosen. The Kandy region is intermediate, but nearer
  the rugged end — to be kept in mind when interpreting results.
- On the 30 m data (`report/tables/terrain_stats.csv`): roughness 64.5 / 37.7 / 3.5 m and
  Fisher information density 12.9 / 6.9 / 0.12 × 10⁻³ m⁻² (hill / Kandy / plains): one altimeter
  sample over the hill country carries ≈ 100× the position information of one over the plains.
  The synthetic reference has a similar 1 km roughness to the hill country (62.7 m) but less than
  half its information density (5.6 × 10⁻³ m⁻²): smooth Gaussian hills lack the real 30 m texture.
  This supports the earlier hypothesis that the real DEM is harder for the EKF's local
  linearisation and easier for a filter that uses the full likelihood — to be tested in E1.
- `data/manifest.json` records SHA-256 of raw, processed and elevation arrays for every region.

---

## 2026-11-19 — Reference check (A9)

Checked against DOI registration records (Crossref), publisher/author pages and the PDFs:
- Verified details now in `report/latex/references.bib`: Bergman, Ljung & Gustafsson, IEEE CSM
  19(3):33–40, 1999; Hostetler & Andreas, IEEE TAC 28(3):315–323, 1983 (middle initials not
  verifiable → initials only as on the DOI record); Tichavský et al., IEEE TSP 46(5):1386–1396,
  1998; Hol et al., NSSPW 2006, Cambridge UK, pp. 79–82; Bergman thesis = Linköping
  Dissertations No. 579; Wilson 1927 JASA 22(158):209–212; McNemar 1947 Psychometrika
  12(2):153–157; Bar-Shalom et al. 2001 (Wiley) — NEES cited by chapter (5) only, the exact
  section was not verifiable; Efron & Tibshirani — DOI record says 1993, the publisher page says
  "Copyright 1994"; cited as 1993 with the DOI.
- **Hol et al. (2006)** recommend systematic resampling ("favourable, both in terms of
  resampling quality and computational complexity"); lower variance than multinomial is argued
  for stratified and residual; for systematic they note a variance analysis is hard and cite an
  artificial example with increased variance; RMSE effects in their simulations "are not
  significant". E3 is compared against exactly these statements.
- **Gordon et al. (1993), §4.2**: roughening σ = K·E·N^(−1/d), E the range of the samples of
  that component, d the state dimension; K = 0.2 in their bearings-only example (N = 4000).
- **Bergman (1999)**: Ch. 4 "Cramér-Rao Bounds" (4.4 Posterior Bounds), §7.2 "Cramér-Rao
  Bound Evaluation" for terrain navigation.
- **Copernicus DEM handbook v5.0 (29 Nov 2022)**, Table 1: absolute vertical accuracy < 4 m,
  relative < 2 m (slope ≤ 20 %) / < 4 m (slope > 20 %), 90 % linear errors; a DSM including
  buildings, infrastructure and vegetation; water bodies flattened in the edited WorldDEM.
  Not stated in the handbook: that the ocean is set to 0 m — the no-data/sea check in
  `tan.data` treating ≤ 0 m as sea is a heuristic, not a documented property.

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
