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

## 2026-11-19 — Rubric self-check (final entry)

Against the five grading criteria of the brief, in order:

1. **Correctness of the implementation.** Strong evidence: PF and RBPF reproduce the exact Kalman
   filter on planar terrain (RBPF ≤ 0.19 σ; PF ≤ 0.17 σ in four of five seeds, the fifth explained
   by bias impoverishment and gone when the bias is rejuvenated); the PCRB recursion equals the
   Kalman covariance to 3·10⁻¹¹; the true-noise model matches the simulator's empirical noise; the
   unconditional RMSE never undercuts the ensemble bound; 83 unit tests (resampling properties,
   likelihood normalisation, monitor behaviour, metric closed forms, common random numbers,
   resumption); the independent reviewer found every equation consistent with the code. Weak spots:
   the EKF baseline has no innovation gating, so the *size* of its failures is baseline-specific.
2. **Clarity and coherence of the report.** Structure follows the brief (Introduction,
   Methodology, Implementation, Results, Discussion, Conclusion, appendices); every number is a
   generated macro, every table and figure is generated from saved results, and a static checker
   replaces the missing local LaTeX build (0 problems). Risk: the PDF has not been compiled here
   (Overleaf only), and its length is estimated, not measured.
3. **Depth of understanding.** Paired tests and confidence intervals throughout; the PCRB is tied
   to the results (gap explained quantitatively by the mismatched-gain analysis); selection effects
   (converged-only RMSE, between- vs within-region correlation) and the failure of Gaussian
   consistency measures for multimodal posteriors are discussed; claims that did not survive the
   independent review were corrected, and the corrections are recorded above.
4. **Effectiveness of the filter for the application.** Over the hills PF 20 000 converges in 98 %
   (σ₀ = 500 m) and 92 % (1 km) of runs and tracks at ≈ 7–8 m; with roughening and the monitor,
   5 000 particles reach 96 % at 2 km; 0.83 ms per step (5 000 particles) is three orders of
   magnitude inside real time. Limits stated honestly: flat plains (≈ 21 m tracking, ≈ 50–65 %
   convergence by the 50 m criterion), constant height offsets ≥ 10 m, simulated sensors.
5. **Creativity and innovation.** Pre-registered, statistics-based region selection; an online
   divergence monitor (NIS window test, mixture re-spread from the dead-reckoning belief) tuned on
   separate development seeds; Student-t likelihood; offset state; Rao-Blackwellised filter; the
   mismatched-gain analysis separating tuning from approximation error; common random numbers
   across all experiments; a static report checker in place of a TeX build.

Open items for the owner: compile on Overleaf and check layout/length; the E6 selection-rule
extension and the "Kandy is not moderate" outcome are disclosed in the report.

---

## 2026-11-19 — M5: independent review, and what changed because of it

A fresh reviewer (no prior context, read-only) checked the report against the code and
`results/*/summary.csv`. Every major finding was re-checked here before acting; all were confirmed.

**Corrections of earlier claims (this log included):**
1. **Synthetic vs real EKF.** "On synthetic terrain the EKF held up to σ₀ = 1 km" (2026-11-04
   entry, 12 runs, fixed flight; repeated in the E1 entry and the report) is **wrong** for the
   matched protocol: the 300-run synthetic E1 (`results/e1@synth`, now part of `make experiments`)
   gives EKF convergence 91/86/68/39/25/16 % for σ₀ = 100…2000 m. The direction holds (500 m: 68 %
   synthetic vs 28 % real hills) but "held up to 1 km" and "far earlier" were overstated.
2. **E2 is mostly between regions.** The pooled ρ (PF −0.63, EKF +0.29) compares terrain types;
   within regions the PF's per-flight ρ is −0.16 (p = 0.02, hill), −0.26 (p < 0.01, Kandy), −0.05
   (p = 0.47, plains) and the EKF has no consistent relation. "Per flight … raises the EKF's error"
   was removed.
3. **PCRB gradient.** The bound used interpolated central-difference slopes (the EKF's Jacobian),
   but the simulated altimeter measures the bilinear interpolant, whose exact gradient carries more
   information (×1.09 hill, ×1.12 Kandy, ×1.58 plains). Fixed (`DEM.bilinear_gradient`, used by the
   PCRB, the mismatched-gain analysis and the Fisher statistics; tests added). Consequences: the
   ensemble check is stronger (all-run RMSE ≥ 1.29 × bound, was 1.10); per flight PF 20 000 is
   2.05× (hill) the bound and 1.10× what its tuning allows (was 1.93× and 1.05×); Fisher densities
   13.5 / 7.4 / 0.16 × 10⁻³ m⁻².
4. **Per-flight bound is not strict.** The caption claim "the unconditional RMSE never [falls below
   it]" was unverified; it is false at single steps (min 0.87, plains and one Kandy plan). For a
   fixed path with a random guess the per-flight bound is a local approximation — stated now.
5. **Validation.** PF 20 000 vs exact KF reaches 0.57 σ in one of five seeds (others ≤ 0.17 σ),
   above the unit test's 0.25 tolerance (different seed). Diagnosed: the position deviation follows
   the bias deviation (corr 0.68 in that seed) and all seeds stay ≤ 0.20 σ when the bias samples are
   rejuvenated → bias impoverishment, not an update error. Reported as such.
6. **ANEES** ≈ 1 in tracking is *conservative by a factor ≈ 2*, not "consistent to slightly
   conservative"; the "share inside χ²₂" metric cannot detect a factor-2 scale.
7. **E7.** The RBPF's bias advantage holds for converged runs only; over all runs its bias error is
   larger (1.52 vs 0.52 m/s hill, N = 5 000) because its exact update follows wrong position
   increments when locked on the wrong terrain. Tracking gain significant at N = 5 000 only
   (paired bootstrap 1.7 [0.4, 2.8] m hill); plains convergence gain significant at N = 2 000 only.
8. **"Worse than no terrain aiding"** is specific to this EKF (no gating, unconstrained bias).
9. Smaller items fixed: dead reckoning converges in up to 5 % of runs at σ₀ = 100 m (caption said
   never); roughening: tracking RMSE keeps falling with K, the K = 0.2 drop is not significant, and
   "quarter of the cost" was unsupported (cost not benchmarked); E6: K = 0.05 was chosen on the same
   hill flights 0–99 (now disclosed); roughening *with or without* the monitor helps the plains;
   E4: the map-error field does raise tracking RMSE (8.2 → 11.4 m) and its correlation length is an
   assumption; "immune" → "robust"; the EKF false-locks most on the hills, not the plains; plains
   non-converged runs end at a few tens of metres; methodology details (pointers, truncated prior,
   d = 5 with the offset state, σ_z in the NIS under Student-t, 13 nats for a 100 m outlier);
   remaining hand-typed numbers replaced by macros; cross-experiment reuse of flights disclosed;
   figure fixes (% axis, legends, prior line, label overlap, ANEES interval per row).

---

## 2026-11-19 — M5: fresh clone and repository hygiene

- **Fresh clone** (`git clone` of the local repo into a temporary folder, only the downloaded
  rasters copied in, no API calls): `make setup` (fresh venv; resolved NumPy 2.5.5 vs 2.5.4 here —
  `requirements-lock.txt` pins the versions used), `make test` (80 passed) and `make quick` (all
  seven experiments + figures) succeed with exit code 0.
- **Hygiene:** `.env` not tracked; no tracked file above 1.2 MB; all commits authored by the
  owner; no AI attribution in commit messages or tracked files.
- `requirements-lock.txt` written with `pip freeze --exclude-editable` (no local paths).

---

## 2026-11-19 — M4: report and documentation

- `report/latex/` is complete and Overleaf-ready (no local TeX by the owner's choice). In place of
  a compile, `scripts/check_report.py` checks that every generated-number macro is defined, every
  reference, input, figure, listing and citation resolves, every bib entry is cited, braces and
  environments balance, generated tables contain no bare % or _ and no unsupported Unicode, and
  listings are ASCII. It passes with 0 problems.
- Length estimate (not measured — no compile): ≈ 9 800 words of body text with 15 figures,
  7 tables and 4 listings ≈ 25–28 pages; three secondary figures were moved to the appendix to
  stay near the 15–25-page guideline.
- Every result number in the text is a macro from `report/generated/`; regenerating all outputs
  (`make figures`, 48 s) leaves every figure byte-identical.

---

## 2026-11-19 — E6 test (main seeds, M = 300 per cell): monitor, roughening and both

Settings fixed before this run (dev study + E3): window 20, p_FA = 0.001, α = 0.5; K = 0.05.
Converged % / false lock % at σ₀ = 2 km (hill / Kandy / plains):

| variant | hill | Kandy | plains |
|---|---|---|---|
| PF 5 000 | 40 / 54 | 39 / 58 | 37 / 6 |
| + monitor (primary) | **84 / 2** | **80 / 4** | 37 / 1 |
| + roughening | 83 / 16 | 81 / 18 | **60** / 1 |
| + both | **96 / 1** | **94 / 0** | 58 / 0 |

- Primary comparison (monitor vs PF): McNemar 3 vs 135 (hill), 4 vs 127 (Kandy) discordant runs,
  p < 0.001; plains 5 vs 6, p = 1.0 (no false locks to detect there).
- At σ₀ = 250 m the monitor alarms in 7–9 % of runs (≈ the 8 % dev estimate) and convergence is
  unchanged (98 → 100 %, 99 → 99 %, 62 → 61 %): false alarms are cheap.
- Roughening and the monitor are complementary: roughening prevents locks (fewer to catch: alarm
  rate at 2 km drops from 78 % to 22 % on the hill country), the monitor recovers from those that
  still happen. The combination with 5 000 particles beats PF 20 000 without either (E1, same
  flights: 70 %).
- On the plains only roughening helps (37 → 60 %): failures there are imprecision, not lock.

---

## 2026-11-19 — E7: Rao-Blackwellised PF vs plain PF (σ₀ = 1 km, M = 100 per cell)

Equal N is also equal runtime here (benchmark: RBPF 0.83 ms vs PF 0.83 ms per step at N = 5 000).
- **Convergence:** no significant difference on the hilly regions (McNemar p ≥ 0.09); on the
  plains the RBPF converges more often at every N (34/47/56/62 % vs 27/36/39/52 % for N = 500 …
  5 000; p = 0.008 at N = 2 000, 0.12–0.28 otherwise).
- **Tracking RMSE (N ≥ 2 000):** RBPF 7.9 / 7.0 m vs PF 9.2 / 8.6 m (hill), 8.8 / 7.6 vs
  11.4 / 10.5 m (Kandy).
- **Bias estimation — the clearest effect:** tracking-window bias RMSE (2-D) RBPF 0.19–0.33 m/s vs
  PF 0.43–0.58 m/s on the hilly regions at N ≥ 2 000. The prior alone gives 0.42 m/s
  (√2 × 0.3), so **the plain PF's bias estimate is no better than the prior — often worse**. This is
  static-parameter impoverishment: the bias random walk (0.002 m/s per step) is tiny, so after
  repeated resampling all bias values descend from a few ancestors and the estimate is roughly
  one random draw from the prior (expected 2-D error ≈ √2 × 0.42 ≈ 0.6 m/s). The RBPF computes the
  bias analytically and cannot impoverish. On the plains, where the PF resamples less (flatter
  weights), its bias error stays near the prior (0.27–0.37 m/s).
- At N = 500 the RBPF's bias error is no better (wide CI): with few particles the per-particle
  Kalman means are driven by few, noisy position increments.

---

## 2026-11-19 — E6 dev study (seed root 1000, M = 100 per cell): monitor parameters chosen

| window L | p_FA | converged at 2 km (pooled) | false alarms at 250 m (pooled) |
|---|---|---|---|
| 5 | 0.001 | 66 [60, 71] % | 19 [15, 23] % |
| 5 | 0.01 | 46 % | 63 % |
| 10 | 0.001 | 65 % | 12 % |
| 10 | 0.01 | 57 % | 43 % |
| 20 | 0.001 | 63 [57, 68] % | 8 [5, 11] % |
| 20 | 0.01 | 57 % | 29 % |
| no monitor | — | 39 [34, 45] % | — |

- **No setting met the pre-registered constraint** (false alarms ≤ 5 %). The rule did not say what
  to do then, and the code's silent fallback (highest convergence → L = 5, 19 % false alarms)
  was not pre-registered. **Decision, made on dev data only and before any test run:** keep the
  constraint's intent and take the lowest false-alarm rate → **L = 20, p_FA = 0.001** (8 %
  false alarms; 63 % vs the best 66 % convergence at 2 km, overlapping intervals).
- The monitor works on the dev seeds: convergence at σ₀ = 2 km rises from 39 % to 63–66 %.
- False alarms are nearly harmless: at σ₀ = 250 m with L ≥ 10, p = 0.001 the monitored PF
  converges in 97–99 % (hill/Kandy) and 63–64 % (plains), vs 98/95/65 % without monitor — the
  kept half of the cloud preserves the right hypothesis. The 5 % constraint was about alarm
  frequency; the measured cost of an alarm is what matters, and it is small.
- Expected false-alarm rate: the window test runs every step on a sliding window, so with
  p_FA = 0.001 per test the nominal rate per 300 s flight lies between ≈ 1.5 % (≈ 15 independent
  20-step windows) and the union bound ≈ 28 %; the observed 5–9 % (L = 20) is inside that range,
  so there is no evidence that the χ² calibration is off. (Correction of an earlier line in this
  entry that quoted an underived "0.3–6 %".)

---

## 2026-11-19 — E5: consistency and the posterior Cramér–Rao bound

Ensemble part (from the E1 runs, M = 300 per cell) and per-flight part (5 fixed flight plans per
region, M = 100 each, σ₀ = 500 m).
- **The bound is never violated:** the all-run RMSE is ≥ 1.10 × the ensemble PCRB at every
  step, for every region, σ₀ and filter (sanity check of the whole chain).
- **Ambiguity dominates the all-run error:** with false locks in the ensemble, the all-run RMSE in
  the tracking window is 37–756 × the bound on the hilly regions for σ₀ ≥ 500 m; at σ₀ = 100 m
  (no false locks) the PF is at 2.3× (hill) and 1.2× (plains) — the BCRB ignores which-hill
  ambiguity and is tight only once the filter is on the right hill.
- **Consistency:** in the tracking window of converged runs the PF's position ANEES is 0.95–1.9
  (95 % interval ≈ [1.8, 2.2]): consistent to conservative, as expected from its inflated process
  noise; 94–96 % of single-run NEES values lie inside the χ²₂ 95 % interval. Over all runs ANEES
  reaches 10¹³–10¹⁶ — false locks (tight wrong clouds) make the covariance meaningless. During
  acquisition even eventually-converging runs have ANEES ≫ 2: while most weight sits on a wrong
  mode and a few particles remain near the truth, the weighted covariance (≈ 50 m) understates
  the error (≈ 500 m). A Gaussian consistency measure does not fit a multimodal posterior.
- **Per-flight bound (hills):** converged PF 20 000 tracks at 1.9× (hill) / 2.0× (Kandy) the
  per-flight PCRB (medians over plans), PF 5 000 at 2.2× / 2.2×, EKF at 2.9× / 2.7×.
  **Why the gap — tested, not assumed:** a Kalman filter that knows the true slopes along the
  path but uses the PF's tuning (σ_p = 2 m/step instead of the true ≈ 0.05 m, σ_z = 3.6 m) has an
  actual tracking RMSE of 6.3 / 6.7 m (mismatched-gain recursion, `tan.pcrb.mismatched_kf_error`)
  — and PF 20 000 achieves 7.0 / 7.4 m, i.e. **1.05 / 1.07 × what its tuning allows**. Nearly the
  whole factor ≈ 2 to the PCRB is the deliberate process-noise inflation, not the particle
  approximation; PF 5 000 adds ≈ 20 % Monte Carlo error, the EKF 40–60 % (it linearises at its
  estimate, not at the truth).
- **Plains:** the conditional (converged-only) RMSE is *below* the bound (ratios 0.6–0.9). Not a
  violation: the bound (25–36 m) is comparable to the 50 m convergence threshold, so selecting
  runs with error < 50 m truncates the error distribution. The unconditional check above holds.

---

## 2026-11-19 — E4: robustness to model mismatch (hill country + Kandy, σ₀ = 500 m, M = 100)

Converged % (hill / Kandy):
- **Radar outliers** (N(0, 100 m) with probability p): Gaussian PF 96/92 → 91/78 → 65/39 →
  42/23 % for p = 0, 1, 5, 10 %; **Student-t PF (ν = 4) 97/95 → 97/96 → 95/96 → 94/92 %**, with
  tracking RMSE ≈ 8–10 m throughout (Gaussian PF degrades to ≈ 19 m). At p = 0 the Student-t
  costs nothing. EKF 23 % → 0 % (hill). Heavy tails are a free robustness gain here.
- **Map error** (correlated field std 2.4 m, ℓ = 1 km, plus a constant offset): the field alone
  is harmless (94/86 %). A **constant offset is what hurts**: nominal PF 73/60 % at 5 m,
  25/14 % at 10 m, 0/0 % at 20 m. σ_z × 4 (14.4 m) keeps 96/90 % at 5 m and 80/61 % at 10 m but
  only 11/9 % at 20 m. The **offset-state PF** is the only variant that still works at 20 m
  (48/48 %) and it estimates the offset well (mean estimate 18.3 / 19.4 m for a true 20 m,
  error std ≈ 4 m, converged runs) — **but it pays for it when there is no offset: 64/61 %
  instead of 94/86 %**: one more unknown lets a profile match at more places, and 5 000 particles
  now cover five dimensions. Trade-off, reported as such; a Rao-Blackwellised offset (linear in
  the measurement) or more particles would be the obvious next steps.
- **INS grade** (initial bias std 0.1 / 0.3 / 1.0 m/s): PF 95/96/83 % (hill), tracking RMSE
  6.4 / 8.2 / 15.9 m; EKF 36/23/17 %; dead reckoning never converges. The worse the INS, the
  worse the tracking, because the bias random walk the filters assume is small and the bias
  must be learned from terrain alone.

---

## 2026-11-19 — E6 test design, fixed before the test runs (extends the pre-registration)

The E6 test study (main seeds, M = 300, σ₀ ∈ {250, 2000} m, all three regions) will run four
variants on identical flights: `pf_5000` (baseline), `pf_monitor` (window and false-alarm
probability from the dev-seed rule; re-spread fraction 0.5), `pf_rough` (roughening K = 0.05)
and `pf_monitor_rough` (both). **Primary comparison (as planned): monitor vs baseline.**
Secondary: roughening and the combination, added because E3 showed roughening to be a strong
remedy against the same failure (false lock). K = 0.05 was taken from E3 (hill country,
σ₀ = 1 km, main seeds) — a different region/σ₀ mix than E6, but not independent data for the hill
country, which is stated in the report.

## 2026-11-19 — E3: particle-filter design (hill country, σ₀ = 1 km, M = 100)

- **N:** convergence 29 % (N = 500) → 71 % (5 000) → 93 % (20 000) → 98 % (50 000); the failure
  mode is false lock (65 % → 2 %). Tracking RMSE improves only modestly, 9.7 → 7.6 m. Runtime
  0.2 → 7 ms per step (`report/tables/runtime.csv`): even N = 50 000 runs ≈ 140× faster than the
  1 Hz measurement rate. Particles buy *acquisition robustness*, not tracking accuracy.
- **Resampling scheme:** no significant difference in convergence (McNemar p ≥ 0.15 vs
  systematic at N = 1 000 and 5 000) or tracking RMSE (overlapping CIs) — consistent with Hol et
  al. (2006), whose RMSE effects were "not significant". Measured offspring-count variance
  (N_eff ≈ 0.4 N, log-normal weights, 400 repetitions): multinomial 1.00 (theory 1.00),
  residual 0.44 (theory 0.44), stratified 0.27, systematic 0.18 = the floor/ceil minimum 0.179.
  Ranking multinomial > residual > stratified > systematic, as expected; systematic attains the
  minimum possible variance here. Runtime: systematic is the cheapest (0.13 ms vs 0.36 ms for
  multinomial at N = 5 000).
- **ESS threshold:** no detectable effect between 0.1 N and 1.0 N (p ≥ 0.5).
- **Roughening — the surprise:** K = 0.05 raises convergence from 71 % to 94 % (McNemar 24 vs 1
  discordant runs, p < 10⁻⁴) and lowers false lock from 27 % to 6 %; K = 0.1 gives 91 % and the
  best tracking RMSE region (6.8 m vs 8.6 m); K = 0.2 falls back to 79 % (p = 0.18). With
  N = 5 000 and K = 0.05 the filter matches N = 20 000 without roughening at a quarter of the
  cost. Interpretation: the jitter scales with the cloud's spread (σ = K E N^(−1/4)), i.e. it is
  an adaptive process-noise inflation — large (tens of metres) while the cloud is wide, where it
  keeps hypotheses near the truth alive, negligible once the cloud has collapsed. Too much (K =
  0.2) blurs the likelihood information. On the synthetic map (12 runs, findings 2026-11-04)
  roughening had looked unreliable; with M = 100 on real terrain the effect is clear.

---

## 2026-11-19 — E2: terrain informativeness (200 random flights per region, σ₀ = 500 m)

Setup: `configs/experiments/e2.yaml`; EKF and PF (N = 5 000); same flights for both.
- Convergence: PF 96 / 94 / 60 % (hill / Kandy / plains), EKF 26 / 26 / 12 %.
- **Per-flight terrain vs error (RMS error over the final 100 s, pooled over regions):**
  PF Spearman ρ = −0.63 [−0.69, −0.57] (rougher terrain under the flight → smaller error);
  **EKF ρ = +0.29 [0.21, 0.37] — the opposite sign**: on rougher terrain the EKF does *worse*.
  The same terrain texture that informs the PF breaks the EKF's local linearisation.
- Within a region the correlations are weak (PF ρ −0.16 / −0.26 / −0.05): most of the effect is
  between terrain types, not between flights of one region.
- Convergence rises with the terrain information collected along the flight (binned, both
  filters) — the PF from ≈ 50 % in the least informative bin to ≈ 95–99 % over the hills.
- **Along- vs cross-track:** no systematic anisotropy on average. Tracking RMS along/cross =
  6.0/6.0 m (hill), 7.4/7.5 m (Kandy), 14.7/15.4 m (plains) for the PF; the median ratio of
  along- to cross-track terrain information is ≈ 1 (0.98–1.05). The hypothesis that errors are
  systematically elongated along ridges/valleys is **not supported** at this level: the 25°
  heading weave and the varied ridge directions in 40 km boxes mix the two directions.
- On the plains the PF's "non-converged" runs sit mostly just above the 50 m threshold
  (RMS 50–100 m), i.e. imprecise, not lost.

## 2026-11-19 — E1: filters vs initial uncertainty on real terrain (M = 300 per cell)

Setup: `configs/experiments/e1.yaml`; 3 regions × σ₀ ∈ {100, 250, 500, 1000, 1500, 2000} m ×
300 random flights; DR, EKF, PF 5 000, PF 20 000 on identical flights. 27 min on 10 workers.
Converged (Wilson 95 % CI) / false lock, hill country:

| σ₀ (m) | EKF | PF 5 000 | PF 20 000 |
|---|---|---|---|
| 100 | 85 [81, 89] / 15 | 100 / 0 | 100 / 0 |
| 500 | 28 [23, 33] / 70 | 95 [92, 97] / 3 | 98 [95, 99] / 1 |
| 1000 | 14 / 84 | 72 [66, 76] / 25 | 92 [89, 95] / 6 |
| 2000 | 6 / 93 | 40 [35, 46] / 54 | 70 [64, 75] / 29 |

Kandy is almost identical to the hill country; on the plains the PF converges in only 67–46 %
(PF 20 000, σ₀ 100 → 2000 m) but **almost never false-locks (≤ 1 %)** — it stays honestly
uncertain — while the EKF false-locks in 42 % of runs at σ₀ = 500 m.

Interpretation:
- **The hypothesis from the synthetic terrain is confirmed, strongly.** On the smooth synthetic map
  the EKF held up to σ₀ = 1 km; on the real 30 m DEM it already fails in 15 % of runs at
  σ₀ = 100 m and 70 % at 500 m. Real terrain has slope variation at the scale of a few cells,
  so the one-cell Jacobian is wrong as soon as the error exceeds a few tens of metres.
- **The EKF is worse than no terrain at all.** Its median error grows during the flight and ends
  well above dead reckoning on both hilly regions: wrong slope-based corrections also corrupt its
  bias estimate, so it drifts faster than the uncorrected INS.
- **The PF's failure mode at large σ₀ is false lock**, not slow convergence (PF 5 000 at 2 km:
  54 % false lock). More particles help significantly from σ₀ ≥ 500 m (McNemar p < 0.05 on every
  region; p < 10⁻¹⁴ at ≥ 1 km on the hilly regions). This motivates E6.
- Tracking RMSE (converged runs) ≈ 6.5–9 m (hill), 7.4–10.6 m (Kandy), ≈ 21–23 m (plains),
  growing mildly with σ₀ (late convergers).
- PF vs EKF convergence: McNemar p < 10⁻¹⁰ in all 18 cells.
- Radar ranges from the corridor altitude: median of the per-flight maximum 1 051 / 872 / 337 m,
  largest 1 738 m (hill). The constant 3 m radar noise is an assumption that is optimistic at
  1–1.7 km range.

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
