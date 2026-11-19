# Walkthrough — how the project fits together

A guide for the project owner: where every equation lives in the code, how to regenerate every
number and figure, what each figure shows, the known limitations, and likely viva questions.
Section numbers refer to the LaTeX report (`report/latex/main.tex`).

---

## 1. Equations → code

| Report (section, equation) | What | Code |
|---|---|---|
| §2.1, Bayes recursion | predict → update → (resample) → estimate | `ParticleFilter.step` in `src/tan/filters/particle_filter.py` |
| §2.2, motion model `x_k = F x_{k-1} + G u_k + w_k` | INS integration with per-particle bias | `ParticleFilter.predict`; `ExtendedKalmanFilter.predict` (`filters/ekf.py`); `DeadReckoning.predict` |
| §2.2, measurement `z = baro − radar = h(p) + v` | simulated in the world, evaluated by the filters | `simulate()` in `src/tan/simulate.py`; `DEM.height` (bilinear, `src/tan/dem.py`) |
| §2.3, SIR weights in log space, log-sum-exp | weight update and normalisation | `ParticleFilter.update` (uses `scipy.special.logsumexp`) |
| §2.3, MMSE estimate and covariance | weighted mean / covariance | `ParticleFilter.estimate` |
| §2.3, `N_eff = 1 / Σ w²`, resample if `N_eff < τN` | degeneracy test | `effective_sample_size` (`filters/resampling.py`), decision in `ParticleFilter.step` |
| §2.3, four resampling schemes | inverse-CDF at sorted pointers | `multinomial`, `systematic`, `stratified`, `residual`, `_cdf`, `_invert` in `filters/resampling.py` |
| §2.3, roughening `σ_j = K E_j N^(−1/d)` | jitter after resampling | `ParticleFilter.roughen` |
| §2.4, EKF with DEM-slope Jacobian, Joseph form | linearised baseline | `ExtendedKalmanFilter.update`; slope from `DEM.gradient` |
| §2.5, Student-t log-likelihood (eq. studentt) | outlier-robust weights | `ParticleFilter.log_likelihood` (`likelihood: student_t`, `nu`) |
| §2.6, NIS, windowed χ² test, re-spread mixture | divergence monitor | `ParticleFilter._check_monitor`, `_respread`, `_propagate_dr`; NIS computed in `update` |
| §2.7, RBPF (marginalised bias, shared Riccati) | Rao-Blackwellised filter | `RaoBlackwellizedPF.predict` / `.estimate` in `filters/rbpf.py` |
| §2.8, PCRB recursion, true noise `Q_true` | theoretical bound | `pcrb`, `true_noise`, `slope_outer` in `src/tan/pcrb.py` |
| §2.9, roughness, slope, Fisher density | terrain statistics | `src/tan/terrain.py` |
| §2.10, converged / false lock / tracking RMSE / ANEES / Wilson / bootstrap / McNemar | evaluation | `src/tan/metrics.py` |
| §3.2, common random numbers | named, fixed-size random streams | `run_streams`, `filter_rng` in `src/tan/simulate.py` |
| optional offset state (E4 extension) | `z = h(p) + c + v` | `height_bias_sigma0` in `ParticleFilter` |

## 2. Re-running and regenerating

```bash
make setup && make test        # environment + 78 unit tests
make data                      # needs OPENTOPO_API_KEY in .env; 4 API calls; cached
make benchmark                 # runtime table — run on an idle machine
make experiments               # e1..e7, resumable, ~1 h on the recorded laptop
make figures                   # all figures/tables/macros from saved results
```

One experiment: `python scripts/sweep.py --experiment e3 [--quick] [--jobs 10]`, then
`python scripts/make_figures.py --experiment e3`. The experiment definitions are
`configs/experiments/e*.yaml` (shared settings in `_common.yaml`). Results land in
`results/<exp>/<study>/<configuration>/run_XXXX.npz` plus `results/<exp>/summary.csv`; figures in
`report/figures/<exp>_*.{pdf,png}`, tables in `report/tables/<exp>_*.{tex,csv}`, numbers in
`report/generated/<exp>.tex` (LaTeX macros used by the report text).

Other scripts: `survey_regions.py` (region choice), `terrain_stats.py` (terrain table and
figures), `validation.py` (filters vs exact Kalman filter), `extract_listings.py` (code excerpts,
parameter table), `benchmark.py` (runtimes).

To try anything without real terrain: append `@synth` to an experiment name
(`--experiment e1@synth`) to run it on the synthetic map; such results never touch `report/`.

## 3. What each figure shows

*(Filled in once the full experiments have run — see below.)*

## 4. Known limitations

- **Simulated sensors and flights.** No real flight data; the altimeter noise (2 m baro, 3 m radar,
  white, range-independent) is an assumption, not a datasheet value.
- **Constant altitude, 2-D state.** The UAV flies level; altitude and attitude errors are not
  estimated.
- **DSM vs what the radar sees.** The map is a surface model; the simulation measures the same
  surface unless E4 adds a map error, so canopy effects are modelled only as an offset and a
  smooth random field.
- **Filter process noise is a tuning choice** (σ_p = 2 m/step vs ≈ 0.05 m/step true), which makes
  the filters conservative and keeps them above the PCRB.
- **One INS model** (velocity bias random walk + white noise); no attitude-coupled errors.

## 5. Likely viva questions

*(Short answers; the numbers are in the report and in `report/generated/`.)*

1. **Why can't you just use an EKF?** The terrain likelihood is multimodal and the EKF linearises
   with the slope of one 30 m cell while the position is uncertain by hundreds of metres; it can
   only follow one hypothesis and confidently picks the wrong one when the slope misleads it.
2. **What is the state and why the bias?** `[px, py, bx, by]`; the INS velocity bias is what makes
   dead reckoning drift, and it is only observable through the terrain match.
3. **Why work with log-weights?** One 50 m residual at σ_z = 3.6 m is a factor e^-96; a few such
   factors underflow double precision.
4. **What does resampling do, and why not every step?** It removes negligible-weight particles and
   duplicates heavy ones (unbiased, `E[N_i] = N w_i`), but adds Monte Carlo noise and loses
   diversity; hence only when `N_eff < 0.5 N`.
5. **Systematic vs multinomial?** Same expectation, lower variance and one random number;
   Hol et al. (2006) recommend systematic.
6. **How do you know the particle filter is correct?** On a tilted plane the EKF is the exact
   Kalman filter; the PF and RBPF match it within a fraction of a standard deviation
   (`scripts/validation.py`).
7. **What are common random numbers and why use them?** Run i has identical flights and noise in
   every configuration, so comparisons are paired; McNemar's test then uses only the runs where
   exactly one filter converged.
8. **What is a false lock and how do you detect it without ground truth?** A tight particle cloud on
   the wrong terrain; the predictive NIS is persistently too large, so a windowed χ² test fires.
9. **What does the PCRB tell you and why is it loose early on?** A lower bound on any estimator's
   MSE; it uses the slope at the true position, so it ignores ambiguity (which hill am I over?).
10. **Why Rao-Blackwellise?** The bias is linear-Gaussian given the position history; computing it
    exactly with a Kalman filter removes its sampling variance.
11. **Why these three regions?** Chosen by a pre-registered roughness rule on a COP90 survey —
    rugged, intermediate and flat — so the terrain comparison could not be tuned.
12. **What is the Copernicus DEM's accuracy and why does DSM matter?** < 4 m absolute, < 2–4 m
    relative (LE90); it includes vegetation and buildings, which the radar may not see.

*(More questions on the experimental results are added with the results.)*
