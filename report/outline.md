# Report outline

Structure follows the assignment brief (introduction, methodology, implementation,
results, discussion, conclusion). Rubric item each section serves is in brackets.

## 1. Introduction  [clarity]
- GNSS-denied navigation; why INS alone drifts
- Terrain-aided navigation: idea, where it is used (aircraft, cruise missiles, AUVs)
- Why a particle filter: nonlinear, multimodal terrain likelihood
- Contributions of this project

## 2. Methodology  [understanding]
- Bayesian filtering recursion (prediction / update)
- State, motion model and measurement model — derive from INS + baro + radar altimeter
- SIR particle filter: importance sampling, weight update, degeneracy, N_eff, resampling
- Resampling schemes compared
- EKF baseline: linearisation through the DEM gradient, and why it breaks
- Data: Copernicus GLO-30, reprojection to UTM 44N, DSM vs DTM

## 3. Implementation  [correctness]
- Code structure (one diagram), vectorisation, log-domain weights
- Key code snippets: predict, update, systematic resampling
- Testing strategy (unit tests on synthetic terrain)
- Parameters table (configs/default.yaml)

## 4. Results  [effectiveness]
- Exp 1: PF vs EKF vs dead reckoning; initial uncertainty sweep
- Exp 2: hill country vs rolling vs flat terrain
- Exp 3: particle count / runtime / resampling scheme
- Exp 4: robustness to outliers and canopy bias
- Exp 5: Monte Carlo RMSE + NEES consistency
- Particle snapshot figure: multimodal posterior collapsing

## 5. Discussion  [depth, creativity]
- When terrain is informative, and when it is not (link to terrain roughness)
- Failure modes: impoverishment, flat terrain, map errors
- Limitations of the simulation; what real flight data would add
- Possible extensions: Rao-Blackwellised PF, point-mass filter, terrain-informativeness gating

## 6. Conclusion

## References / Appendix (full code listing link to the repo)
