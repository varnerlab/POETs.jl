# Paper Review Issues — Reviewer Adversary Analysis

Issues identified during internal review of the ParetoEnsembles.jl manuscript, organized by severity and effort to address.

## Critical (could lead to rejection)

### Issue 2: Inverse crime on coagulation — ADDRESSED
~~The coagulation example uses the same model to generate and fit data.~~
**Resolution:** Added model misspecification study (SI Section S7) where 6 fixed rate constants are perturbed by ±30%. Ensemble still predicts held-out peaks to 4–6%, but trajectory coverage degrades to 31–48%. Discussion now explicitly acknowledges the synthetic data limitation and references the misspecification study. This preempts the criticism directly.

### Issue 8: "Ensemble" vs "posterior" — no probabilistic interpretation
The ensemble is not a Bayesian posterior. The 95% CI is based on quantiles of the ensemble, not a credible interval. Bayesian reviewers will ask how this compares to MCMC/SMC posterior sampling. The paper never discusses this distinction. Need a paragraph in the Discussion positioning the ensemble relative to Bayesian approaches and being clear about what the "95% CI" actually represents.

## Important (will be raised, need good answers)

### Issue 1: "Why not just use NSGA-II?"
NSGA-II is 50-150× faster. The answer is ensemble density (10-30× denser fronts), but the paper doesn't make the case that density *matters* for UQ quality. Need to connect front density to prediction interval quality explicitly.

### Issue 5: Coverage failures are unexplained
Peak thrombin and ETP are systematically not covered at held-out conditions. Reported honestly but with no mechanistic explanation. Is it ensemble size? SA underexploration? Biased noise draw? Need a hypothesis.

### Issue 6: No formal coverage analysis
Coverage is reported for a single noise realization. A reviewer will ask: is the bias reproducible or just this noise draw? Running 5 replicates with different seeds would turn an anecdote into a result.

### Issue 10: Missing related work
No citations for Bayesian ODE inference (Stan, Turing.jl, emcee), parameter estimation toolboxes (PESTO), or approximate Bayesian computation (pyABC). Need to position the method against Bayesian approaches, not just evolutionary MOO.

## Minor (should address but won't sink the paper)

### Issue 3: Only one real-data example
Cell-free uses experimental data; coagulation is synthetic. A second real-data application would strengthen the paper but may be out of scope for this revision.

### Issue 4: No comparison of UQ quality between methods
HV and IGD are compared with NSGA-II, but not the *uncertainty quantification* (prediction intervals, coverage). Does an NSGA-II ensemble give similar prediction intervals?

### Issue 7: Rank ≤ 1 cutoff is ad hoc
Why include rank 1 solutions in the ensemble? This choice affects all downstream results. No sensitivity analysis of this cutoff.

### Issue 9: Scaling claim is weak
34 species / 10 parameters is moderate by modern MOO standards. The "scales effectively" claim should be scoped more carefully.
