# Paper Review Issues — Reviewer Adversary Analysis

Issues identified during internal review of the ParetoEnsembles.jl manuscript, organized by severity and effort to address.

## Critical (could lead to rejection)

### Issue 2: Inverse crime on coagulation — ADDRESSED
~~The coagulation example uses the same model to generate and fit data.~~
**Resolution:** Added model misspecification study (SI Section S7) where 6 fixed rate constants are perturbed by ±30%. Ensemble still predicts held-out peaks to 4–6%, but trajectory coverage degrades to 31–48%. Discussion now explicitly acknowledges the synthetic data limitation and references the misspecification study. This preempts the criticism directly.

### Issue 8: "Ensemble" vs "posterior" — ADDRESSED
~~The ensemble is not a Bayesian posterior. The 95% CI is based on quantiles of the ensemble, not a credible interval.~~
**Resolution:** Added paragraph in Discussion distinguishing ensemble from Bayesian posterior. Added Stan, Turing.jl, emcee citations in intro. Explained that 95% intervals are empirical quantiles, not credible intervals, and that coverage depends on SA exploration, not convergence to a stationary distribution.

### Issue 3: Only one real-data example — ADDRESSED
~~Cell-free uses experimental data; coagulation is synthetic.~~
**Resolution:** Added real-data coagulation example (Section 3.5) using Butenas et al. 1999 prothrombin titration data (Figure 3). Train on 50%, 100%, 150% FII → validate on 75%, 125% FII. Results: 3-18% training error, 8-9% validation error. New figure (fig_coagulation_realdata.pdf) added to main text. Abstract and Conclusions updated. **Note:** digitized data points are approximate — should verify with precise digitization before final submission.

## Important (will be raised, need good answers)

### Issue 1: "Why not just use NSGA-II?" — ADDRESSED
~~NSGA-II is 50-150× faster.~~
**Resolution:** Added paragraph in Discussion connecting ensemble density to UQ quality — meaningful prediction intervals require hundreds to thousands of members, not the 100-200 from NSGA-II.

### Issue 5: Coverage failures explained — ADDRESSED
~~Peak thrombin and ETP systematically not covered at held-out conditions.~~
**Resolution:** Added hypothesis in Discussion: SA explores along the Pareto front efficiently but may underexplore orthogonal directions where objective values are constant but predictions differ.

### Issue 6: No formal coverage analysis — ADDRESSED
~~Coverage is reported for a single noise realization.~~
**Resolution:** Ran 5 replicates with independent noise seeds. Results: lag time 100% coverage (5/5), time-to-peak 100% (5/5), peak thrombin only 40% (2/5), ETP 60% (3/5). Trajectory coverage 87-90% average. Mean peak error 7.1 ± 5.0%. Pattern is systematic and reproducible — timing well-calibrated, amplitude overconfident. Wired into Discussion.

### Issue 10: Missing related work — ADDRESSED
~~No citations for Bayesian ODE inference.~~
**Resolution:** Added Stan, Turing.jl, emcee citations in intro with paragraph positioning Pareto ensemble methods relative to Bayesian approaches.

## Minor (should address but won't sink the paper)

### Issue 4: No comparison of UQ quality between methods — PARTIALLY ADDRESSED
~~HV and IGD compared with NSGA-II, but not prediction intervals or coverage.~~
**Resolution:** Added argument in Discussion that meaningful prediction intervals require hundreds-to-thousands of members, not the 100-200 from NSGA-II. The density → UQ argument is now explicit. A direct computational comparison on the coagulation problem could strengthen this further but is not critical given the benchmark comparison already in SI.

### Issue 7: Rank ≤ 1 cutoff is ad hoc — ADDRESSED
~~Why include rank 1 solutions?~~
**Resolution:** Sweep of rank cutoff from 0 to 5 on the coagulation ensemble showed mean held-out peak error changes by <0.2 percentage points (5.9% at rank 0 to 6.1% at rank ≤ 5). Added sentence in Discussion confirming insensitivity. Non-issue.

### Issue 9: Scaling claim is weak — ADDRESSED
~~"Scales effectively" overstates.~~
**Resolution:** Softened to "can handle systems of moderate complexity."
