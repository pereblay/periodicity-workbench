# Bondi-Hoyle Model Laboratory: implementation plan

Status: implemented for version 1.7.0. The work packages below are retained as
the scientific design, validation checklist, and release traceability record.

## Objective

Upgrade the current normalized Bondi-Hoyle teaching proxy into a scientifically
traceable model laboratory with:

- correct relative-velocity geometry in eccentric orbits;
- explicit classical and revised-binary formulations;
- separate normalized and physical calculation modes;
- validity diagnostics and physically meaningful warnings;
- optional observational attenuation and response delay;
- uncertainty estimates and comparison with simpler models;
- complete UI, report, help, and regression-test coverage.

The Newtonian wind-fed binary problem is the scope of this work. Relativistic
Kerr, Horndeski, Lee-Wick, shock-cone, and QPO calculations remain outside this
module and would require a separate laboratory.

## Scientific conventions

- `T0` is the periastron epoch and is never shifted by a response-delay fit.
- `v_o = 2 pi a / P` is the orbital velocity scale.
- `w_inf = v_inf / v_o` is the terminal-wind ratio.
- `v_w(nu) / v_o` is the instantaneous wind ratio.
- The compact object velocity is resolved into radial and tangential terms:

  ```text
  v_orb,r / v_o = e sin(nu) / sqrt(1 - e^2)
  v_orb,t / v_o = (1 + e cos(nu)) / sqrt(1 - e^2)
  v_rel^2 = (v_w - v_orb,r)^2 + v_orb,t^2
  ```

- The pressure-aware classical rate is:

  ```text
  Mdot_BH = 4 pi G^2 M_compact^2 rho /
            (v_rel^2 + c_s^2)^(3/2)
  ```

- The existing pressure-free expression is identified as the high-Mach
  Hoyle-Lyttleton limit.

## Work package 0 — Baseline and test scaffold

### Changes

- Add a test suite, initially `tests/test_bondi_hoyle.py`.
- Record reference outputs for the existing circular-orbit proxy before
  changing the implementation.
- Extract small, pure calculation functions from the current model so the
  scientific equations can be tested without Streamlit.
- Keep the current result dictionary readable while the new fields are added.

### Acceptance criteria

- The current circular reference curve is reproducible.
- Tests run without launching Streamlit.
- Existing model-laboratory families remain unaffected.

## Work package 1 — Orbital geometry and hard validity checks

### Changes in `app.py`

- Replace the scalar quadrature for `v_rel` with the full radial/tangential
  vector calculation.
- Return diagnostic arrays for separation, true anomaly, wind speed, radial
  orbital speed, tangential orbital speed, and relative speed.
- Remove the silent physical interpretation of the beta-law floor.
- Validate:
  - `R_donor < r_periastron`;
  - positive wind velocity throughout the sampled orbit;
  - finite orbital quantities;
  - donor Roche-lobe filling factor using an Eggleton approximation.
- Produce structured warnings rather than only text embedded in the UI.

### Acceptance criteria

- For `e = 0`, the new vector expression reduces to
  `v_rel^2 = v_w^2 + v_orb^2`.
- For eccentric test cases, the result matches the analytic vector equation.
- An orbit intersecting the donor is rejected.
- A near Roche-lobe-filling donor produces a prominent warning.
- No artificial velocity floor can hide an invalid geometry.

## Work package 2 — Classical and revised-binary formulations

### Changes

- Add a formulation selector:
  - `Classical BHL / Hoyle-Lyttleton`;
  - `Revised binary BHL (Tejeda-Toala)`.
- Add sound speed or gas temperature input and calculate the local Mach
  number.
- Calculate the accretion radius:

  ```text
  R_acc = 2 G M_compact / (v_rel^2 + c_s^2)
  ```

- Implement the revised geometric efficiency:

  ```text
  eta = 1/4 |1 - v_orb,r / v_w| (R_acc / r)^2
  ```

- Report the assumptions and reference attached to the selected formulation.
- Flag classical-model efficiencies above unity instead of silently accepting
  them. The revised formulation must remain within its physical efficiency
  domain.

### Acceptance criteria

- The sound-speed-zero case converges to the pressure-free limit.
- Circular revised-model results reproduce
  `eta = [q / (1 + w^2)]^2`.
- Low-wind test cases do not return an unflagged efficiency above one.
- Every output records the selected formulation and assumptions.

## Work package 3 — Separate normalized and physical modes

### Normalized teaching mode

- Preserve `y = C + A proxy` for shape exploration.
- Rename the input and output to `w_inf = v_inf / v_o`.
- Display instantaneous `v_w / v_o` separately.
- Mark fitted quantities as phenomenological.

### Physical mode

- Add donor mass-loss rate.
- Use absolute wind density:

  ```text
  rho_w = Mdot_w / (4 pi r^2 v_w)
  ```

- Include the explicit compact-mass dependence in `Mdot_acc`.
- Add compact-object radius or radiative-efficiency input.
- Calculate intrinsic accretion luminosity with a selectable prescription:
  `G M Mdot / R` or `eta_rad Mdot c^2`.
- Calculate and display the Eddington ratio; make an Eddington cap optional
  and never silently apply it.

### Acceptance criteria

- Physical `Mdot_acc` scales linearly with donor mass-loss rate and
  quadratically with compact mass when the remaining quantities are fixed.
- Physical outputs carry units and are not median-normalized.
- Normalized mode remains available and produces dimensionless curves.
- The UI and report cannot confuse the two modes.

## Work package 4 — Observable transfer model

### Changes

- Keep intrinsic accretion/luminosity separate from observed count rate.
- Add an optional attenuation layer:

  ```text
  F_observed = F_intrinsic exp[-sigma_eff N_H(phase)]
  ```

- Start with an explicitly phenomenological column-density model; do not claim
  full radiative transfer.
- Replace the current geometry-shifting `phase_lag` with an optional response
  delay applied after the intrinsic accretion curve is calculated.
- Support either a simple phase delay or a causal exponential response kernel.
- Warn when the fitted scale is negative and explain that absorption,
  reprocessing, or model failure can cause anticorrelation.

### Acceptance criteria

- Changing the response delay never changes the periastron epoch or orbital
  geometry.
- Intrinsic and attenuated curves are both available in plots and reports.
- A negative count-rate/accretion scale always produces a warning.
- With attenuation and delay disabled, the result reduces to the intrinsic
  model.

## Work package 5 — Inference, uncertainty, and model comparison

### Changes

- Keep robust least-squares as the fast default fit.
- Add parameter covariance where locally valid.
- Add bootstrap confidence intervals for the principal fitted parameters and
  extrema.
- Show a parameter correlation matrix.
- Add bounded priors/ranges informed by independently supplied system values.
- Compare the BHL model with:
  - constant flux;
  - one-harmonic sinusoid;
  - the existing Fourier alternative.
- Report delta AIC and delta BIC, not only the BHL model's isolated values.
- Identify parameters that hit bounds or are weakly constrained.

### Acceptance criteria

- Synthetic datasets recover injected parameters within reported uncertainty
  for identifiable cases.
- Strongly degenerate synthetic cases are labelled as such.
- Bootstrap execution can be disabled for quick classroom use.
- Model comparison uses the same filtered data and likelihood convention.
- Parameter-bound hits and singular covariance are visible warnings.

## Work package 6 — Regime and validity dashboard

### Diagnostics

- Minimum and maximum local Mach number.
- `R_acc / a` and `R_acc / r`.
- Roche-lobe filling factor.
- Minimum `r / R_donor`.
- Local and terminal wind ratios.
- Classical and revised capture efficiency.
- Eddington ratio in physical mode.
- Indicators for:
  - subsonic or transonic flow;
  - slow-wind regime;
  - large accretion radius;
  - likely importance of gradients/turbulence;
  - possible transient-disc behaviour;
  - likely absorption-dominated observable.

### Presentation

- Use status levels `valid`, `caution`, and `outside assumptions`.
- Explain that the flags are regime diagnostics, not hydrodynamic simulation.
- Reference Edgar, Tejeda-Toala, Xu-Stone, and the Vela X-1 observational
  example in the help text.

### Acceptance criteria

- Each warning is generated by a testable numerical condition.
- Warnings are returned by the scientific layer and reused by the UI/report.
- No result outside the model assumptions is presented without a warning.

## Work package 7 — UI, plots, report, and documentation

### `streamlit_app.py`

- Group controls into:
  - orbital geometry;
  - wind and gas;
  - accretor and physical normalization;
  - formulation;
  - observable response;
  - fitting and uncertainties.
- Add plots for:
  - intrinsic and observed folded curves;
  - separation and velocity components;
  - density, accretion radius, Mach number, and efficiency;
  - residuals;
  - confidence intervals and parameter correlations.
- Add a compact validity panel above the fitted-parameter table.

### Report

- Record the formulation, normalization mode, all physical assumptions, fit
  bounds, diagnostics, warnings, and literature attribution.
- Keep the existing logo and footer layout.
- Ensure legacy result dictionaries still render.

### Documentation

- Update `README.md`, `HELP.md`, formulas, parameter names, examples, and
  interpretation warnings.
- Add a short literature/assumptions section.
- Update the version only at release time; this scope is appropriate for a
  proposed `1.7.0` feature release.

### Acceptance criteria

- All new controls survive Streamlit reruns and are passed to the model.
- Every plotted quantity has units or is explicitly dimensionless.
- Reported formulas match the implementation.
- README, HELP, UI labels, report, and version agree.

## Work package 8 — End-to-end validation and release

### Validation cases

1. Circular, fast, highly supersonic wind: classical and revised models should
   approach the familiar BHL behaviour.
2. Eccentric orbit: vector velocity must create the expected asymmetric
   profile without moving `T0`.
3. Slow wind: classical over-capture warning and revised efficiency behaviour.
4. Finite sound speed: smooth transition away from the pressure-free limit.
5. Invalid donor/orbit intersection: hard failure with an actionable message.
6. Near Roche-lobe filling: caution/outside-assumptions result.
7. Physical scaling test using mass, mass-loss rate, and wind speed.
8. Absorbed synthetic light curve: intrinsic and observed maxima can differ.
9. Delayed response: orbital geometry stays fixed.
10. Negative fitted scale: absorption/model-failure warning.

### Release checklist

- Run unit and synthetic-regression tests.
- Run the app locally and exercise all BHL control combinations.
- Generate and visually inspect a BHL PDF report.
- Check that other model-laboratory families still work.
- Update `VERSION`, README, HELP, and release notes together.
- Commit in reviewable stages and push only after the end-to-end check passes.

## Recommended implementation order and commit boundaries

1. `test: add BHL scientific regression scaffold`
2. `fix: use vector relative velocity and validate orbital geometry`
3. `feat: add pressure-aware and revised binary BHL formulations`
4. `feat: separate normalized and physical accretion modes`
5. `feat: add attenuation and causal response options`
6. `feat: add BHL uncertainty and model comparison`
7. `feat: add BHL validity dashboard and diagnostics`
8. `docs: update BHL UI, report, README, and HELP`
9. `release: validate and prepare version 1.7.0`

Each commit should leave the application runnable. Scientific-core commits
must include their associated tests rather than deferring all tests to the end.
