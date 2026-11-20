# Overview

Periodicity Workbench is an interactive tool for exploring periodic signals in unevenly sampled light curves. It was designed around the workflow that often appears in optical and X-ray time-series analysis: load a light-curve table, search for periods with a Lomb-Scargle periodogram, check whether peaks may be caused by the sampling pattern, fold the data, model the folded profile, remove known signals by prewhitening, and then explore more specialized physical or pedagogical models.

The tool is intentionally exploratory. It can help identify candidate periods, aliases, harmonics, sampling-window artefacts, time-dependent period changes, and simple phenomenological models. It should not be treated as a replacement for a final publication-grade physical solution. The results are best used as a guide for deciding what deserves a more careful analysis.

Current application version: **1.8.0**.

## Version 1.8 Student Workflow and Reporting Highlights

Version 1.8 introduces a guided evidence notebook for the minimum student workflow. It tracks five required stages: inspection of the uploaded light curve, reviewed bibliography, frequency analysis and an explicit periodicity decision, conditional interpretation of the folded profile, and a final assessment compared with the literature.

The report builder now has a fixed core workflow. Incomplete evidence produces a clearly labelled draft and a list of pending stages; completing the five stages produces a final report. Iterative Prewhitening, Period Tomography, and one justified Model Laboratory model are voluntary and may be selected in any combination, while `None` excludes the other choices. Their report controls become available only after the corresponding analyses exist.

Every report page carries the 150-pixel VIU logo and the ruled `12MAST AG2` / `Student name - Course` footer. All analysis-step expanders in the left sidebar start closed so the student can open the workflow progressively.

## Version 1.7 Scientific Model Highlights

Version 1.7 expands the Bondi-Hoyle laboratory with full vector wind/orbit relative velocity for eccentric systems, classical pressure-aware and revised-binary formulations, normalized and physical accretion modes, Mach-number and accretion-radius diagnostics, Roche-lobe and capture-efficiency checks, optional attenuation and response-delay experiments, parameter covariance, residual-bootstrap intervals, model comparison, and structured validity warnings shared by the interface and report.

## Version 1.6 Timing Highlights

Version 1.6 adds timing-provenance checks and a substantial expansion of the X-ray pulse-timing laboratory. It can inspect FITS barycentric metadata, apply a clearly labelled geocentric approximation when possible, estimate epoch-folding significance and period uncertainty, derive candidate template TOAs, compare F0 and F0+F1 spin ephemerides, and explore circular or Keplerian Roemer-delay models. Pulsed fractions can include a supplied background and Monte Carlo intervals.

PDF reports now include the 150-pixel VIU logo on every generated page and a ruled footer showing `12MAST AG2` and `Student name - Course`. Report metadata is taken directly from the visible report form to avoid stale student or course values.

## General Workflow

1. Upload an ASCII numeric table or a FITS table.
2. Select the columns containing time, flux/count rate/magnitude, and optionally error.
3. Choose time units: days or seconds.
4. Set a frequency range for the Lomb-Scargle search.
5. Run the main analysis.
6. Inspect the Lomb-Scargle periodogram, the sampling window, and the folded light curve.
7. Exclude obvious sampling-window artefacts if needed.
8. Use prewhitening to remove selected periodicities and search for residual signals.
9. Use Period Tomography to inspect whether the periodicity changes with time.
10. Use Model Laboratory for more specific phenomenological or educational fits.
11. Complete the five-stage student evidence notebook and generate the PDF report. Missing evidence is allowed only as a clearly marked draft.

## Minimum Student Workflow

This is the minimum required workflow for a student report. The evidence notebook appears below the analysis results and tracks completion of the five stages.

1. **Upload and inspect the light curve.**
   Upload the ASCII or FITS light curve, select the relevant columns, and inspect the original data before interpreting a periodogram. Comment on the appearance of the curve: observing coverage, gaps, scatter, trends, outliers, changes in amplitude, and possible instrumental features.

2. **Search and review the bibliography.**
   Use **Bibliographic Search** from the Input panel and consult ADS, arXiv, or other appropriate sources. Record the useful references and add a concise review of each relevant published result. A list of links without commentary is not sufficient evidence.

3. **Analyse frequencies and decide whether periodicity is measurable.**
   Inspect the Lomb-Scargle periodogram, sampling window, false-alarm probabilities, aliases, baseline, and signal-to-noise. Select and justify a frequency only when the data support it. The notebook requires an explicit conclusion: reliable periodicity, no reliable periodicity, or inconclusive. Discuss possible harmonics and distinguish them from independent signals.

4. **Study the folded profile when periodicity is reliable.**
   Fold the light curve with the selected period and comment on coherence, phase-dependent structure, maxima and minima, binning, harmonics, and possible `P` versus `2P` ambiguity. This stage is required only when the student concludes that a reliable periodicity can be determined. For a justified non-periodic or inconclusive result it is marked not required.

5. **Write the final assessment and compare it with the literature.**
   Summarize what was done, state the limitations of the result, and compare the final conclusion with the reviewed publications. Explain agreement, disagreement, or why the available data cannot resolve the question.

The final report always includes this core workflow. If any required evidence is missing, the interface and PDF identify the report as a draft and list the pending stages.

### Voluntary Extension

The student may additionally:

- use Iterative Prewhitening to subtract one or more justified signals and inspect the residual periodogram;
- use Period Tomography to look for stable, drifting, intermittent, or otherwise time-dependent behaviour; and/or
- apply one Model Laboratory model and justify why that single model is appropriate for the observed data.

These sections are voluntary and can be selected in any combination. Selecting `None` excludes all three; selecting any analysis removes `None`. The report controls enable each section only after the corresponding analysis exists. A Model Laboratory report includes only the currently fitted model, not a collection of unrelated model trials.

## Input

The input file can be either a plain ASCII table with numeric columns or a FITS table. The app accepts ASCII files with `.txt`, `.dat`, `.lc`, or no extension. FITS files are detected from their content, so common extensions such as `.fits`, `.fit`, `.fts`, `.ftz`, `.lc`, or compressed FITS-like uploads can be used as long as the file contains a valid FITS payload. In ASCII files, comment lines beginning with `#` or `%` are ignored.

For ASCII uploads, the app rejects files containing non-ASCII bytes, control characters, or command-like text. This is a basic safety check intended to avoid treating executable or shell-like content as a data file. FITS uploads are opened with `astropy.io.fits` and only scalar numeric table columns are offered for analysis.

When a FITS file is uploaded, the workspace preview shows the available table extensions. After selecting an extension, the app lists its scalar numeric columns and lets the user choose the time, flux, and optional error columns. The data table preview and light-curve preview are then built from that exact selection.

### Parameters

**Time**
Column number, or FITS column name, containing the observation time.

**Flux**
Column number, or FITS column name, containing the measured signal. This may be a flux, count rate, magnitude, or any scalar observable.

**Error**
Column number, or FITS column name, containing the uncertainty on the measured signal. This can be disabled with **Use error column**.

**Time units**
If set to `days`, frequencies are shown in cycles/day and periods in days. If set to `seconds`, frequencies are shown in Hz and periods in seconds.

**Use error column**
When enabled, weighted fits use weights

```text
w_i = 1 / sigma_i^2
```

where `sigma_i` is the error value. When disabled, all points are assigned equal weight.

**Flux is magnitude**
When enabled, plots using the flux axis are displayed with an inverted vertical axis, as is customary for magnitudes.

**Object name / Resolve SIMBAD**
Object name used to resolve sky coordinates through the SIMBAD/Sesame name resolver. The resolved ICRS coordinates are written into the RA and Dec fields and can be edited manually.

**RA / Dec**
Target coordinates in decimal degrees. They are recorded in the input summary and are required when the automatic timing mode needs to apply an approximate geocentric-to-barycentric correction.

**Barycentric timing**
Controls how the time reference is assessed. In automatic mode, FITS headers such as `TIMEREF`, `TIMESYS`, `TASSIGN`, `MJDREF`, and `PLEPHEM` are inspected. `TIMEREF=SOLARSYSTEM` is treated as confirmation that the selected times are already barycentric. A selected `BJD` column is also treated as barycentric.

When barycentric provenance is not confirmed, the app attempts an approximate geocenter-to-Solar-System-barycenter correction if target coordinates and an absolute MJD/JD epoch are available. FITS `TIME` columns can be converted to an absolute epoch when `MJDREF` and the time unit are present. ASCII data can only be corrected automatically when its numerical time axis is recognizable as MJD or JD.

The approximate correction does not use the spacecraft orbit. For a low-Earth-orbit satellite this can leave timing errors of up to roughly 25 ms, and mission-specific clock corrections may also be absent. Results produced under this approximation are therefore labelled demonstrative and must not be assigned a robust physical timing interpretation. Use **Already barycentric** when an ASCII file is known to contain corrected times, or **Keep input times** to disable correction while retaining the warning.

**Bibliographic Search**
After the object has been resolved with SIMBAD/Sesame, the Bibliographic Search button below the coordinates opens a temporary workspace view, similar to this help page, without clearing the current analysis state. It uses the resolved object name, prepares astronomy-literature searches for time-series, period, periodicity, timing, and variability terms, and displays an automatic arXiv result list when network access is available. The panel also provides a direct ADS query link, which should be used for the most complete astronomy bibliography.

**xmin, xmax, ymin, ymax**
Optional limits applied before the analysis. These restrict the time and flux ranges used by the calculations.

## Frequency Search

This block controls and launches the main Lomb-Scargle analysis.

For unevenly sampled data, the Lomb-Scargle periodogram is a standard way to estimate the power of a sinusoidal signal as a function of angular frequency or frequency. In this app the searched frequency is `f`, and the period is

```text
P = 1 / f
```

The Lomb-Scargle model at each frequency is essentially

```text
y(t) = C + A cos(2 pi f t) + B sin(2 pi f t)
```

where `C`, `A`, and `B` are fitted at each trial frequency.

### Parameters

**Run analysis**
Runs the Lomb-Scargle analysis using the current input file, selected columns, analysis limits, and frequency-search settings.

**Min Frequency / Max Frequency**
The frequency interval searched by the Lomb-Scargle periodogram. Use cycles/day when the time axis is in days, and Hz when the time axis is in seconds.

**Samples per peak**
Controls how finely the frequency grid samples the expected width of periodogram peaks. A larger value gives a denser grid and smoother-looking peaks, but increases computation time.

The natural frequency resolution is approximately

```text
Delta f ~ 1 / T
```

where `T` is the total time baseline. The number of evaluated frequencies scales roughly as

```text
N_f ~ (f_max - f_min) T * samples_per_peak
```

**Max considered peaks**
Maximum number of Lomb-Scargle peaks reported as candidates.

**Min. considered period**
Peaks with periods shorter than this value are not considered as primary candidates. This is useful for suppressing known short-period aliases, such as daily sampling features.

**Sampling-window threshold**
Min. sampling-window power required for a window peak to be considered important.

**Sampling-window tolerance**
Relative tolerance used to decide whether a Lomb-Scargle peak is close enough to a sampling-window peak to be flagged as an artefact.

## Sampling Window

The sampling window describes the periodicities introduced only by the observation times. It is computed from the observation times, not from the measured fluxes.

A simplified form of the spectral window is

```text
W(f) = |sum_j exp(2 pi i f t_j)|^2 / N^2
```

where `t_j` are the observation times and `N` is the number of observations.

If the sampling window has a strong peak at a period also seen in the Lomb-Scargle periodogram, the data peak may be an alias or artefact rather than an intrinsic signal. This is especially important near one day, lunar/monthly cadences, yearly aliases, or regular satellite/orbital sampling patterns.

## False Alarm Probability

For each Lomb-Scargle peak the app reports a false alarm probability, FAP. It estimates the probability that noise alone could produce a peak with equal or greater power somewhere in the searched frequency range.

Small FAP values suggest a more significant periodicity. However, FAP values should be interpreted cautiously when the data contain red noise, trends, strong outliers, non-Gaussian errors, or when the searched frequency range was tuned after looking at the data.

## Uncertainties

The **Uncertainties** subsection inside **Frequency Search** estimates period and frequency errors using bootstrap resampling.

The basic idea is:

1. Resample the light curve points with replacement.
2. Recompute the Lomb-Scargle periodogram.
3. Search for the local peak near the original candidate period.
4. Repeat many times.
5. Use the distribution of recovered frequencies or periods to estimate the uncertainty.

The reported error is based on the central percentile interval, usually the 16th to 84th percentile range.

### Parameters

**Bootstrap iterations**
Number of bootstrap resamplings. Larger values give more stable errors but take longer. A common practical value is 1000.

**Bootstrap local width**
Fractional search width around each candidate frequency. For example, a width of `0.03` searches within about +/-3 percent of the original frequency.

## Manual Exclusions

The **Manual Exclusions** subsection inside **Frequency Search** allows the user to prevent selected periods from being used as the primary period. This is useful when a peak is clearly due to sampling, aliasing, instrumental effects, or a known contaminating signal.

### Parameters

**Exclude periods from primary selection**
List of detected candidate periods that can be excluded.

**User-specified period(s)**
Manual list of periods to exclude, separated by spaces, commas, or semicolons.

**Manual exclusion tolerance**
Relative tolerance used to decide whether a detected period matches an excluded period.

For a candidate period `P` and excluded period `P_ex`, the comparison is approximately

```text
|P - P_ex| < tolerance * max(P, P_ex)
```

## Folded Profile

Folding converts observation time into orbital or pulse phase:

```text
phase = ((t - T0) / P) mod 1
```

where `P` is the folding period and `T0` is the reference epoch.

The app bins the folded data into phase bins. The weighted mean in a bin is

```text
mean = sum(w_i y_i) / sum(w_i)
```

with

```text
w_i = 1 / sigma_i^2
```

when errors are available.

### Parameters

**Phase bins**
Number of phase bins used to compute the displayed folded profile.

**T0**
Reference epoch for phase zero. If left blank, the app uses the default reference from the analysis.

**Folded-fit frequencies: harmonics**
Fits the folded profile with the main period and a selected number of harmonics:

```text
y(phi) = C + sum_k [a_k cos(2 pi k phi) + b_k sin(2 pi k phi)]
```

**Folded-fit frequencies: selected**
Allows selected detected periods to be included in the folded model. The first selected period is used as the folding period, while the others enter the fitted model as additional frequencies.

**Harmonic diagnostics**
After the frequency search, the app checks the primary period, its harmonics (`P/2`, `P/3`, ...), and longer multiples such as `2P` and `3P`. The table reports whether a nearby Lomb-Scargle peak is present, the false-alarm probability, the sampling-window power, and a short interpretation note. The `2P` row is useful for eclipsing binaries, where two similar minima can make the strongest periodogram peak appear at half the true orbital period.

**Sinusoidal fitting**
Controls how simple sinusoidal models are fitted in folded profiles. The same stored choice is also used as the default for related sinusoidal fits in prewhitening and some Model Laboratory blocks.

**standard** uses weighted least squares.
**robust** uses a soft-L1 loss to reduce the effect of strong outliers.
**display-optimized** rescales the plotted model amplitude and offset to better match the displayed folded profile. This is useful for visual teaching, but it should not be interpreted as a strictly statistical best fit.

**Update folded profile**
Recomputes only the folded profile without rerunning the full Lomb-Scargle analysis.

## Iterative Prewhitening

Prewhitening removes selected periodic signals from the original data and searches for remaining periodicities in the residuals.

At each step the app fits a sinusoidal model to the selected period or periods:

```text
y(t) = C + sum_k [a_k cos(2 pi f_k t) + b_k sin(2 pi f_k t)]
```

Then it computes residuals:

```text
r(t) = y(t) - y_model(t)
```

The residuals are searched again with Lomb-Scargle using the same frequency range selected in Frequency Search.

### Parameters

**Next period to remove**
Detected period selected for the next prewhitening step.

**Manual period for next step**
Manual period entered by the user. If provided, it overrides the dropdown selection.

**Next step**
Adds the selected period to the prewhitening chain and recomputes the residual periodogram.

Each prewhitening step is fitted globally with all periods currently in the chain. The result includes compact diagnostics: RMS reduction, residual Lomb-Scargle peak reduction, AIC, and BIC. These values help decide whether adding another period genuinely improves the model or merely adds flexibility.

**Show model**
Displays the original data, the prewhitening model evaluated at the data points, a full continuous model, and the residuals.

**Show errors**
Toggles error bars in the model and residual plots.

**Clear chain**
Removes all prewhitening steps and restores the analysis without prewhitening.

## Period Tomography

Period Tomography explores whether the strength or location of a periodic signal changes with time.

The app implements two exploratory methods.

### v1 Sliding LS

The light curve is divided into moving time windows. In each window the Lomb-Scargle periodogram is computed. The result is shown as a time-period map, with period on one axis, time on the other, and color representing power or amplitude.

This is useful for asking questions such as:

- Is the period stable?
- Does the signal appear only during part of the data set?
- Does the best period drift with time?

### v2 WWZ

WWZ stands for Weighted Wavelet Z-transform. It is a time-frequency method often used for unevenly sampled astronomical time series. The idea is to fit localized sinusoidal models whose weights decrease away from the chosen time center.

A simplified weight is

```text
w_i = exp(-c omega^2 (t_i - tau)^2)
```

where `tau` is the time center, `omega` is the angular frequency, and `c` controls the time-frequency tradeoff.

### Parameters

**Min. frequency / Max. frequency**
Frequency interval used for the tomography map.

**Period bins**
Number of period values displayed in the map.

**Window width**
Width of each sliding time window.

**Window step**
Spacing between consecutive window centers.

**Min. points per window**
Windows with fewer points are skipped.

**Color metric**
For v1, color can represent Lomb-Scargle power or fitted amplitude.

**Keep best-period track near a detected period**
Restricts the displayed best-period track to a region around a selected period. This is useful when the global strongest peak jumps to aliases or sampling features.

**Track reference period**
Detected period around which the best-period track is constrained.

**Track half-width**
Allowed fractional period range around the selected reference period.

**WWZ decay**
Controls the localization of the WWZ transform. Larger values give more local time resolution but poorer frequency resolution.

**Show best period track**
Displays or hides the best-period curve on top of the map.

## Model Laboratory

Model Laboratory contains specialized phenomenological models. These blocks are intended to help interpret the signal shape, test simple physical ideas, and provide educational diagnostics.

## Fourier Multi-Harmonic Model

This model represents a periodic profile as a Fourier series:

```text
y(phi) = C + sum_k [a_k cos(2 pi k phi) + b_k sin(2 pi k phi)]
```

It is useful for non-sinusoidal but strictly periodic shapes, such as asymmetric orbital modulation, broad pulses, double-peaked profiles, and harmonic-rich signals.

### Parameters

**Model period**
Period used to fold the data and define phase.

**T0**
Reference epoch for phase zero.

**Display phase bins**
Number of bins used in the displayed folded profile.

**Harmonic selection**
Manual uses the selected number of harmonics. AIC and BIC test several harmonic orders and choose the one with the best information criterion.

The information criteria are approximately

```text
AIC = n ln(RSS/n) + 2 k
BIC = n ln(RSS/n) + k ln(n)
```

where `n` is the number of points, `k` is the number of free parameters, and `RSS` is the residual sum of squares.

**Manual harmonics**
Number of harmonics fitted when manual selection is active.

**Max harmonics for AIC/BIC**
Maximum harmonic order tested by automatic selection.

**Fourier fit method**
Least-squares method used for the Fourier coefficients.

**Show full data set with model**
Displays the model in the original time domain.

## Eclipsing / Eccentric Binary Models

This block provides two simple models for binary-like folded profiles.

## Binary Interpretation Assistant

This Model Laboratory option is a compact morphological diagnostic for folded profiles. It marks the two strongest separated minima, estimates their phase separation, computes a depth-ratio proxy, checks whether the separation is close to 0.5, and flags possible half-period ambiguity using the harmonic diagnostics. It is intended as a guide before trying a more physical eclipsing-binary model, not as a unique binary solution.

### Eccentric Harmonic Model

The phase is mapped into true anomaly through Kepler's equation:

```text
M = 2 pi phase
M = E - e sin(E)
nu = 2 atan( sqrt((1+e)/(1-e)) tan(E/2) )
```

The profile is then modeled as

```text
y(t) = C + sum_k [a_k cos(k nu(t,e)) + b_k sin(k nu(t,e))]
```

This is not a full physical binary solution, but it introduces the idea that eccentric motion changes the mapping between time and orbital angle.

### Empirical Eclipse Model

The empirical eclipse model uses Gaussian-like eclipse dips or peaks in phase:

```text
y(phi) = C + E_c cos(4 pi phi) + E_s sin(4 pi phi)
       + D_1 exp[-0.5 d(phi,phi_1)^2 / w_1^2]
       + D_2 exp[-0.5 d(phi,phi_2)^2 / w_2^2]
```

where `d(phi, phi_i)` is the circular phase distance from eclipse center.

The depth ratio and width ratio are useful pedagogically. They may suggest brightness or size contrasts, but they do not by themselves determine mass ratio, radius ratio, or inclination without a physical binary model and external constraints.

### Physical Eclipse Toy Model

The physical eclipse toy model is a deliberately simplified alternative to full binary-light-curve solvers such as PHOEBE or ellc. It is intended for teaching: students can change physical parameters and immediately see how the folded profile reacts.

The model first uses Kepler's third law to estimate the semi-major axis:

```text
a^3 = G (M1 + M2) (P / 2 pi)^2
```

The stellar radii entered in solar radii, either manually or through the spectral-type presets, are converted to fractional radii `R1/a` and `R2/a`. The eccentric orbit is described by

```text
r/a = (1 - e^2) / (1 + e cos nu)
theta = nu + omega
d_proj/a = (r/a) sqrt[cos^2(theta) + sin^2(theta) cos^2(i)]
```

where `d_proj` is the projected separation of the two stellar disks. The eclipse depth is estimated from the overlap area of two circles, scaled by a simple surface-brightness proxy. If stellar luminosities are available, the toy model uses

```text
S2 / S1 ~ (L2 / R2^2) / (L1 / R1^2)
```

where `S` is a disk-averaged surface-brightness proxy. For manual exploration, luminosity may be entered directly. The spectral-type presets provide approximate `M`, `R`, `L`, and `Teff` values for OB stars and for A, F, G, K, and M stars in luminosity classes V, III, and I. The temperature-only blackbody scaling,

```text
S2 / S1 ~ (T2 / T1)^4
```

is still reported as a diagnostic, but luminosity and radius control the displayed physical eclipse proxy.

The displayed fit is

```text
y(phi) = C + A Q(phi; e, omega, i, R1/a, R2/a, L2/L1, S2/S1, u1, u2, L3)
```

where `Q` is the geometric eclipse proxy. Optional ellipsoidal, reflection, and beaming terms can be added as simple sinusoidal proxies:

```text
Q = Q_eclipse + E cos(4 pi phi) + R cos(2 pi phi) + B sin(2 pi phi)
```

Only the vertical offset `C` and scale `A` are fitted automatically. The physical parameters are user-controlled and should be interpreted as pedagogical diagnostics unless a proper physical solver and external constraints are used. The preset values are intentionally approximate: they help students see how mass changes the orbital scale, how radius changes eclipse probability and duration, and how luminosity/temperature changes eclipse depth.

### Parameters

**Binary period**
Period used to fold the binary light curve.

**T0 / periastron epoch**
Reference epoch. For eccentric harmonic mode, it is interpreted as a periastron epoch. For empirical eclipses, it is simply the phase-zero reference.

**Display phase bins**
Number of bins used in the folded display.

**Binary model**
Selects eccentric harmonic, empirical eclipses, or the physical eclipse toy model.

**True-anomaly harmonics**
Number of harmonic terms in true anomaly.

**Initial/fixed eccentricity**
Initial eccentricity if fitting is enabled, or fixed eccentricity if fitting is disabled.

**Fit eccentricity**
Optimizes eccentricity between 0 and 0.9.

**Include secondary eclipse**
Adds a second eclipse component.

**Primary phase guess / Secondary phase guess**
Optional starting phases for empirical eclipse centers.

**Eccentricity / Omega / Inclination**
Physical eclipse toy parameters controlling the orbital shape, orientation of periastron, and viewing angle.

**M1 / M2**
Component masses. They set the semi-major axis through Kepler's law and therefore change the fractional radii for fixed input radii.

**Primary / Secondary spectral type**
Optional stellar presets for the two components. Choose `Manual` to edit physical quantities directly, or choose approximate spectral subclasses from O, B, A, F, G, K, and M.

**Primary / Secondary luminosity class**
Luminosity class for the preset: V, III, or I. This changes the typical stellar mass, radius, luminosity, and effective temperature.

**R1 / R2**
Component radii in solar radii. Larger fractional radii produce broader and more likely eclipses.

**T1 / T2**
Effective temperatures. They are displayed and used as a blackbody diagnostic; with presets or entered luminosities, the model surface-brightness proxy is mainly driven by `L/R^2`.

**L1 / L2**
Bolometric luminosities in solar luminosities. They set the relative surface-brightness proxy through `(L2/R2^2)/(L1/R1^2)`, which affects the relative eclipse depths.

**u1 / u2**
Linear limb-darkening coefficients. Larger values reduce the disk-averaged occulted surface brightness in this simplified implementation.

**Third light**
Additional constant light as a fraction of the two-star flux. It dilutes eclipse amplitudes.

**Ellipsoidal / Reflection / Beaming**
Optional teaching terms that add approximate double-wave ellipsoidal modulation, single-wave reflection modulation, and sine-like Doppler-beaming modulation.

## Bondi-Hoyle Accretion Model

The Bondi-Hoyle block explores wind-fed accretion in a binary system. It can
operate as a normalized teaching model or calculate a physical mass-accretion
rate and luminosity. The app reports validity diagnostics because an analytic
BHL prescription is not a hydrodynamic or radiative-transfer simulation.

### Orbital and wind geometry

The orbital separation is calculated from a Keplerian eccentric orbit. The
compact object's velocity is resolved into radial and tangential components:

```text
r/a = (1 - e^2) / (1 + e cos(nu))
v_orb,r / v_o = e sin(nu) / sqrt(1 - e^2)
v_orb,t / v_o = (1 + e cos(nu)) / sqrt(1 - e^2)
```

where `v_o = 2 pi a / P`. Since the stellar wind is radial, the relative
velocity is

```text
v_rel^2 = (v_w - v_orb,r)^2 + v_orb,t^2
```

This vector expression is essential for eccentric orbits. The simpler
`v_w^2 + v_orb^2` expression is recovered for a circular orbit, where the
orbital velocity is tangential.

The wind follows

```text
v_w(r) = v_inf (1 - R_donor / r)^beta
rho_w = Mdot_w / (4 pi r^2 v_w)
```

The app rejects an orbit that intersects the donor rather than hiding it with
an artificial minimum wind velocity.

### Accretion formulations

The classical pressure-aware Bondi-Hoyle rate is

```text
Mdot_BH = 4 pi G^2 M_compact^2 rho_w /
          (v_rel^2 + c_s^2)^(3/2)
```

Setting `c_s=0` gives the pressure-free, high-Mach Hoyle-Lyttleton limit.

The revised binary formulation includes the orientation of the accretion
cylinder relative to the radial wind:

```text
R_acc = 2 G M_compact / (v_rel^2 + c_s^2)
eta = 1/4 |1 - v_orb,r/v_w| (R_acc/r)^2
```

For a circular orbit it reduces to

```text
eta = [q / (1 + w^2)]^2
q = M_compact / (M_donor + M_compact)
w = v_w / v_o
```

This formulation is particularly useful for identifying the classical
over-capture problem in slow winds. It is presented as a recent analytic
binary prescription, not as a replacement for hydrodynamic simulation in all
regimes.

### Normalized and physical modes

Normalized teaching mode median-normalizes the accretion curve and fits

```text
y(t) = C + A proxy_BHL(t)
```

Its amplitude and sign are phenomenological. A negative `A` triggers a warning
because observed count rate may be dominated by absorption or reprocessing.

Physical mode requires `v_inf`, component masses, donor radius, donor mass-loss
rate, and an accretor luminosity prescription. It reports density, `Mdot_acc`,
luminosity, and Eddington ratio. An Eddington cap is optional and is never
applied silently.

### Observable transfer

Intrinsic accretion and observed count rate are kept separate. The optional
attenuation layer uses

```text
F_observed = F_intrinsic exp[-tau_eff N_H,proxy(phase)]
```

This is a phenomenological wind-column proxy, not energy-dependent radiative
transfer. A phase delay or causal exponential response can be applied after
the intrinsic curve has been computed. It does not change `T0` or move the
orbital geometry.

### Parameters

**Orbital period / T0**
Set the orbital phase and periastron epoch. `T0` remains geometrical even when
a response delay is fitted.

**Initial/fixed eccentricity**
Initial or fixed eccentricity. The allowed upper bound is reduced
automatically when necessary to keep the donor inside the periastron
separation.

**Wind speed input**
Choose terminal speed `v_inf` in km/s or the dimensionless terminal ratio
`w_inf = v_inf/v_o`. `w_inf` is not the instantaneous local wind/orbital speed
ratio.

**Donor radius / a**
Dimensionless wind-launching radius in terminal-ratio mode.

**Sound speed**
Entered in km/s for physical wind input or as `c_s/v_o` in dimensionless mode.
The displayed Mach number uses the local relative velocity.

**Accretion formulation**
Selects classical pressure-aware BHL or the revised binary geometric
efficiency.

**Calculation mode**
Selects a normalized teaching curve or physical mass rate and luminosity.
Physical mode requires `v_inf` input.

**Mass-loss rate / compact radius / radiative efficiency**
Set the physical density, accretion rate, and luminosity normalization.

**Response model**
Choose no response, a pure phase delay, or a causal exponential response.

**Effective attenuation tau**
Controls the optional wind-column attenuation proxy.

**Residual bootstrap iterations**
Estimates confidence intervals by refitting residual-resampled light curves.
Use zero for fast exploration and increase it for final analysis.

### Validity dashboard

The model reports:

- minimum and maximum Mach number;
- local and terminal wind ratios;
- maximum `R_acc/a` and `R_acc/r`;
- maximum capture efficiency;
- minimum separation in donor-radius units;
- periastron Roche-lobe filling factor;
- parameter-bound and negative-scale warnings;
- comparison with constant, sinusoidal, and two-harmonic models.

Status is `VALID`, `CAUTION`, or `OUTSIDE ASSUMPTIONS`. Slow winds, transonic
flow, a large accretion radius, Roche-lobe filling, or efficiency above unity
indicate that gradients, turbulence, stream accretion, or transient-disc
physics may dominate.

### Interpretation and literature

The analytic foundation and its limitations follow the Bondi-Hoyle-Lyttleton
review by Edgar. The revised geometric efficiency follows Tejeda and Toala.
The gradient/turbulence warnings are motivated by the supergiant X-ray binary
simulations of Xu and Stone. The intrinsic-versus-observed distinction is
important in systems such as Vela X-1, where obscuration and transient
disc-like structures can change the observed count rate.

Relativistic shock-cone and QPO calculations around Kerr, Horndeski, or
Lee-Wick black holes are outside this Newtonian wind-fed binary module.

## X-ray Pulsation Timing

This block is designed for pulsating X-ray sources. It combines pulse-period refinement, pulse-profile modeling, pulsed-fraction estimates, and pulse-arrival-time diagnostics.

### Epoch Folding

Epoch folding tests trial pulse periods by folding the data and measuring how strongly the folded profile deviates from a constant value.

For each trial period:

```text
phase_i = ((t_i - T0) / P_trial) mod 1
```

The data are binned in phase. A simple statistic is

```text
chi2_epoch = sum_j [(mean_j - global_mean) / error_j]^2
```

The best trial period is the one with the largest epoch-folding statistic.

The app reports the number of degrees of freedom, a single-trial chi-square probability, a conservative trials-corrected false-alarm probability, and an empirical permutation false-alarm probability. Residual bootstrap realizations provide a Monte Carlo period interval. A maximum at the search boundary is flagged because the period range should then be widened.

### Pulse Profile Model

The pulse profile is modeled as a Fourier series:

```text
y(phi) = C + sum_k [a_k cos(2 pi k phi) + b_k sin(2 pi k phi)]
```

This allows broad, asymmetric, and multi-peaked pulse shapes.

### Pulsed Fraction

After subtracting the user-supplied background `B`, the peak-to-peak pulsed fraction is estimated from the fitted profile:

```text
PF_peak_to_peak = (F_max - F_min) / (F_max + F_min - 2 B)
```

The RMS pulsed fraction is approximated from Fourier amplitudes:

```text
PF_rms = sqrt( sum_k max[0, a_k^2 + b_k^2 - var(a_k) - var(b_k)] / 2 ) / (C - B)
```

Coefficient Monte Carlo draws provide 16th–84th percentile intervals. These definitions require a positive background-subtracted mean; otherwise the app suppresses the result and shows a warning.

### Pulse Arrival Times

The data are divided into time segments. In each segment, a global, leave-one-segment-out, or highest-S/N template is shifted in phase until it best matches the segment data. A profile-likelihood scan supplies the shift uncertainty. The phase shift gives an arrival-time offset:

```text
Delta t = Delta phase * P_pulse
```

The O-C value is then

```text
O-C = observed arrival time - calculated arrival time
```

The integer cycle nearest each segment centre defines the calculated arrival. The observed value is reported as a **candidate template phase-zero TOA**. This cycle assignment is an assumption, not proof of phase connection.

### Spin Ephemeris

Candidate TOAs are fitted with

```text
phi(t) = phi0 + F0 (t - Tref) + 0.5 F1 (t - Tref)^2
```

The app compares an `F0` model with `F0+F1` using BIC and reports residuals and a phase-connection diagnostic. Even a favourable diagnostic remains provisional until the integer cycle count and barycentric provenance are independently verified.

### Orbital Clues From O-C

If the compact object is in a binary, orbital motion can delay or advance pulse arrival times through light-travel time. For a circular orbit, the first-order O-C curve is approximately sinusoidal:

```text
O-C(t) = C + A_c cos(2 pi (t-Tref)/P_orb)
           + A_s sin(2 pi (t-Tref)/P_orb)
```

The O-C amplitude is a proxy for

```text
a_x sin(i) / c
```

where `a_x` is the compact object's projected orbital semimajor axis and `i` is inclination. The app can alternatively fit the Keplerian Roemer-delay form using orbital period, periastron epoch, eccentricity, longitude of periastron, and projected light-travel time. These fits operate on residuals after the spin ephemeris.

### Parameters

**Pulse period**
Initial period used for the pulse search.

**T0**
Reference epoch for pulse phase.

**Profile bins**
Number of bins in the displayed pulse profile.

**Profile harmonics**
Number of Fourier harmonics used to model the pulse shape.

**Epoch search half-width [%]**
Fractional search range around the input pulse period.

**Trial periods**
Number of trial periods evaluated by epoch folding.

**Epoch-folding bins**
Number of phase bins used in the epoch-folding search.

**Pulse-arrival segments**
Number of time segments used for arrival-time estimation.

**Min points per segment**
Segments with fewer points are skipped.

**TOA template**
Selects a global Fourier template, a leave-one-segment-out template that reduces self-matching bias, or a template from the highest-S/N segment.

**Monte Carlo iterations**
Controls period bootstrap, permutation significance, and pulsed-fraction uncertainty calculations. Zero disables Monte Carlo calculations.

**Background level**
Background subtracted when calculating pulsed fractions.

**Test frequency derivative F1**
Allows the quadratic spin term and retains it only when it improves BIC sufficiently.

**O-C orbital model / trial orbital period**
Selects no orbital fit, a circular delay, or a Keplerian delay. Local period fitting is constrained to the selected search width.

**Pulse-shape fit method**
Least-squares method used for the pulse-profile Fourier model.

**Show full data set with model**
Displays the pulse model over the full data set.

## Data Downloads, Evidence, and PDF Reports

Several result blocks include download buttons for the plotted data. These export plain text tables with the numerical values used in the displayed figures, so the same periodograms, sampling-window curves, folded profiles, model curves, epoch-folding searches, or pulse-arrival diagnostics can be inspected outside the app.

### Student Evidence Notebook

The notebook groups the required evidence into the five workflow stages described above. A progress indicator and status table show which stages are complete and what evidence is still missing. Closing an expander does not erase its contents; the notes are cleared only when the workspace is reset.

For a reliable periodicity, the frequency stage requires a selected frequency or period, a reasoned selection, and harmonic analysis. The folded-profile stage then becomes mandatory. For a justified non-periodic or inconclusive conclusion, the folded stage is automatically treated as not required.

The voluntary block uses a multiple-selection control for `None`, Iterative Prewhitening, Tomography, and Model Laboratory. It records the prewhitening periods and residual interpretation, tomography settings and interpretation, and/or the justification for one currently fitted Model Laboratory model.

### PDF Report Builder

The report panel creates a compact PDF report from the current workspace. Its required core always includes:

- the input data overview and original light curve,
- workflow completion and pending-evidence status,
- initial light-curve commentary,
- bibliographic reviews,
- the Lomb-Scargle and folded-profile plots,
- the folded-fit equation and detected-period table,
- the periodicity decision, selected frequency or period, and harmonic analysis,
- the conditional folded-profile interpretation, and
- the final assessment and comparison with the bibliography.

Iterative Prewhitening, Period Tomography, and one currently fitted Model Laboratory model are voluntary sections. They may be combined, and their report checkboxes remain unavailable until the corresponding result exists.

The report metadata fields are:

**Object name**
Name of the astronomical source or target being analyzed.

**Student name**
Name of the student or group preparing the report.

**Course**
Course, lab session, or activity name.

**Output PDF file name**
Name of the generated PDF file.

**Previous PDF report**
Optional PDF file. If supplied, the newly generated report is appended after the pages of the uploaded PDF. This is useful when a student wants to build a multi-part report over several analysis sessions.

**Sections to include**
The required core-workflow checkbox is fixed on. Separate voluntary controls add Iterative Prewhitening, Period Tomography, and one Model Laboratory model. When required evidence is missing, the download is labelled as a draft; when all five stages are complete, it is labelled as a final report.

The PDF report is a teaching/reporting aid. It preserves the main plots, fitted formulae, tables, and evidence text, but it should still be checked by the student before submission.

Every generated page includes the VIU logo in the upper-right corner and a footer separated by a horizontal rule. The footer shows `12MAST AG2` on the left and `Student name - Course` on the right, using the report metadata fields.

## Interpreting Results Safely

Strong periodogram peaks should be checked against the sampling window. Harmonics should be considered separately from independent periods. Prewhitening can reveal weaker signals, but it can also create misleading residuals if the removed model is not appropriate. Period tomography is useful for non-stationary signals, but maps can be sensitive to window size and data gaps.

Model Laboratory fits are deliberately simple and educational. They are useful for testing ideas, plotting interpretable diagnostics, and developing intuition. For final scientific inference, especially in eccentric binaries, eclipsing systems, wind accretion, and pulse-timing orbital solutions, the simplified fits should be followed by a dedicated physical model and uncertainty analysis.
