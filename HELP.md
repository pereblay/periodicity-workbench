# Overview

Periodicity Workbench is an interactive tool for exploring periodic signals in unevenly sampled light curves. It was designed around the workflow that often appears in optical and X-ray time-series analysis: load a light-curve table, search for periods with a Lomb-Scargle periodogram, check whether peaks may be caused by the sampling pattern, fold the data, model the folded profile, remove known signals by prewhitening, and then explore more specialized physical or pedagogical models.

The tool is intentionally exploratory. It can help identify candidate periods, aliases, harmonics, sampling-window artefacts, time-dependent period changes, and simple phenomenological models. It should not be treated as a replacement for a final publication-grade physical solution. The results are best used as a guide for deciding what deserves a more careful analysis.

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

**Minimum considered period**
Peaks with periods shorter than this value are not considered as primary candidates. This is useful for suppressing known short-period aliases, such as daily sampling features.

**Sampling-window threshold**
Minimum sampling-window power required for a window peak to be considered important.

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

The uncertainty block estimates period and frequency errors using bootstrap resampling.

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

Manual exclusions allow the user to prevent selected periods from being used as the primary period. This is useful when a peak is clearly due to sampling, aliasing, instrumental effects, or a known contaminating signal.

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

**Minimum points per window**
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

The Bondi-Hoyle block is a toy model for wind-fed X-ray binaries. It tests whether orbital changes in separation and relative velocity can produce the observed modulation.

The qualitative accretion-rate scaling is

```text
Mdot_acc ~ rho / v_rel^3
```

where `rho` is the local wind density and `v_rel` is the relative velocity between the compact object and the wind.

The toy implementation uses

```text
rho ~ 1 / (r^2 v_w)
v_rel^2 ~ v_w^2 + v_orb^2
y(t) = C + A [rho(r) / v_rel^3]
```

The separation `r` and orbital velocity are computed from a Keplerian orbit. The app can use either a dimensionless wind-speed ratio or a more pedagogical physical input based on terminal wind speed, masses, and stellar radius.

When physical wind input is selected, the orbital scale is estimated from Kepler's third law:

```text
a = [G (M_donor + M_compact) (P_orb / 2 pi)^2]^(1/3)
v_orb,scale = 2 pi a / P_orb
v_inf / v_orb = v_inf / v_orb,scale
```

The toy wind law then uses

```text
v_w(r) = v_inf * (1 - R_donor / r)^beta
```

implemented internally as a normalized ratio. In `v_inf` mode, `R_donor/a` is computed from the physical donor radius and the semi-major axis. In dimensionless `v_wind / v_orb` mode, `R_donor/a` is entered directly. The fitted parameters remain pedagogical and phenomenological unless physical masses, wind laws, inclination, and independent orbital constraints are available.

The donor spectral-type presets provide approximate stellar masses, radii, luminosities, and effective temperatures for quick classroom exploration. They include OB stars and representative A, F, G, K, and M stars. They are useful starting values, not a replacement for system-specific stellar calibrations.

### Parameters

**Orbital period**
Period used to compute orbital phase.

**T0 / periastron epoch**
Reference epoch for periastron.

**Display phase bins**
Number of bins in the displayed folded profile.

**Initial/fixed eccentricity**
Initial or fixed eccentricity.

**Wind speed input**
Selects whether the model uses a dimensionless `v_wind / v_orb` ratio or physical wind parameters based on `v_inf`.

**v_wind / v_orb**
Ratio of wind speed to characteristic orbital speed. This is the original dimensionless toy-model parameter.

**Donor radius / a**
Dimensionless donor radius used in the beta-law wind acceleration term when the wind speed is entered as `v_wind / v_orb`.

**v_inf [km/s]**
Terminal wind speed used to derive an effective `v_wind / v_orb` ratio.

**Donor spectral type / Luminosity class**
Optional preset for typical donor mass, radius, luminosity, and effective temperature. The available presets cover OB and representative A, F, G, K, and M stars for luminosity classes V, III, and I. Choose `Manual` to edit mass and radius directly.

**Donor mass [Msun]**
Mass of the donor star used in Kepler's third law.

**Compact mass [Msun]**
Mass of the compact object, usually around 1.4 Msun for a neutron star.

**Donor radius [Rsun]**
Radius used in the simple beta-law wind acceleration term.

**Wind beta**
Controls the simple wind acceleration law used in the toy model.

**Fit eccentricity**
Optimizes eccentricity.

**Fit wind speed**
Optimizes the wind-speed ratio in the dimensionless mode. It is disabled in `v_inf` mode so that the user can explore how physical input parameters change the model.

**Fit phase lag**
Allows a phase offset between the simple periastron reference and the observed modulation.

**Show full data set with model**
Displays the fitted model over the full time span.

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

### Pulse Profile Model

The pulse profile is modeled as a Fourier series:

```text
y(phi) = C + sum_k [a_k cos(2 pi k phi) + b_k sin(2 pi k phi)]
```

This allows broad, asymmetric, and multi-peaked pulse shapes.

### Pulsed Fraction

The peak-to-peak pulsed fraction is estimated from the fitted profile:

```text
PF_peak_to_peak = (F_max - F_min) / (F_max + F_min)
```

The RMS pulsed fraction is approximated from Fourier amplitudes:

```text
PF_rms = sqrt( sum_k (a_k^2 + b_k^2) / 2 ) / |C|
```

These definitions are most meaningful when the signal is a positive count rate or flux. They can be less physically meaningful for magnitudes or background-subtracted data near zero.

### Pulse Arrival Times

The data are divided into time segments. In each segment, the fitted pulse template is shifted in phase until it best matches the segment data. The phase shift gives an arrival-time offset:

```text
Delta t = Delta phase * P_pulse
```

The O-C value is then

```text
O-C = observed arrival time - calculated arrival time
```

where the calculated arrival assumes a constant pulse period.

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

where `a_x` is the compact object's projected orbital semimajor axis and `i` is inclination. In an eccentric orbit the O-C curve is not a pure sinusoid, and a Keplerian pulse-timing model is needed.

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

**O-C sinusoid period**
Optional trial orbital period used to fit a sinusoid to the O-C values.

**Pulse-shape fit method**
Least-squares method used for the pulse-profile Fourier model.

**Show full data set with model**
Displays the pulse model over the full data set.

## Interpreting Results Safely

Strong periodogram peaks should be checked against the sampling window. Harmonics should be considered separately from independent periods. Prewhitening can reveal weaker signals, but it can also create misleading residuals if the removed model is not appropriate. Period tomography is useful for non-stationary signals, but maps can be sensitive to window size and data gaps.

Model Laboratory fits are deliberately simple and educational. They are useful for testing ideas, plotting interpretable diagnostics, and developing intuition. For final scientific inference, especially in eccentric binaries, eclipsing systems, wind accretion, and pulse-timing orbital solutions, the simplified fits should be followed by a dedicated physical model and uncertainty analysis.
