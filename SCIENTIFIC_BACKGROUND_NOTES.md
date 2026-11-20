# Scientific Background Notes for Periodicity Workbench

This file is a private working draft for a future long-form explanation of the
scientific basis of Periodicity Workbench. It is intentionally more explanatory
than the in-app help and is not meant to be committed yet.

## Scope of the App

Periodicity Workbench is an exploratory time-series analysis tool aimed at light
curves from optical and X-ray astronomy. Its goal is not to provide final
publication-grade physical solutions, but to help users identify periodicities,
inspect sampling artefacts, test simple phenomenological models, and build
physical intuition before moving to dedicated analysis codes.

The app combines several layers of analysis:

- Lomb-Scargle period searches for unevenly sampled time series.
- Sampling-window analysis to identify aliases and cadence-driven artefacts.
- Bootstrap uncertainty estimates for selected frequency or period peaks.
- Phase folding and binned folded profiles.
- Sinusoidal, multi-harmonic, and prewhitening models.
- Time-frequency maps for non-stationary signals.
- Pedagogical model-laboratory blocks, including eccentric/eclipsing binaries,
  wind-fed Bondi-Hoyle accretion, and X-ray pulsation timing.

The central philosophy is to keep the models transparent enough for teaching:
each block exposes the assumptions, fitted quantities, and limitations.

## Eclipsing and Eccentric Binary Toy Model

### Purpose of the Block

The eclipsing/eccentric binary block is designed as a pedagogical bridge
between purely empirical folded-light-curve fitting and full binary modelling
codes such as `ellc`, PHOEBE, Wilson-Devinney-type implementations, or
dedicated radial-velocity plus photometry solvers.

The app should make clear that this is not a final physical binary solution.
Its purpose is to let students explore how orbital and stellar parameters
affect a folded light curve:

- Increasing inclination makes eclipses more likely and deeper.
- Increasing stellar radii broadens eclipses and increases eclipse probability.
- Increasing eccentricity changes the separation at conjunction and can shift
  the phase separation of eclipses away from exactly 0.5.
- Changing the argument of periastron changes which conjunction occurs closer
  to periastron.
- Changing luminosity ratio and surface brightness ratio changes which eclipse
  is deeper.
- Third light dilutes both eclipses.
- Simple ellipsoidal, reflection, and beaming terms can add out-of-eclipse
  modulation.

### Orbital Geometry

For a binary period `P`, component masses `M1` and `M2`, and a Keplerian orbit,
the semi-major axis is estimated from Kepler's third law:

```text
a^3 = G (M1 + M2) (P / 2 pi)^2
```

The user-entered stellar radii are converted to fractional radii:

```text
r1 = R1 / a
r2 = R2 / a
```

The eccentric orbit is represented through the true anomaly `nu`:

```text
r/a = (1 - e^2) / (1 + e cos nu)
```

The orientation of the ellipse is controlled by the argument of periastron
`omega`, and the projected separation of the two stellar disks is approximated
as

```text
theta = nu + omega
d_proj/a = (r/a) sqrt[cos^2(theta) + sin^2(theta) cos^2(i)]
```

where `i` is the inclination. Eclipses are possible when the projected
separation is smaller than the sum of the fractional radii:

```text
d_proj/a < R1/a + R2/a
```

A useful quick diagnostic is

```text
cos(i) < R1/a + R2/a
```

for near-circular orbits and favorable conjunction geometry. In eccentric
systems the instantaneous separation at conjunction also matters.

### Eclipse Depth Proxy

The toy model treats the stars as circular disks. The occulted area is computed
from the geometric overlap of two circles:

```text
A_overlap = overlap_area(d_proj, R1/a, R2/a)
```

The model then estimates the lost light as

```text
Delta F = A_overlap * S_blocked
```

where `S_blocked` is a disk-averaged surface-brightness proxy for the star that
is being occulted. The implementation uses the sign of the line-of-sight
coordinate to decide which component is in front.

### Luminosity, Temperature, and Surface Brightness

The current version allows the user to set or load, for each component:

- Mass `M`
- Radius `R`
- Bolometric luminosity `L`
- Effective temperature `Teff`

The preferred surface-brightness proxy is now luminosity per projected stellar
area:

```text
S2 / S1 ~ (L2 / R2^2) / (L1 / R1^2)
```

This is still simplified, because a real observed eclipse depth depends on the
instrumental bandpass, spectral energy distributions, limb darkening,
gravity darkening, irradiation, extinction, and passband-specific surface
brightness. However, using `L/R^2` is more educational than using temperature
alone because students can see independently how radius and luminosity affect
the result.

The blackbody-like temperature relation is still a useful diagnostic:

```text
S2 / S1 ~ (T2 / T1)^4
```

and the corresponding bolometric luminosity scaling is

```text
L2 / L1 ~ (R2 / R1)^2 (T2 / T1)^4
```

The app reports this diagnostic separately from the luminosity-based ratio.

### Spectral-Type and Luminosity-Class Presets

The eclipsing-binary block now uses the same approximate stellar preset table
as the wind-accretion block, extended beyond OB stars. The purpose is
pedagogical: the user can select a rough spectral type and luminosity class and
immediately see how typical masses, radii, luminosities, and temperatures
change the geometry and depth of the model.

Available representative spectral subclasses include:

```text
O5, O7, O9,
B0, B1, B2, B3, B5, B8,
A0, A5,
F0, F5,
G0, G2, G5,
K0, K5,
M0, M5
```

Available luminosity classes are:

```text
V, III, I
```

The values are deliberately approximate and should be documented as classroom
starting points. They are not a substitute for calibrated stellar parameter
tables, Gaia constraints, spectroscopy, evolutionary tracks, or detailed SED
modelling.

### Fitted Equation

The physical eclipse toy model builds a dimensionless geometric proxy `Q(phi)`
and then fits the observed folded profile with

```text
y(phi) = C + A Q(phi)
```

where:

- `C` is the fitted vertical offset.
- `A` is the fitted scale factor in the observed flux, count-rate, or magnitude
  units.

The geometric proxy can include optional teaching terms:

```text
Q(phi) = Q_eclipse
       + E cos(4 pi phi)
       + R cos(2 pi phi)
       + B sin(2 pi phi)
```

Interpretation:

- `E cos(4 pi phi)` is a crude double-wave ellipsoidal proxy.
- `R cos(2 pi phi)` is a crude reflection/heating proxy.
- `B sin(2 pi phi)` is a crude Doppler-beaming proxy.

These terms are intentionally not full physical calculations. They are visual
and conceptual aids.

### What the Parameters Teach

**Masses**
Masses set the semi-major axis through Kepler's law. At fixed period and fixed
stellar radii, increasing total mass increases `a` and therefore decreases
`R/a`, making eclipses narrower or less likely.

**Radii**
Radii control eclipse duration and eclipse probability. Larger stars have
larger overlap areas and broader conjunction signatures.

**Luminosities**
Luminosities affect which eclipse is deeper. If the hotter/brighter star is
blocked, the lost light is larger.

**Temperatures**
Temperatures remain useful pedagogically because students expect hotter stars
to have higher surface brightness. The app reports temperature-ratio and
blackbody-ratio diagnostics for comparison.

**Inclination**
Inclination controls impact parameter. Near `90 deg`, eclipses are more likely.
At lower inclinations, the disks may never overlap.

**Eccentricity and omega**
Eccentricity changes the separation at conjunction. The argument of periastron
decides whether conjunction occurs closer to periastron or apastron, changing
eclipse width, depth, and phase separation.

**Third light**
Third light adds constant flux from an unresolved component or background
source. It dilutes all fractional eclipse depths.

### Limitations

The eclipsing-binary block has important limitations:

- Stars are treated as circular disks.
- The model does not solve Roche geometry.
- Limb darkening is represented only by a simple disk-averaged scaling.
- Gravity darkening is not physically computed.
- Reflection and beaming are sinusoidal proxies, not detailed calculations.
- Stellar spectra and passband response are not used.
- No radial velocities are fitted.
- The physical parameters are not automatically optimized, except for the
  global offset and scale factor.
- Parameter degeneracies are severe: inclination, radii, luminosity ratio,
  third light, and scale can mimic each other.

Recommended wording for future documentation:

```text
The physical eclipse model is a teaching model. It is useful for exploring
which physical ingredients can change eclipse width, depth, and phase, but it
should not be quoted as a final binary solution. For scientific inference, use
a dedicated binary solver and include uncertainties, priors, radial velocities
where possible, and passband-specific stellar atmospheres.
```

## Bondi-Hoyle Wind-Accretion Toy Model

### Physical Motivation

In a wind-fed high-mass X-ray binary, the compact object accretes material from
the stellar wind of the donor. In the simplest Bondi-Hoyle-Lyttleton picture,
the compact object gravitationally focuses material from a region whose
characteristic accretion radius is

```text
R_acc = 2 G M_X / v_rel^2
```

where:

- `M_X` is the compact-object mass.
- `v_rel` is the relative velocity between the compact object and the wind.

The approximate accretion rate is then

```text
Mdot_acc = pi R_acc^2 rho v_rel
```

Substituting `R_acc` gives

```text
Mdot_acc = 4 pi G^2 M_X^2 rho / v_rel^3
```

The accretion luminosity can be written schematically as

```text
L_acc = G M_X Mdot_acc / R_X
```

where `R_X` is the compact-object radius. Therefore

```text
L_acc = 4 pi G^3 M_X^3 rho / (R_X v_rel^3)
```

If `M_X` and `R_X` are fixed, the orbital modulation of the accretion
luminosity is controlled mainly by

```text
rho / v_rel^3
```

This is the quantity represented by the app's Bondi-Hoyle proxy.

### Proxy Used in the App

The app currently computes a dimensionless Bondi-Hoyle-like proxy:

```text
proxy_raw ~ rho / v_rel^3
```

with

```text
rho ~ 1 / (r^2 v_w)
v_rel^2 ~ v_w^2 + v_orb^2
```

Here:

- `r` is the instantaneous orbital separation in units of the semi-major axis.
- `v_w` is the local wind speed.
- `v_orb` is a dimensionless orbital-speed proxy.

The separation is computed from a Keplerian eccentric orbit:

```text
r/a = (1 - e^2) / (1 + e cos nu)
```

where:

- `e` is the eccentricity.
- `nu` is the true anomaly.

The dimensionless orbital-speed proxy follows the vis-viva scaling:

```text
v_orb^2 ~ 2/r - 1
```

The wind acceleration is represented through a simplified beta-law:

```text
v_w(r) = v_inf_or_ratio * (1 - R_donor / r)^beta
```

Inside the app this is implemented in dimensionless form:

```text
v_w = wind_speed_ratio * max(1 - (R_donor/a) / (r/a), 0.05)^beta
```

The floor avoids zero or negative wind speed close to the stellar surface,
which would otherwise produce numerical divergences.

### Donor Radius Improvement

Earlier toy versions used a fixed launch radius equivalent to

```text
R_donor/a = 0.15
```

The current implementation uses the donor radius explicitly:

- In `v_inf` mode, the app computes `R_donor/a` from the physical donor radius
  and the semi-major axis.
- In dimensionless `v_wind / v_orb` mode, the user enters `R_donor/a`
  directly.

The semi-major axis is estimated from Kepler's third law:

```text
a = [G (M_donor + M_compact) (P_orb / 2 pi)^2]^(1/3)
```

Then

```text
R_donor/a = R_donor / a
```

This makes the wind acceleration term more physically transparent: a larger
donor radius means the compact object samples wind that is less accelerated at
a given orbital separation, which can increase density and alter the predicted
orbital modulation.

### Normalization of the Proxy

The raw proxy has arbitrary units because it omits the absolute wind mass-loss
rate, compact-object radius, radiative efficiency, distance, absorption, and
instrument response. The app therefore normalizes it as

```text
proxy = proxy_raw / median(proxy_raw) - 1
```

This turns the curve into a relative modulation with

```text
median(proxy) = 0
```

Interpretation:

- `proxy > 0`: predicted accretion above the typical orbital level.
- `proxy < 0`: predicted accretion below the typical orbital level.

This normalization keeps the model focused on the shape and phase dependence of
the expected modulation, rather than pretending to predict an absolute
luminosity.

### From Proxy to Predicted Count Rate

The app fits the observed signal with a linear model:

```text
count_rate_model(t) = C + A * proxy(t)
```

where:

- `C` is the fitted vertical offset, approximately the mean observed level.
- `A` is the fitted scale factor that converts the dimensionless modulation into
  the observed units.

Thus the relation to the physical BHL luminosity is:

```text
proxy_raw proportional to Mdot_acc proportional to L_acc
```

but the app does not currently compute the absolute luminosity:

```text
L_acc = G M_X Mdot_acc / R_X
```

Instead, all missing absolute factors are absorbed into `A` and `C`.

The omitted absolute factors include:

- Donor wind mass-loss rate `Mdot_w`.
- Compact-object mass `M_X`, except indirectly in orbital-scale estimates.
- Compact-object radius `R_X`.
- Radiative efficiency.
- Source distance.
- Interstellar and local absorption.
- Instrumental conversion from flux to count rate.

This is why the model is best understood as a shape model:

```text
Does the expected orbital BHL modulation have maxima and minima in roughly the
right places, with a plausible width and asymmetry?
```

rather than as an absolute luminosity calculator.

### Spectral-Type Presets

The app includes approximate stellar presets for classroom use. A user can
select a donor spectral type and luminosity class, and the app fills typical
mass, radius, luminosity, and effective-temperature values.

Currently available spectral types:

```text
O5, O7, O9, B0, B1, B2, B3, B5, B8, A0, A5, F0, F5, G0, G2, G5, K0, K5, M0, M5
```

Currently available luminosity classes:

```text
V, III, I
```

These values are intentionally approximate. They should be described as
pedagogical starting points rather than calibrated stellar parameters.

Possible future documentation wording:

```text
The spectral-type presets are intended to help students explore how typical
stellar radius, mass, luminosity, and effective temperature change the orbital
scale, wind acceleration, eclipse depth, and eclipse geometry. They should not
be used as a substitute for system-specific stellar modelling, spectroscopic
classification, or calibrated evolutionary tracks.
```

### Limitations of the Bondi-Hoyle Toy Model

The current model is deliberately simplified. Its main limitations are:

- The wind is assumed spherical and smooth.
- The wind terminal velocity or wind-speed ratio is simplified.
- The orbital velocity and wind velocity are combined as
  `v_rel^2 ~ v_w^2 + v_orb^2`, without full vector geometry.
- The model does not include wind clumping.
- The model does not include photoionization feedback from the X-ray source.
- The model does not include accretion inhibition, magnetic gating, or
  propeller effects.
- The model does not include absorption changes along the line of sight.
- The model does not compute an absolute `L_acc` in erg/s.
- The observed count rate is fitted phenomenologically through `C + A * proxy`.

These limitations should be stated clearly in any scientific-background text.

## Possible Future Extension: Absolute BHL Luminosity

A later version could add an optional absolute-luminosity mode:

```text
rho(r) = Mdot_w / [4 pi r^2 v_w(r)]
Mdot_acc = 4 pi G^2 M_X^2 rho / v_rel^3
L_acc = eta G M_X Mdot_acc / R_X
F_X = L_acc / (4 pi d^2)
count_rate = conversion_factor * F_X
```

Additional parameters would be needed:

- Donor mass-loss rate `Mdot_w`.
- Compact-object mass `M_X`.
- Compact-object radius `R_X`.
- Accretion efficiency `eta`.
- Distance `d`.
- Absorption column or spectral conversion factor.
- Instrument response or an approximate count-rate conversion.

This would make the model more physically ambitious, but also much more
dependent on assumptions. For the current educational scope, the normalized
proxy is safer and easier to interpret.

## Notes for Future Scientific Background Text

Useful narrative structure:

1. Start with the observational problem: unevenly sampled light curves and
   periodic modulation.
2. Explain Lomb-Scargle as the search tool for candidate periods.
3. Explain sampling windows and aliases.
4. Explain phase folding and simple sinusoidal models.
5. Explain prewhitening as a residual search, with cautions.
6. Introduce model-laboratory blocks as pedagogical physical intuition tools.
7. Present Bondi-Hoyle as a shape model for wind-fed X-ray binaries.
8. Explicitly distinguish relative modulation from absolute luminosity.
9. Close with limitations and the need for dedicated modelling for final
   scientific inference.

## Bibliography and Reference Map

This section lists real references used to ground the scientific explanations
above. It is not yet formatted as a journal bibliography; it is a working map
from app modules to the literature that should support the final long-form
scientific documentation.

### Unevenly Sampled Period Searches

Used for the Lomb-Scargle, generalized Lomb-Scargle, frequency-grid,
false-alarm-probability, and bootstrap-significance discussion.

- Lomb, N. R. 1976, "Least-squares frequency analysis of unequally spaced
  data", Astrophysics and Space Science, 39, 447.
  DOI: https://doi.org/10.1007/BF00648343

- Scargle, J. D. 1982, "Studies in astronomical time series analysis. II.
  Statistical aspects of spectral analysis of unevenly spaced data", The
  Astrophysical Journal, 263, 835.
  ADS: https://ui.adsabs.harvard.edu/abs/1982ApJ...263..835S/abstract

- Horne, J. H., & Baliunas, S. L. 1986, "A prescription for period analysis of
  unevenly sampled time series", The Astrophysical Journal, 302, 757.
  ADS: https://ui.adsabs.harvard.edu/abs/1986ApJ...302..757H/abstract

- Zechmeister, M., & Kuerster, M. 2009, "The generalised Lomb-Scargle
  periodogram. A new formalism for the floating-mean and Keplerian
  periodograms", Astronomy & Astrophysics, 496, 577.
  arXiv: https://arxiv.org/abs/0901.2573

- VanderPlas, J. T. 2018, "Understanding the Lomb-Scargle Periodogram", The
  Astrophysical Journal Supplement Series, 236, 16.
  arXiv: https://arxiv.org/abs/1703.09824

- Baluev, R. V. 2008, "Assessing statistical significance of periodogram
  peaks", Monthly Notices of the Royal Astronomical Society, 385, 1279.
  arXiv: https://arxiv.org/abs/0711.0330

- Vio, R., Diaz-Trigo, M., & Andreani, P. 2013, "Irregular time series in
  astronomy and the use of the Lomb-Scargle periodogram", Astronomy &
  Astrophysics, 549, A32.
  arXiv: https://arxiv.org/abs/1301.4826

### Sampling Windows, Aliases, Prewhitening, and Time-Frequency Maps

Used for the sampling-window interpretation, alias cautions, iterative
prewhitening warnings, and period-tomography / WWZ-style sections.

- Roberts, D. H., Lehar, J., & Dreher, J. W. 1987, "Time Series Analysis with
  CLEAN. I. Derivation of a Spectrum", The Astronomical Journal, 93, 968.
  ADS: https://ui.adsabs.harvard.edu/abs/1987AJ.....93..968R/abstract

- Foster, G. 1996, "Wavelets for period analysis of unevenly sampled time
  series", The Astronomical Journal, 112, 1709.
  ADS: https://ui.adsabs.harvard.edu/abs/1996AJ....112.1709F/abstract

- Foster, G. 1995, "The CLEANest Fourier Spectrum", The Astronomical Journal,
  109, 1889.
  ADS: https://ui.adsabs.harvard.edu/abs/1995AJ....109.1889F/abstract

- Schwarzenberg-Czerny, A. 1989, "On the advantage of using analysis of
  variance for period search", Monthly Notices of the Royal Astronomical
  Society, 241, 153.
  ADS: https://ui.adsabs.harvard.edu/abs/1989MNRAS.241..153S/abstract

- Andronov, I. L., & Naumova, A. V. 2013, "Program WWZ: Wavelet Analysis of
  Astronomical Signals With Irregularly Spaced Arguments".
  arXiv: https://arxiv.org/abs/1310.5031

### Eclipsing and Eccentric Binary Models

Used for the pedagogical eclipse geometry, cautionary notes about simplified
models, references to proper binary solvers, and the statement that the app
does not replace full physical modelling.

- Wilson, R. E., & Devinney, E. J. 1971, "Realization of accurate close-binary
  light curves: application to MR Cygni", The Astrophysical Journal, 166, 605.
  ADS: https://ui.adsabs.harvard.edu/abs/1971ApJ...166..605W/abstract

- Wilson, R. E. 1979, "Eccentric orbit generalization and simultaneous solution
  of binary star light and velocity curves", The Astrophysical Journal, 234,
  1054.
  ADS: https://ui.adsabs.harvard.edu/abs/1979ApJ...234.1054W/abstract

- Wilson, R. E. 1994, "Binary-star light-curve models", Publications of the
  Astronomical Society of the Pacific, 106, 921.
  ADS: https://ui.adsabs.harvard.edu/abs/1994PASP..106..921W/abstract

- Prsa, A., & Zwitter, T. 2005, "A Computational Guide to Physics of Eclipsing
  Binaries. I. Demonstrations and Perspectives", The Astrophysical Journal,
  628, 426.
  arXiv: https://arxiv.org/abs/astro-ph/0503361

- Prsa, A., & Zwitter, T. 2006, "Disentangling effective temperatures of
  individual eclipsing binary components by means of color-index constraining",
  The Astrophysical Journal, 647, 1461.
  arXiv: https://arxiv.org/abs/astro-ph/0602205

- Maxted, P. F. L. 2016, "ELLC: a fast, flexible light curve model for detached
  eclipsing binary stars and transiting exoplanets", Astronomy & Astrophysics,
  591, A111.
  arXiv: https://arxiv.org/abs/1603.08484

- Mandel, K., & Agol, E. 2002, "Analytic Light Curves for Planetary Transit
  Searches", The Astrophysical Journal Letters, 580, L171.
  arXiv: https://arxiv.org/abs/astro-ph/0210099

### Stellar Parameter Presets

Used for the caveat that spectral-type presets are approximate and educational,
and as literature context for masses, radii, luminosities, and effective
temperatures assigned from spectral type and luminosity class.

- Martins, F., Schaerer, D., & Hillier, D. J. 2005, "A new calibration of
  stellar parameters of Galactic O stars", Astronomy & Astrophysics, 436, 1049.
  arXiv: https://arxiv.org/abs/astro-ph/0503346

- Pecaut, M. J., & Mamajek, E. E. 2013, "Intrinsic Colors, Temperatures, and
  Bolometric Corrections of Pre-main-sequence Stars", The Astrophysical Journal
  Supplement Series, 208, 9. The paper also provides a modern main-sequence
  dwarf color and temperature sequence.
  arXiv: https://arxiv.org/abs/1307.2657

- Eker, Z., Soydugan, F., Soydugan, E., Bilir, S., Yaz Gokce, E., Steer, I.,
  Tuysuz, M., Senyuz, O., & Demircan, O. 2015, "Main-Sequence Effective
  Temperatures from a Revised Mass-Luminosity Relation Based on Accurate
  Properties", The Astronomical Journal, 149, 131.
  arXiv: https://arxiv.org/abs/1501.06585

- Cox, A. N. 2000, Allen's Astrophysical Quantities, 4th edition, Springer.
  Used as a general reference point for standard stellar quantities and
  constants. Book page: https://link.springer.com/book/9780387987460

### Bondi-Hoyle, Wind Accretion, and Stellar Winds

Used for the BHL accretion-rate scaling, the distinction between relative
proxy and absolute luminosity, and the beta-law wind acceleration term.

- Hoyle, F., & Lyttleton, R. A. 1939, "The effect of interstellar matter on
  climatic variation", Mathematical Proceedings of the Cambridge Philosophical
  Society, 35, 405.
  DOI: https://doi.org/10.1017/S0305004100021150

- Bondi, H., & Hoyle, F. 1944, "On the Mechanism of Accretion by Stars",
  Monthly Notices of the Royal Astronomical Society, 104, 273.
  ADS: https://ui.adsabs.harvard.edu/abs/1944MNRAS.104..273B/abstract

- Bondi, H. 1952, "On spherically symmetrical accretion", Monthly Notices of
  the Royal Astronomical Society, 112, 195.
  ADS: https://ui.adsabs.harvard.edu/abs/1952MNRAS.112..195B/abstract

- Castor, J. I., Abbott, D. C., & Klein, R. I. 1975, "Radiation-driven winds in
  Of stars", The Astrophysical Journal, 195, 157.
  ADS: https://ui.adsabs.harvard.edu/abs/1975ApJ...195..157C/abstract

- Lamers, H. J. G. L. M., & Cassinelli, J. P. 1999, Introduction to Stellar
  Winds, Cambridge University Press.
  Publisher page: https://www.cambridge.org/core/books/introduction-to-stellar-winds/0D28596F360D0C78B04F36B376951EDF

- Edgar, R. 2004, "A review of Bondi-Hoyle-Lyttleton accretion", New Astronomy
  Reviews, 48, 843.
  arXiv: https://arxiv.org/abs/astro-ph/0406166

- Blondin, J. M., Stevens, I. R., & Kallman, T. R. 1991, "Hydrodynamic
  simulations of stellar wind disruption by a compact X-ray source", The
  Astrophysical Journal, 371, 684.
  ADS: https://ui.adsabs.harvard.edu/abs/1991ApJ...371..684B/abstract

### X-ray Pulsation, Epoch Folding, and Arrival Times

Used for the pulse-period-analysis block, epoch-folding idea, pulsed-fraction
discussion, and the planned use of pulse arrival times to detect orbital
Doppler delays.

- Leahy, D. A., Darbro, W., Elsner, R. F., Weisskopf, M. C., Kahn, S.,
  Sutherland, P. G., & Grindlay, J. E. 1983, "On searches for pulsed emission
  with application to four globular cluster X-ray sources", The Astrophysical
  Journal, 266, 160.
  ADS: https://ui.adsabs.harvard.edu/abs/1983ApJ...266..160L/abstract

- Buccheri, R., Bennett, K., Bignami, G. F., Bloemen, J. B. G. M.,
  Boriakoff, V., Caraveo, P. A., Hermsen, W., Kanbach, G., Manchester, R. N.,
  Masnou, J. L., Mayer-Hasselwander, H. A., Ozel, M. E., Paul, J. A.,
  Sacco, B., Scarsi, L., & Strong, A. W. 1983, "Search for pulsed gamma-ray
  emission from radio pulsars in the COS-B data", Astronomy & Astrophysics,
  128, 245.
  ADS: https://ui.adsabs.harvard.edu/abs/1983A%26A...128..245B/abstract

- Blandford, R., & Teukolsky, S. A. 1976, "Arrival-time analysis for a pulsar
  in a binary system", The Astrophysical Journal, 205, 580.
  ADS: https://ui.adsabs.harvard.edu/abs/1976ApJ...205..580B/abstract

- Taylor, J. H., & Weisberg, J. M. 1982, "A new test of general relativity:
  gravitational radiation and the binary pulsar PSR 1913+16", The Astrophysical
  Journal, 253, 908.
  ADS: https://ui.adsabs.harvard.edu/abs/1982ApJ...253..908T/abstract

- Weisberg, J. M., & Huang, Y. 2016, "Relativistic Measurements from Timing the
  Binary Pulsar PSR B1913+16", The Astrophysical Journal, 829, 55.
  arXiv: https://arxiv.org/abs/1606.02744

- Ransom, S. M., Eikenberry, S. S., & Middleditch, J. 2002, "Fourier
  techniques for very long astrophysical time-series analysis", The
  Astronomical Journal, 124, 1788.
  arXiv: https://arxiv.org/abs/astro-ph/0204349

### Implementation and Numerical Context

Used for the app implementation context, especially standard scientific Python
periodogram tooling and the warning that practical LS behavior depends on
normalization, centering, weights, frequency grid, and assumptions.

- Astropy Collaboration, Price-Whelan, A. M., Lim, P. L., Earl, N., Starkman,
  N., Bradley, L., Shupe, D. L., Patil, A. A., Corrales, L., Brasseur, C. E.,
  Nothaft, E., Donath, A., Tollerud, E., Morris, B. M., Ginsburg, A.,
  Vaher, E., Weaver, B. A., Tocknell, J., Jamieson, W., et al. 2022, "The
  Astropy Project: Sustaining and Growing a Community-oriented Open-source
  Project and the Latest Major Release (v5.0) of the Core Package", The
  Astrophysical Journal, 935, 167.
  arXiv: https://arxiv.org/abs/2206.14220

## Notes on Citation Use

- The app's Lomb-Scargle and false-alarm sections should cite Lomb (1976),
  Scargle (1982), Zechmeister & Kuerster (2009), VanderPlas (2018), and Baluev
  (2008).
- Sampling-window and alias discussion should cite Roberts et al. (1987) and
  Foster (1995, 1996), with Vio et al. (2013) as an additional cautionary
  reference for irregular sampling.
- Period tomography should cite Foster (1996) and Andronov & Naumova (2013).
- The binary toy model should cite Wilson & Devinney (1971), Wilson (1979,
  1994), Prsa & Zwitter (2005), and Maxted (2016).
- The stellar preset table should cite Martins et al. (2005), Pecaut &
  Mamajek (2013), Eker et al. (2015), and Cox (2000), while explicitly stating
  that the values in the app are approximate teaching presets rather than a
  reproduction of any one table.
- Bondi-Hoyle sections should cite Hoyle & Lyttleton (1939), Bondi & Hoyle
  (1944), Bondi (1952), Castor et al. (1975), Lamers & Cassinelli (1999), and
  Edgar (2004).
- Pulse-period sections should cite Leahy et al. (1983), Buccheri et al.
  (1983), Blandford & Teukolsky (1976), and Ransom et al. (2002).
