# Periodicity Workbench Streamlit

Streamlit version of the local Periodicity Workbench app.

Current version: `1.7.0`.

## What's New in 1.7

- Full vector wind/orbit relative velocity for eccentric systems.
- Classical pressure-aware BHL and revised binary BHL formulations.
- Separate normalized teaching and physical mass-accretion/luminosity modes.
- Sound speed, Mach number, accretion radius, Roche-lobe and capture-efficiency diagnostics.
- Optional phenomenological attenuation and causal response delay without moving the periastron epoch.
- Parameter covariance, residual-bootstrap intervals, bound warnings, and comparison with constant, sinusoidal, and two-harmonic models.
- Structured validity warnings shared by the interface and PDF report.

It supports:

- Uploading ASCII text tables with `.txt`, `.dat`, `.lc`, or no extension.
- Uploading FITS-like tables detected by file content, including common `.fits`, `.fit`, `.fts`, `.ftz`, `.lc`, and `.fits.gz` files.
- Choosing the time, flux/count-rate/magnitude, and optional error columns.
- Previewing the selected table rows and original light curve before running the analysis.
- Resolving and recording target coordinates with SIMBAD/Sesame.
- Inspecting FITS barycentric timing metadata and applying a clearly labelled geocentric approximation when possible.
- Bibliographic lookups for the target object with time-series and periodicity search terms.
- Weighted Lomb-Scargle periodograms.
- Sampling-window inspection and tentative artefact classification.
- Local residual bootstrap errors for detected peaks.
- Iterative prewhitening with user-selected periods.
- Compact harmonic diagnostics and prewhitening quality metrics.
- A second Lomb-Scargle search on prewhitened residuals.
- Orbital phase folding with user-selected `T0` and number of phase bins.
- Folded-profile fitting with selected periods or harmonics, with maxima marked.
- Period tomography through sliding Lomb-Scargle and WWZ-style maps.
- A model laboratory for Fourier multi-harmonic profiles, eclipsing/eccentric binaries, classical or revised Bondi-Hoyle wind accretion, and X-ray pulse timing with epoch-folding significance, candidate TOAs, F0/F1 spin ephemerides, background-aware pulsed fractions, and circular/Keplerian Roemer-delay fits.
- A binary interpretation assistant for folded-profile minima, depth ratios, and half-period ambiguity checks.
- Stellar spectral-type presets for toy binary and wind-accretion models, including OB and A/F/G/K/M representative stars.
- Download buttons for the plotted data tables.
- Evidence boxes for student notes and literature comparisons.
- Compact branded PDF report generation with selectable sections, per-page header/footer metadata, and optional append-to-existing-PDF support.

## Student Report Workflow

The built-in help includes a minimum student workflow:

1. Load the light curve without error weighting for the first inspection.
2. Run an initial frequency search and inspect the Lomb-Scargle periodogram, sampling window, and folded profile.
3. Refine the frequency range and then enable uncertainty estimates.
4. Investigate the folded profile and test whether harmonics improve the fit.
5. Fill in the evidence boxes with adopted periods, uncertainties, and literature comparison.
6. Generate a PDF report including the selected analysis sections.

## Model Laboratory Notes

The model laboratory is intended for exploration and teaching. Its analytic
models expose their assumptions and report when the selected parameters leave
their validity domain.

The Bondi-Hoyle block now resolves the compact object's radial and tangential
orbital velocity:

```text
v_rel^2 = (v_w - v_orb,r)^2 + v_orb,t^2
```

The classical pressure-aware calculation uses

```text
Mdot_BH = 4 pi G^2 M_compact^2 rho /
          (v_rel^2 + c_s^2)^(3/2)
```

An alternative revised-binary prescription applies the geometric capture
efficiency described by Tejeda and Toala. Both formulations use the beta wind
law

```text
v_w(r) = v_inf * (1 - R_donor / r)^beta
```

Normalized mode fits `y=C+A proxy` and remains phenomenological. Physical mode
uses terminal wind speed, masses, donor radius, mass-loss rate, compact-object
radius or radiative efficiency to calculate density, accretion rate, luminosity,
and Eddington ratio. Optional attenuation and response models are deliberately
labelled phenomenological. The validity panel reports Mach number, accretion
radius, Roche-lobe filling, slow-wind conditions, efficiency limits, parameter
bounds, and negative flux/accretion correlations.

## Local Run

```bash
cd /path_of_installation/periodicity_app_streamlit
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Go to Streamlit Community Cloud.
3. Create a new app from the repository.
4. Set the main file path to:

```text
streamlit_app.py
```

The default bootstrap count is 1000. For interactive exploration, use 50-200 first, then increase for final values.

## Notes

The upload validator only accepts ASCII numeric tables or valid FITS payloads. Header/comment lines starting with `#` or `%` are ignored for ASCII tables. ASCII uploads reject non-ASCII content, unsupported extensions, and simple command-like patterns before parsing the table. FITS uploads are opened with `astropy.io.fits`, and only scalar numeric columns are offered for analysis.

Sampling-window classification is heuristic. It flags a periodogram peak as a likely artefact when it coincides, within the approximate Rayleigh resolution, with one of the stronger peaks in the sampling window.
