# Periodicity Workbench Streamlit

Streamlit version of the local Periodicity Workbench app.

Current version: `1.4.0`.

It supports:

- Uploading ASCII text tables with `.txt`, `.dat`, `.lc`, or no extension.
- Uploading FITS-like tables detected by file content, including common `.fits`, `.fit`, `.fts`, `.ftz`, `.lc`, and `.fits.gz` files.
- Choosing the time, flux/count-rate/magnitude, and optional error columns.
- Previewing the selected table rows and original light curve before running the analysis.
- Resolving and recording target coordinates with SIMBAD/Sesame.
- Bibliographic lookups for the target object with time-series and periodicity search terms.
- Weighted Lomb-Scargle periodograms.
- Sampling-window inspection and tentative artefact classification.
- Local residual bootstrap errors for detected peaks.
- Iterative prewhitening with user-selected periods.
- A second Lomb-Scargle search on prewhitened residuals.
- Orbital phase folding with user-selected `T0` and number of phase bins.
- Folded-profile fitting with selected periods or harmonics, with maxima marked.
- Period tomography through sliding Lomb-Scargle and WWZ-style maps.
- A model laboratory for Fourier multi-harmonic profiles, eclipsing/eccentric binary toy models, Bondi-Hoyle wind-accretion toy models, and X-ray pulse-period timing.
- Stellar spectral-type presets for toy binary and wind-accretion models, including OB and A/F/G/K/M representative stars.
- Download buttons for the plotted data tables.
- Evidence boxes for student notes and literature comparisons.
- Compact PDF report generation with selectable sections and optional append-to-existing-PDF support.

## Student Report Workflow

The built-in help includes a minimum student workflow:

1. Load the light curve without error weighting for the first inspection.
2. Run an initial frequency search and inspect the Lomb-Scargle periodogram, sampling window, and folded profile.
3. Refine the frequency range and then enable uncertainty estimates.
4. Investigate the folded profile and test whether harmonics improve the fit.
5. Fill in the evidence boxes with adopted periods, uncertainties, and literature comparison.
6. Generate a PDF report including the selected analysis sections.

## Model Laboratory Notes

The model laboratory is intended for exploration and teaching. The models are deliberately lightweight and should not be treated as final physical solutions.

The Bondi-Hoyle block estimates a dimensionless wind-accretion proxy,

```text
Mdot_acc ~ rho / v_rel^3
```

and fits the observed count rate with

```text
y(t) = C + A * proxy_BHL(t)
```

The wind acceleration term now uses the donor radius explicitly:

```text
v_w(r) = v_inf_or_ratio * (1 - R_donor / r)^beta
```

In `v_inf` mode, `R_donor/a` is computed from the orbital period, donor and compact-object masses, and donor radius. In dimensionless `v_wind / v_orb` mode, `R_donor/a` is entered directly. Optional stellar presets provide approximate masses, radii, luminosities, and effective temperatures for OB and representative A/F/G/K/M spectral types with luminosity classes V, III, and I. These presets are pedagogical starting values, not system-specific calibrations.

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
