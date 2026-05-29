# Periodicity Workbench Streamlit

Streamlit version of the local Periodicity Workbench app.

It supports:

- Uploading an ASCII text table with `.txt`, `.dat`, or no extension.
- Choosing the time, flux/count-rate, and error columns.
- Weighted Lomb-Scargle periodograms.
- Sampling-window inspection and tentative artefact classification.
- Local residual bootstrap errors for detected peaks.
- Prewhitening using the main non-artefact periodicity and, optionally, its first harmonic.
- A second Lomb-Scargle search on prewhitened residuals.
- Orbital phase folding with user-selected `T0` and number of phase bins.
- A folded-profile fit with the fundamental and optional harmonic, with maxima marked.

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

The upload validator only accepts ASCII numeric tables. Header/comment lines starting with `#` or `%` are ignored. It rejects non-ASCII content, unsupported extensions, and simple command-like patterns before parsing the table.

Sampling-window classification is heuristic. It flags a periodogram peak as a likely artefact when it coincides, within the approximate Rayleigh resolution, with one of the stronger peaks in the sampling window.
