from __future__ import annotations

import base64
from io import BytesIO

import pandas as pd
import streamlit as st

from app import run_analysis


st.set_page_config(
    page_title="Periodicity Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)


def image_bytes(data_uri: str) -> BytesIO:
    _, encoded = data_uri.split(",", 1)
    return BytesIO(base64.b64decode(encoded))


def peaks_dataframe(peaks: list[dict]) -> pd.DataFrame:
    rows = []
    for peak in peaks:
        rows.append(
            {
                "label": peak["label"],
                "period_d": peak["period"],
                "period_err_d": peak["period_error"],
                "frequency_c_d": peak["frequency"],
                "frequency_err_c_d": peak["frequency_error"],
                "power": peak["power"],
                "FAP": peak["fap"],
                "type": peak["kind"],
                "window_period_d": peak["window_period"],
            }
        )
    return pd.DataFrame(rows)


st.title("Periodicity Workbench")
st.caption("Lomb-Scargle, sampling-window checks, bootstrap errors, prewhitening, and folded profiles.")

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader(
        "Text table (.txt, .dat, or no extension)",
        type=["txt", "dat"],
        accept_multiple_files=False,
        help="Files without extension can be uploaded by choosing 'All files' in the file picker.",
    )

    st.subheader("Columns")
    col_a, col_b, col_c = st.columns(3)
    time_col = col_a.number_input("Time", min_value=1, value=1, step=1)
    flux_col = col_b.number_input("Flux", min_value=1, value=2, step=1)
    error_col = col_c.number_input("Error", min_value=1, value=3, step=1)

    st.subheader("Frequency Search")
    fmin = st.number_input("Min frequency [cycles/day]", min_value=0.0, value=0.01, step=0.001, format="%.6f")
    fmax = st.number_input("Max frequency [cycles/day]", min_value=0.0, value=1.0, step=0.01, format="%.6f")
    samples_per_peak = st.number_input("Samples per peak", min_value=1.0, value=10.0, step=1.0)
    max_peaks = st.number_input("Max marked peaks", min_value=1, max_value=20, value=6, step=1)
    min_marked_period = st.number_input("Minimum marked period [d]", min_value=0.0, value=2.0, step=0.1)

    st.subheader("Uncertainty")
    n_bootstrap = st.number_input("Bootstrap iterations", min_value=0, value=1000, step=50)
    bootstrap_width = st.number_input("Bootstrap local width", min_value=0.001, value=0.03, step=0.001, format="%.3f")

    st.subheader("Folded Profile")
    fold_bins = st.number_input("Phase bins", min_value=4, max_value=80, value=10, step=1)
    t0_text = st.text_input("T0 / MJD", value="", placeholder="first data point")
    include_harmonic = st.checkbox("Include first harmonic", value=True)

    run = st.button("Run analysis", type="primary", use_container_width=True)


if uploaded is not None:
    with st.expander("File preview", expanded=False):
        raw_preview = uploaded.getvalue().decode("ascii", errors="replace")
        st.code("\n".join(raw_preview.splitlines()[:12]))

if run:
    if uploaded is None:
        st.error("Upload a text table first.")
        st.stop()

    fields = {
        "time_col": str(time_col),
        "flux_col": str(flux_col),
        "error_col": str(error_col),
        "fmin": str(fmin),
        "fmax": str(fmax),
        "samples_per_peak": str(samples_per_peak),
        "max_peaks": str(max_peaks),
        "min_marked_period": str(min_marked_period),
        "n_bootstrap": str(n_bootstrap),
        "bootstrap_width": str(bootstrap_width),
        "fold_bins": str(fold_bins),
    }
    if t0_text.strip():
        fields["t0"] = t0_text.strip()
    if include_harmonic:
        fields["include_harmonic"] = "on"

    with st.spinner("Running analysis..."):
        try:
            result = run_analysis(fields, uploaded.getvalue(), uploaded.name)
        except Exception as exc:
            st.error(str(exc))
            st.stop()

    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows used", f"{result['n_points']}")
    metric_cols[1].metric("Baseline", f"{result['baseline']:.1f} d")
    metric_cols[2].metric("Primary period", f"{result['primary_period']:.4f} d")
    metric_cols[3].metric("T0", f"{result['t0']:.4f}")

    plot_cols = st.columns(2)
    with plot_cols[0]:
        st.image(image_bytes(result["plots"]["periodogram"]), caption="Lomb-Scargle periodogram", use_container_width=True)
    with plot_cols[1]:
        st.image(image_bytes(result["plots"]["window"]), caption="Sampling window", use_container_width=True)
    with plot_cols[0]:
        st.image(image_bytes(result["plots"]["prewhitened"]), caption="After prewhitening", use_container_width=True)
    with plot_cols[1]:
        st.image(image_bytes(result["plots"]["folded"]), caption="Folded profile", use_container_width=True)

    st.subheader("Detected peaks")
    st.dataframe(peaks_dataframe(result["peaks"]), use_container_width=True, hide_index=True)

    st.subheader("After prewhitening")
    st.dataframe(peaks_dataframe(result["residual_peaks"]), use_container_width=True, hide_index=True)

    st.subheader("Folded-profile maxima")
    st.dataframe(pd.DataFrame(result["folded_maxima"]), use_container_width=True, hide_index=True)
else:
    st.info("Upload a light curve and press Run analysis. For exploration, use 50-200 bootstrap iterations; use 1000 for final numbers.")
