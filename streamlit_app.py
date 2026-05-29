from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import run_analysis


st.set_page_config(
    page_title="Periodicity Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)


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


def period_options_from_result(result: dict | None, key: str) -> dict[str, float]:
    if result is None:
        return {}
    options: dict[str, float] = {}
    for peak in result.get(key, []):
        if "artefact" in peak["kind"]:
            continue
        label = f"{peak['label']} - {peak['period']:.4f} d ({peak['kind']})"
        options[label] = float(peak["period"])
    return options


def unique_periods(periods: list[float], tolerance: float = 1e-5) -> list[float]:
    unique: list[float] = []
    for period in periods:
        if all(abs(period - old) > tolerance * max(period, old) for old in unique):
            unique.append(period)
    return unique


def add_peak_markers(fig: go.Figure, peaks: list[dict], y_key: str = "power") -> None:
    for peak in peaks:
        color = "#b13b32" if "artefact" in peak["kind"] else "#2457a6"
        y_value = peak.get(y_key)
        if y_value is None:
            continue
        fig.add_vline(x=peak["period"], line_dash="dash", line_color=color, opacity=0.75)
        fig.add_trace(
            go.Scatter(
                x=[peak["period"]],
                y=[y_value],
                mode="markers+text",
                marker=dict(color=color, size=7),
                text=[f"{peak['period']:.2f} d"],
                textposition="top right",
                showlegend=False,
                hovertemplate=(
                    f"{peak['label']}<br>"
                    "Period=%{x:.6f} d<br>"
                    "Power=%{y:.6g}<br>"
                    f"Type={peak['kind']}<extra></extra>"
                ),
            )
        )


def periodogram_figure(result: dict, key: str, title: str, peaks_key: str) -> go.Figure:
    series = result["series"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["period"],
            y=series[key],
            mode="lines",
            line=dict(color="#20242a", width=1.2),
            name=title,
            hovertemplate="Period=%{x:.6f} d<br>Power=%{y:.6g}<extra></extra>",
        )
    )
    add_peak_markers(fig, result[peaks_key])
    fig.update_layout(
        title=title,
        xaxis_title="Period [d]",
        yaxis_title="Lomb-Scargle power",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig


def window_figure(result: dict) -> go.Figure:
    series = result["series"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["period"],
            y=series["window_power"],
            mode="lines",
            line=dict(color="#20242a", width=1.2),
            hovertemplate="Period=%{x:.6f} d<br>Window power=%{y:.6g}<extra></extra>",
        )
    )
    for peak in result["peaks"]:
        if peak["window_period"] is not None:
            fig.add_vline(x=peak["window_period"], line_dash="dash", line_color="#b13b32", opacity=0.75)
    fig.update_layout(
        title="Sampling window",
        xaxis_title="Period [d]",
        yaxis_title="Sampling-window power",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    return fig


def folded_figure(result: dict) -> go.Figure:
    series = result["series"]
    phase = series["fold_phase"]
    flux = series["fold_flux"]
    error = series["fold_error"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=phase + [p + 1.0 for p in phase],
            y=flux + flux,
            error_y=dict(type="data", array=error + error, visible=True),
            mode="markers",
            marker=dict(color="#20242a", size=7),
            name="Folded bins",
            hovertemplate="Phase=%{x:.5f}<br>Flux=%{y:.6g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["fold_model_phase"],
            y=series["fold_model_flux"],
            mode="lines",
            line=dict(color="#2457a6", width=2, dash="dot"),
            name="Folded model",
            hovertemplate="Phase=%{x:.5f}<br>Model=%{y:.6g}<extra></extra>",
        )
    )
    for maximum in result["folded_maxima"]:
        for offset in [0.0, 1.0]:
            fig.add_vline(x=maximum["phase"] + offset, line_dash="dash", line_color="#2457a6", opacity=0.75)
    fig.update_layout(
        title="Folded profile",
        xaxis_title="Orbital phase",
        yaxis_title="Weighted mean flux",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
    )
    fig.update_xaxes(range=[0, 2])
    return fig


def render_results(result: dict) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows used", f"{result['n_points']}")
    metric_cols[1].metric("Baseline", f"{result['baseline']:.1f} d")
    metric_cols[2].metric("Primary period", f"{result['primary_period']:.4f} d")
    metric_cols[3].metric("T0", f"{result['t0']:.4f}")

    plot_cols = st.columns(2)
    with plot_cols[0]:
        st.plotly_chart(periodogram_figure(result, "power", "Lomb-Scargle periodogram", "peaks"), use_container_width=True)
    with plot_cols[1]:
        st.plotly_chart(window_figure(result), use_container_width=True)
    with plot_cols[0]:
        st.plotly_chart(periodogram_figure(result, "residual_power", "After prewhitening", "residual_peaks"), use_container_width=True)
    with plot_cols[1]:
        st.plotly_chart(folded_figure(result), use_container_width=True)

    st.subheader("Detected peaks")
    st.dataframe(peaks_dataframe(result["peaks"]), use_container_width=True, hide_index=True)

    st.subheader("After prewhitening")
    st.dataframe(peaks_dataframe(result["residual_peaks"]), use_container_width=True, hide_index=True)

    st.subheader("Folded-profile maxima")
    st.dataframe(pd.DataFrame(result["folded_maxima"]), use_container_width=True, hide_index=True)

    st.subheader("Folded-fit periods")
    st.dataframe(
        pd.DataFrame(
            {
                "period_d": result["fold_fit_periods"],
                "frequency_ratio_in_folded_phase": result["fold_fit_ratios"],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


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
    max_peaks = st.number_input("Max considered peaks", min_value=1, max_value=20, value=6, step=1)
    min_considered_period = st.number_input("Minimum considered period [d]", min_value=0.0, value=2.0, step=0.1)
    window_artifact_power = st.number_input(
        "Sampling-window artefact threshold",
        min_value=0.0,
        value=0.01,
        step=0.005,
        format="%.4f",
        help="Lomb-Scargle peaks close to sampling-window peaks stronger than this are excluded from primary-period selection.",
    )
    window_artifact_tolerance = st.number_input(
        "Sampling-window match tolerance",
        min_value=0.001,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="Relative tolerance in period/frequency for considering a peak close to a sampling-window feature.",
    )

    st.subheader("Uncertainty")
    n_bootstrap = st.number_input("Bootstrap iterations", min_value=0, value=1000, step=50)
    bootstrap_width = st.number_input("Bootstrap local width", min_value=0.001, value=0.03, step=0.001, format="%.3f")

    st.subheader("Folded Profile")
    fold_bins = st.number_input("Phase bins", min_value=4, max_value=80, value=10, step=1)
    t0_text = st.text_input("T0 / MJD", value="", placeholder="first data point")
    include_harmonic = st.checkbox("Include first harmonic", value=True)
    fold_fit_mode = st.radio(
        "Folded-fit frequencies",
        ["Harmonics of primary", "Selected periods"],
        horizontal=False,
    )
    if fold_fit_mode == "Harmonics of primary":
        fold_fit_harmonics = st.number_input(
            "Number of harmonics in folded fit",
            min_value=1,
            max_value=8,
            value=2 if include_harmonic else 1,
            step=1,
        )
        selected_period_values: list[float] = []
    else:
        previous_result = st.session_state.get("last_result")
        detected_options = period_options_from_result(previous_result, "peaks")
        prewhitened_options = period_options_from_result(previous_result, "residual_peaks")
        selected_detected_labels = st.multiselect(
            "Use periods from detected peaks",
            options=list(detected_options.keys()),
            default=[],
            help="Run once to populate this list, then select periods and run again.",
        )
        selected_prewhitened_labels = st.multiselect(
            "Use periods from after-prewhitening peaks",
            options=list(prewhitened_options.keys()),
            default=[],
            help="You can select more than one period from this list.",
        )
        selected_period_values = [
            detected_options[label] for label in selected_detected_labels
        ] + [
            prewhitened_options[label] for label in selected_prewhitened_labels
        ]
        manual_periods = st.text_input(
            "Additional periods [d]",
            value="",
            placeholder="e.g. 19.13, 9.56",
            help="Comma, semicolon, or whitespace separated.",
        )
        if manual_periods.strip():
            selected_period_values.extend(
                float(value)
                for value in manual_periods.replace(",", " ").replace(";", " ").split()
                if value.strip()
            )
        selected_period_values = unique_periods(selected_period_values)
        if selected_period_values:
            st.caption(
                "Selected folded-fit periods: "
                + ", ".join(f"{period:.4f} d" for period in selected_period_values)
            )
        fold_fit_harmonics = 1

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
        "min_considered_period": str(min_considered_period),
        "window_artifact_power": str(window_artifact_power),
        "window_artifact_tolerance": str(window_artifact_tolerance),
        "n_bootstrap": str(n_bootstrap),
        "bootstrap_width": str(bootstrap_width),
        "fold_bins": str(fold_bins),
        "fold_fit_mode": "selected" if fold_fit_mode == "Selected periods" else "harmonics",
        "fold_fit_harmonics": str(fold_fit_harmonics),
    }
    if selected_period_values:
        fields["fold_fit_periods"] = ",".join(f"{period:.12g}" for period in selected_period_values)
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
    st.session_state["last_result"] = result

if "last_result" in st.session_state:
    if not run:
        st.caption("Showing the last completed analysis. Press Run analysis to apply the current settings.")
    render_results(st.session_state["last_result"])
else:
    st.info("Upload a light curve and press Run analysis. For exploration, use 50-200 bootstrap iterations; use 1000 for final numbers.")
