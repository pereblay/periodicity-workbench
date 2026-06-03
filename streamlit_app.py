from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import run_analysis, update_folded_profile


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
                "FAP": f"{float(peak['fap']):.5f}",
                "type": peak["kind"],
                "window_period_d": peak["window_period"],
            }
        )
    return pd.DataFrame(rows)


def window_peaks_dataframe(peaks: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "period_d": peak["period"],
                "frequency_c_d": peak["frequency"],
                "window_power": peak["power"],
            }
            for peak in peaks
        ]
    )


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


def period_select_options_from_result(result: dict | None, key: str) -> dict[float, str]:
    if result is None:
        return {}
    options: dict[float, str] = {}
    for peak in result.get(key, []):
        if "artefact" in peak["kind"] or "excluded" in peak["kind"]:
            continue
        period = float(peak["period"])
        if any(abs(period - old) <= 1e-5 * max(period, old) for old in options):
            continue
        options[period] = f"{peak['label']} - {period:.4f} d ({peak['kind']})"
    return options


def unique_periods(periods: list[float], tolerance: float = 1e-5) -> list[float]:
    unique: list[float] = []
    for period in periods:
        if all(abs(period - old) > tolerance * max(period, old) for old in unique):
            unique.append(period)
    return unique


def parse_period_text(text: str) -> list[float]:
    if not text.strip():
        return []
    values: list[float] = []
    for item in text.replace(",", " ").replace(";", " ").split():
        period = float(item)
        if period <= 0:
            raise ValueError("Periods must be positive")
        values.append(period)
    return values


def exclusion_options_from_result(result: dict | None) -> dict[float, str]:
    if result is None:
        return {}
    options: dict[float, str] = {}
    for group, key in [("detected", "peaks"), ("prewhitened", "residual_peaks")]:
        for peak in result.get(key, []):
            period = float(peak["period"])
            if any(abs(period - old) <= 1e-5 * max(period, old) for old in options):
                continue
            options[period] = f"{group}: {peak['label']} - {period:.4f} d ({peak['kind']})"
    return options


def add_peak_markers(fig: go.Figure, peaks: list[dict], y_key: str = "power") -> None:
    for peak in peaks:
        if "excluded" in peak["kind"]:
            color = "#777777"
        elif "artefact" in peak["kind"]:
            color = "#b13b32"
        else:
            color = "#2457a6"
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
    for peak in result.get("window_peaks", []):
        fig.add_vline(x=peak["period"], line_dash="dash", line_color="#b13b32", opacity=0.35)
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


def prewhitening_model_figure(result: dict) -> go.Figure:
    series = result["series"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=series["time"],
            y=series["flux"],
            error_y=dict(type="data", array=series["error"], visible=True),
            mode="markers",
            marker=dict(color="#20242a", size=5),
            name="Original data",
            hovertemplate="Time=%{x:.5f}<br>Flux=%{y:.6g}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=series["time"],
            y=series["prewhitening_model_flux"],
            mode="lines",
            line=dict(color="#2457a6", width=2),
            name="Prewhitening model",
            hovertemplate="Time=%{x:.5f}<br>Model=%{y:.6g}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Original light curve with prewhitening model",
        xaxis_title="Time",
        yaxis_title="Flux",
        height=430,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def render_results(result: dict) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows used", f"{result['n_points']}")
    metric_cols[1].metric("Baseline", f"{result['baseline']:.1f} d")
    metric_cols[2].metric("Primary period", f"{result['primary_period']:.4f} d")
    metric_cols[3].metric("Folded period", f"{result.get('folded_period', result['primary_period']):.4f} d")

    top_plot_cols = st.columns(3)
    with top_plot_cols[0]:
        st.plotly_chart(periodogram_figure(result, "power", "Lomb-Scargle periodogram", "peaks"), use_container_width=True)
    with top_plot_cols[1]:
        st.plotly_chart(window_figure(result), use_container_width=True)
    with top_plot_cols[2]:
        st.plotly_chart(folded_figure(result), use_container_width=True)

    st.subheader("Detected peaks")
    st.dataframe(peaks_dataframe(result["peaks"]), use_container_width=True, hide_index=True)

    st.subheader("Prewhitening")
    prewhitening_cols = st.columns([1.3, 1.0])
    with prewhitening_cols[0]:
        if result.get("has_prewhitening"):
            st.plotly_chart(
                periodogram_figure(result, "residual_power", "After prewhitening", "residual_peaks"),
                use_container_width=True,
            )
        else:
            st.info("No prewhitening step has been applied yet.")
    with prewhitening_cols[1]:
        st.dataframe(pd.DataFrame(result.get("prewhitening_terms", [])), use_container_width=True, hide_index=True)

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

    if st.session_state.get("show_prewhitening_model"):
        if result.get("has_prewhitening"):
            st.plotly_chart(prewhitening_model_figure(result), use_container_width=True)
        else:
            st.info("Add at least one prewhitening step before showing the model.")


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
            value=1,
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
            selected_period_values.extend(parse_period_text(manual_periods))
        selected_period_values = unique_periods(selected_period_values)
        if selected_period_values:
            st.caption(
                "Folding on the first selected period; fitted periods: "
                + ", ".join(f"{period:.4f} d" for period in selected_period_values)
            )
        fold_fit_harmonics = 1
    update_folded = st.button(
        "Update folded profile",
        use_container_width=True,
        help="Refresh only the folded profile using the last completed analysis.",
    )

    st.subheader("Manual Exclusions")
    exclusion_options = exclusion_options_from_result(st.session_state.get("last_result"))
    selected_exclusion_periods = st.multiselect(
        "Exclude periods from primary selection",
        options=list(exclusion_options.keys()),
        format_func=lambda period: exclusion_options.get(period, f"{period:.4f} d"),
        default=[],
        help="Select peaks that are clearly sampling-window aliases or otherwise unwanted.",
    )
    manual_excluded_periods = st.text_input(
        "Additional excluded periods [d]",
        value="",
        placeholder="e.g. 1.04, 24.13",
        help="Comma, semicolon, or whitespace separated.",
    )
    exclusion_tolerance = st.number_input(
        "Manual exclusion tolerance",
        min_value=0.001,
        value=0.015,
        step=0.001,
        format="%.3f",
        help="Relative period tolerance used to match manual exclusions to detected peaks.",
    )
    excluded_period_values = list(selected_exclusion_periods)
    if manual_excluded_periods.strip():
        excluded_period_values.extend(parse_period_text(manual_excluded_periods))
    excluded_period_values = unique_periods(excluded_period_values)
    if excluded_period_values:
        st.caption("Excluded periods: " + ", ".join(f"{period:.4f} d" for period in excluded_period_values))

    apply_exclusions = st.button(
        "Apply exclusions",
        use_container_width=True,
        help="Re-select the primary period using the last uploaded file and the current manual exclusions. Bootstrap is skipped for this quick update.",
    )

    st.subheader("Iterative Prewhitening")
    if "prewhitening_periods" not in st.session_state:
        st.session_state["prewhitening_periods"] = []
    prewhitening_periods = list(st.session_state["prewhitening_periods"])
    prewhitening_options = period_select_options_from_result(st.session_state.get("last_result"), "residual_peaks")
    if not prewhitening_options:
        prewhitening_options = period_select_options_from_result(st.session_state.get("last_result"), "peaks")
    if prewhitening_options:
        next_prewhitening_period = st.selectbox(
            "Next period to remove",
            options=list(prewhitening_options.keys()),
            format_func=lambda period: prewhitening_options.get(period, f"{period:.4f} d"),
            index=None,
            placeholder="Choose a detected period",
        )
    else:
        next_prewhitening_period = None
        st.caption("Run an analysis to populate prewhitening candidates.")
    if prewhitening_periods:
        st.caption("Current prewhitening chain: " + ", ".join(f"{period:.4f} d" for period in prewhitening_periods))
    next_prewhitening = st.button(
        "Next prewhitening step",
        use_container_width=True,
        help="Add the selected period and recompute the residual periodogram. Detected harmonics are included only when they appear as non-artefact peaks.",
    )
    show_model = st.button(
        "Show model",
        use_container_width=True,
        help="Show the model fitted to the original light curve using all periods accumulated in the prewhitening table.",
    )
    clear_prewhitening = st.button("Clear prewhitening chain", use_container_width=True)

    run = st.button("Run analysis", type="primary", use_container_width=True)


if uploaded is not None:
    with st.expander("File preview", expanded=False):
        raw_preview = uploaded.getvalue().decode("ascii", errors="replace")
        st.code("\n".join(raw_preview.splitlines()[:12]))

def current_fields(bootstrap_value: int | None = None) -> dict[str, str]:
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
        "n_bootstrap": str(n_bootstrap if bootstrap_value is None else bootstrap_value),
        "bootstrap_width": str(bootstrap_width),
        "fold_bins": str(fold_bins),
        "fold_fit_mode": "selected" if fold_fit_mode == "Selected periods" else "harmonics",
        "fold_fit_harmonics": str(fold_fit_harmonics),
        "exclusion_tolerance": str(exclusion_tolerance),
    }
    if selected_period_values:
        fields["fold_fit_periods"] = ",".join(f"{period:.12g}" for period in selected_period_values)
    if excluded_period_values:
        fields["excluded_periods"] = ",".join(f"{period:.12g}" for period in excluded_period_values)
    prewhitening_periods_for_fields = st.session_state.get("prewhitening_periods", [])
    if prewhitening_periods_for_fields:
        fields["prewhiten_periods"] = ",".join(f"{period:.12g}" for period in prewhitening_periods_for_fields)
    if t0_text.strip():
        fields["t0"] = t0_text.strip()
    return fields


def current_live_signature() -> tuple[int, float]:
    return int(max_peaks), float(min_considered_period)


def run_with_last_file(fields: dict[str, str], spinner_text: str) -> dict:
    file_bytes = st.session_state.get("last_file_bytes")
    filename = st.session_state.get("last_filename", "uploaded.dat")
    if file_bytes is None:
        raise ValueError("Run an analysis with an uploaded file first.")
    with st.spinner(spinner_text):
        return run_analysis(fields, file_bytes, filename)


if update_folded:
    if "last_result" not in st.session_state:
        st.error("Run an analysis before updating the folded profile.")
        st.stop()
    try:
        fields = current_fields(bootstrap_value=0)
        st.session_state["last_result"] = update_folded_profile(st.session_state["last_result"], fields)
        st.session_state["last_fields"] = fields
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    st.rerun()


if clear_prewhitening:
    st.session_state["prewhitening_periods"] = []
    if "last_file_bytes" in st.session_state:
        try:
            fields = current_fields(bootstrap_value=0)
            result = run_with_last_file(fields, "Clearing prewhitening chain...")
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        st.session_state["last_result"] = result
        st.session_state["last_fields"] = fields
        st.session_state["last_live_signature"] = current_live_signature()
        st.rerun()


if next_prewhitening:
    if next_prewhitening_period is None:
        st.error("Choose a detected period before adding a prewhitening step.")
        st.stop()
    st.session_state["prewhitening_periods"] = unique_periods(
        list(st.session_state.get("prewhitening_periods", [])) + [float(next_prewhitening_period)]
    )
    try:
        fields = current_fields(bootstrap_value=0)
        result = run_with_last_file(fields, "Running next prewhitening step...")
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    st.session_state["last_result"] = result
    st.session_state["last_fields"] = fields
    st.session_state["last_live_signature"] = current_live_signature()
    st.rerun()


if show_model:
    if "last_result" not in st.session_state:
        st.error("Run an analysis before showing the model.")
        st.stop()
    st.session_state["show_prewhitening_model"] = True


if (
    "last_file_bytes" in st.session_state
    and not run
    and not apply_exclusions
    and not next_prewhitening
    and not clear_prewhitening
    and not show_model
    and not update_folded
):
    live_signature = current_live_signature()
    previous_signature = st.session_state.get("last_live_signature")
    if previous_signature is not None and live_signature != previous_signature:
        try:
            fields = current_fields(bootstrap_value=0)
            result = run_with_last_file(fields, "Updating peak selection...")
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        st.session_state["last_result"] = result
        st.session_state["last_fields"] = fields
        st.session_state["last_live_signature"] = live_signature
        st.rerun()


if run:
    if uploaded is None:
        st.error("Upload a text table first.")
        st.stop()

    fields = current_fields()

    with st.spinner("Running analysis..."):
        try:
            result = run_analysis(fields, uploaded.getvalue(), uploaded.name)
        except Exception as exc:
            st.error(str(exc))
            st.stop()
    st.session_state["last_result"] = result
    st.session_state["last_file_bytes"] = uploaded.getvalue()
    st.session_state["last_filename"] = uploaded.name
    st.session_state["last_fields"] = fields
    st.session_state["last_live_signature"] = current_live_signature()
    st.rerun()

if apply_exclusions:
    fields = current_fields(bootstrap_value=0)
    try:
        result = run_with_last_file(fields, "Applying manual exclusions...")
    except Exception as exc:
        st.error(str(exc))
        st.stop()
    st.session_state["last_result"] = result
    st.session_state["last_fields"] = fields
    st.session_state["last_live_signature"] = current_live_signature()

if "last_result" in st.session_state:
    if not run and not apply_exclusions:
        st.caption("Showing the last completed analysis. Press Run analysis to apply the current settings.")
    if apply_exclusions:
        st.caption("Manual exclusions applied using the last uploaded file; bootstrap was skipped for this quick update.")
    render_results(st.session_state["last_result"])
else:
    st.info("Upload a light curve and press Run analysis. For exploration, use 50-200 bootstrap iterations; use 1000 for final numbers.")
