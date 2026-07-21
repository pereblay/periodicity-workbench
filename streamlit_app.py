from __future__ import annotations

import hashlib
import io
from pathlib import Path
import ssl
import textwrap
from urllib.parse import quote, urlencode
from urllib.request import urlopen
import xml.etree.ElementTree as ET

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from astropy.coordinates import SkyCoord

try:
    from pypdf import PdfReader, PdfWriter
except Exception:  # pragma: no cover - optional deployment dependency
    PdfReader = None
    PdfWriter = None

from app import BHL_DONOR_PRESETS, advanced_time_frequency_map, bhl_donor_preset_values, binary_model_lab, bondi_hoyle_model_lab, fits_table_metadata, fourier_model_lab, is_fits_upload, pulse_period_model_lab, read_columns, run_analysis, update_folded_profile, validate_upload


try:
    APP_VERSION = (Path(__file__).resolve().parent / "VERSION").read_text(encoding="utf-8").strip()
except OSError:
    APP_VERSION = "1.2.0"


st.set_page_config(
    page_title="Periodicity Workbench",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        height: 100vh !important;
        overflow: hidden !important;
    }
    [data-testid="stSidebar"] > div:first-child,
    [data-testid="stSidebarContent"] {
        height: 100vh !important;
        max-height: 100vh !important;
        overflow-y: auto !important;
        overscroll-behavior: contain;
        padding-bottom: 3rem;
    }
    @media (max-width: 900px) {
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div:first-child {
            width: 100vw !important;
            min-width: 100vw !important;
            max-width: 100vw !important;
        }
    }
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label,
    [data-testid="stSidebar"] [data-testid="stCheckbox"] label p {
        white-space: normal !important;
        overflow-wrap: normal !important;
        word-break: normal !important;
        hyphens: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stNumberInput"] label,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stTextInput"] label,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] label {
        min-height: 1.35rem !important;
        margin-bottom: 0.05rem !important;
        align-items: flex-start !important;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stNumberInput"] label p,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stTextInput"] label p,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"] label p {
        line-height: 1.12 !important;
        margin-bottom: 0 !important;
        white-space: normal !important;
        overflow-wrap: normal !important;
        word-break: normal !important;
        hyphens: none !important;
    }
    [data-testid="stSidebar"] button[kind="primary"],
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"],
    [data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: #2f343b !important;
        border-color: #2f343b !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] button[kind="primary"]:hover,
    [data-testid="stSidebar"] [data-testid="stBaseButton-primary"]:hover,
    [data-testid="stSidebar"] div.stButton > button[kind="primary"]:hover {
        background: #1f2328 !important;
        border-color: #1f2328 !important;
        color: #ffffff !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:hover,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:hover,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:focus,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:focus,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:focus-visible,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:focus-visible,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:active,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:active {
        border-color: #2f343b !important;
        background: #2f343b !important;
        color: #ffffff !important;
        box-shadow: 0 0 0 0.15rem rgba(47, 52, 59, 0.20) !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:hover svg,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:hover svg,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:focus svg,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:focus svg,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepUp"]:active svg,
    [data-testid="stSidebar"] [data-testid="stNumberInputStepDown"]:active svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
    }
    [data-testid="stSidebar"] input[type="checkbox"] {
        accent-color: #2f343b !important;
    }
    [data-testid="stSidebar"] [data-testid="stCheckbox"]:has(input:checked) label > span:first-child,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) > div:first-child {
        background: #2f343b !important;
        border-color: #2f343b !important;
    }
    [data-testid="stSidebar"] [data-testid="stCheckbox"]:has(input:checked) svg,
    [data-testid="stSidebar"] [data-testid="stCheckbox"]:has(input:checked) svg path,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) svg,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) svg path {
        color: #2f343b !important;
        fill: #2f343b !important;
        stroke: #2f343b !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


STATE_KEYS = [
    "app_result",
    "app_file_bytes",
    "app_filename",
    "app_fields",
    "app_prewhitening_periods",
    "app_advanced_result",
    "app_model_lab_result",
    "app_show_model",
    "app_show_help",
    "app_show_bibliography",
    "app_report_previous_pdf",
]

HELP_FILE = "HELP.md"


def return_from_help_button(key: str) -> None:
    if st.button("Back to analysis", use_container_width=True, key=key):
        st.session_state["app_show_help"] = False
        st.rerun()


def render_help_document() -> None:
    st.divider()
    return_from_help_button("app_help_back_top")
    st.divider()
    try:
        with open(HELP_FILE, "r", encoding="utf-8") as handle:
            st.markdown(handle.read())
    except OSError:
        st.error("Help documentation is not available in this deployment.")
    st.divider()
    return_from_help_button("app_help_back_bottom")


def return_from_bibliography_button(key: str) -> None:
    if st.button("Back to analysis", use_container_width=True, key=key):
        st.session_state["app_show_bibliography"] = False
        st.rerun()


def current_object_name(prefix: str = "l1") -> str:
    fields = st.session_state.get("app_fields", {})
    candidates = [
        st.session_state.get(f"{prefix}_target_name", ""),
        fields.get("target_name", "") if isinstance(fields, dict) else "",
        st.session_state.get("report_object_name", ""),
    ]
    for value in candidates:
        text = str(value).strip()
        if text:
            return text
    return ""


def bibliography_search_urls(object_name: str) -> dict[str, str]:
    keywords = '"time series" OR period OR periods OR periodicity OR timing OR variability'
    ads_query = f'"{object_name}" AND ({keywords})'
    arxiv_query = f'all:"{object_name}" AND (all:"time series" OR all:period OR all:periodicity OR all:timing OR all:variability)'
    return {
        "ADS": "https://ui.adsabs.harvard.edu/search/q=" + quote(ads_query),
        "arXiv": "https://arxiv.org/search/astro-ph?query=" + quote(object_name) + "&searchtype=all",
        "arXiv API": "https://export.arxiv.org/api/query?" + urlencode({
            "search_query": arxiv_query,
            "start": 0,
            "max_results": 20,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }),
    }


def trusted_ssl_context() -> ssl.SSLContext:
    try:
        import certifi
    except Exception:
        return ssl.create_default_context()
    return ssl.create_default_context(cafile=certifi.where())


@st.cache_data(ttl=3600, show_spinner=False)
def arxiv_bibliography_search(object_name: str) -> list[dict[str, str]]:
    urls = bibliography_search_urls(object_name)
    with urlopen(urls["arXiv API"], timeout=12, context=trusted_ssl_context()) as response:
        raw = response.read()
    root = ET.fromstring(raw)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    rows: list[dict[str, str]] = []
    for entry in root.findall("atom:entry", ns):
        title = " ".join((entry.findtext("atom:title", default="", namespaces=ns) or "").split())
        summary = " ".join((entry.findtext("atom:summary", default="", namespaces=ns) or "").split())
        published = (entry.findtext("atom:published", default="", namespaces=ns) or "")[:10]
        link = ""
        for link_node in entry.findall("atom:link", ns):
            if link_node.attrib.get("rel") == "alternate":
                link = link_node.attrib.get("href", "")
                break
        authors = []
        for author in entry.findall("atom:author", ns):
            name = author.findtext("atom:name", default="", namespaces=ns)
            if name:
                authors.append(name)
        keyword_hits = [
            word for word in ["time series", "period", "periodicity", "timing", "variability", "Lomb", "folding"]
            if word.lower() in f"{title} {summary}".lower()
        ]
        rows.append({
            "published": published,
            "title": title,
            "authors": ", ".join(authors[:4]) + (" et al." if len(authors) > 4 else ""),
            "matched_terms": ", ".join(keyword_hits) if keyword_hits else "object match",
            "url": link,
            "summary": summary,
        })
    return rows


def render_bibliography_search(prefix: str = "l1") -> None:
    st.divider()
    return_from_bibliography_button("app_biblio_back_top")
    st.divider()
    object_name = current_object_name(prefix)
    st.subheader("Bibliographic Search")
    if not object_name:
        st.info("Enter an object name in the Input panel, or in the report metadata, before running the bibliographic search.")
        st.divider()
        return_from_bibliography_button("app_biblio_back_bottom")
        return

    urls = bibliography_search_urls(object_name)
    st.caption(f"Object searched: {object_name}")
    st.markdown(
        f"[Open ADS search]({urls['ADS']}) | [Open arXiv search]({urls['arXiv']})"
    )
    st.caption("The automatic list below uses arXiv results filtered with time-series and periodicity terms. ADS is linked for a fuller astronomy literature search.")
    try:
        rows = arxiv_bibliography_search(object_name)
    except Exception as exc:
        st.warning(f"Automatic arXiv search is not available right now: {exc}")
        rows = []
    if not rows:
        st.info("No automatic arXiv matches were returned. Use the ADS/arXiv links above to continue the literature search.")
    else:
        table = pd.DataFrame([
            {
                "Date": row["published"],
                "Title": row["title"],
                "Authors": row["authors"],
                "Matched terms": row["matched_terms"],
                "URL": row["url"],
            }
            for row in rows
        ])
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.markdown("**Abstract snippets**")
        for index, row in enumerate(rows[:10], start=1):
            with st.expander(f"{index}. {row['title']}", expanded=False):
                st.caption(f"{row['published']} | {row['authors']} | {row['matched_terms']}")
                st.write(row["summary"])
                if row["url"]:
                    st.markdown(f"[Open record]({row['url']})")
    st.divider()
    return_from_bibliography_button("app_biblio_back_bottom")


def file_signature(uploaded_file) -> tuple[str, int, str]:
    data = uploaded_file.getvalue()
    return uploaded_file.name, len(data), hashlib.sha256(data).hexdigest()


def clear_state(reset_file: bool = False) -> None:
    for key in STATE_KEYS:
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if key.startswith("evidence_"):
            st.session_state.pop(key, None)
    st.session_state["app_prewhitening_periods"] = []
    if reset_file:
        st.session_state.pop("app_file_signature", None)
        st.session_state["app_upload_key"] = st.session_state.get("app_upload_key", 0) + 1


def pdf_file_name(value: str, default: str = "periodicity_workbench_report.pdf") -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", ".", " "} else "_" for char in value.strip())
    cleaned = "_".join(cleaned.split())
    if not cleaned:
        cleaned = default
    if not cleaned.lower().endswith(".pdf"):
        cleaned = f"{cleaned}.pdf"
    return cleaned


def parse_periods(text: str) -> list[float]:
    values = []
    for chunk in text.replace(",", " ").replace(";", " ").split():
        value = float(chunk)
        if value <= 0:
            raise ValueError("Periods must be positive")
        values.append(value)
    return values


def unique_periods(periods: list[float], tolerance: float = 1e-5) -> list[float]:
    out = []
    for period in periods:
        if not any(abs(period - old) <= tolerance * max(period, old) for old in out):
            out.append(period)
    return out


def axis_labels(time_unit: str) -> dict[str, str]:
    if (time_unit or "days").lower().startswith("sec"):
        return {"time": "Time [s]", "period": "Period [s]", "frequency": "Frequency [Hz]", "baseline": "s"}
    return {"time": "Time [d]", "period": "Period [d]", "frequency": "Frequency [cycles/day]", "baseline": "d"}


def resolve_target_name(name: str) -> tuple[float, float, str]:
    target = name.strip()
    if not target:
        raise ValueError("Enter an object name before resolving coordinates.")
    coord = SkyCoord.from_name(target).icrs
    return float(coord.ra.deg), float(coord.dec.deg), coord.to_string("hmsdms", precision=2)


def optional_state_float(prefix: str, name: str) -> float | None:
    value = str(st.session_state.get(f"{prefix}_{name}", "")).strip()
    if not value:
        return None
    return float(value)


def flux_is_magnitude(result: dict | None = None) -> bool:
    if result is not None and "flux_is_magnitude" in result:
        return bool(result["flux_is_magnitude"])
    return bool(st.session_state.get("l1_flux_is_magnitude", False))


def flux_axis_title(result: dict | None = None) -> str:
    return "Magnitude" if flux_is_magnitude(result) else "Flux"


def apply_flux_axis(fig: go.Figure, result: dict | None = None, row: int | None = None, col: int | None = None) -> None:
    if flux_is_magnitude(result):
        if row is not None and col is not None:
            fig.update_yaxes(autorange="reversed", row=row, col=col)
        else:
            fig.update_yaxes(autorange="reversed")


def selected_fits_columns(prefix: str) -> tuple[int | None, str | None, str | None, str | None]:
    extension = st.session_state.get(f"{prefix}_fits_extension")
    time_col = st.session_state.get(f"{prefix}_fits_time_col")
    flux_col = st.session_state.get(f"{prefix}_fits_flux_col")
    use_error = bool(st.session_state.get(f"{prefix}_use_error", True))
    error_col = st.session_state.get(f"{prefix}_fits_error_col") if use_error else None
    if extension is None or not time_col or not flux_col:
        return None, None, None, None
    if error_col in {"", "None"}:
        error_col = None
    return int(extension), str(time_col), str(flux_col), None if error_col is None else str(error_col)


def selected_light_curve(
    file_bytes: bytes,
    filename: str,
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if is_fits_upload(filename, file_bytes):
        extension, time_col, flux_col, error_col = selected_fits_columns(prefix)
        return read_columns(
            file_bytes,
            filename,
            0,
            1,
            None,
            fits_extension=extension,
            fits_time_col=time_col,
            fits_flux_col=flux_col,
            fits_error_col=error_col,
        )
    time_col = int(st.session_state.get(f"{prefix}_time_col", 1)) - 1
    flux_col = int(st.session_state.get(f"{prefix}_flux_col", 2)) - 1
    use_error = bool(st.session_state.get(f"{prefix}_use_error", True))
    error_col = int(st.session_state.get(f"{prefix}_error_col", 3)) - 1 if use_error else None
    return read_columns(file_bytes, filename, time_col, flux_col, error_col)


def preferred_fits_column(columns: list[str], keywords: list[str], fallback_index: int = 0) -> str:
    upper_map = {column.upper(): column for column in columns}
    for keyword in keywords:
        if keyword.upper() in upper_map:
            return upper_map[keyword.upper()]
    for keyword in keywords:
        for column in columns:
            if keyword.upper() in column.upper():
                return column
    return columns[min(fallback_index, len(columns) - 1)]


def preferred_fits_extension(metadata: list[dict]) -> int:
    def score(row: dict) -> tuple[int, int, int, int, int]:
        column_names = [str(column["name"]).upper() for column in row.get("columns", [])]
        has_time = any(name in column_names for name in ["TIME", "MJD", "JD", "BJD"])
        has_flux = any(
            name in column_names
            for name in ["RATE", "COUNT_RATE", "FLUX", "MAG", "COUNTS", "TOT_COUNTS"]
        )
        has_error = any(
            name in column_names
            for name in ["ERROR", "ERR", "RATE_ERR", "FLUX_ERR", "MAG_ERR", "SIGMA"]
        )
        n_rows = int(row.get("n_rows", 0))
        return (int(n_rows >= 10), int(has_time), int(has_flux), int(has_error), n_rows)

    return int(max(metadata, key=score)["index"])


def render_fits_selector(prefix: str, file_bytes: bytes, filename: str) -> None:
    if not is_fits_upload(filename, file_bytes):
        return
    metadata = fits_table_metadata(file_bytes)
    st.caption("FITS table selection")
    extension_options = [row["index"] for row in metadata]
    extension_by_index = {row["index"]: row for row in metadata}
    preferred_extension = preferred_fits_extension(metadata)
    current_extension = st.session_state.get(f"{prefix}_fits_extension")
    if current_extension not in extension_options:
        current_extension = preferred_extension
        st.session_state[f"{prefix}_fits_extension"] = current_extension
    selected_extension = st.selectbox(
        "FITS extension",
        options=extension_options,
        index=extension_options.index(current_extension),
        format_func=lambda idx: f"{extension_by_index[idx]['label']} ({extension_by_index[idx]['n_rows']} rows)",
        key=f"{prefix}_fits_extension",
    )
    selected_meta = extension_by_index[int(selected_extension)]
    column_rows = selected_meta["columns"]
    column_names = [row["name"] for row in column_rows]
    for key_suffix, keywords, fallback in [
        ("fits_time_col", ["TIME", "MJD", "JD", "BJD"], 0),
        ("fits_flux_col", ["RATE", "COUNT_RATE", "FLUX", "MAG", "COUNTS", "TOT_COUNTS"], 1),
        ("fits_error_col", ["ERROR", "ERR", "RATE_ERR", "FLUX_ERR", "MAG_ERR", "SIGMA"], 2),
    ]:
        key = f"{prefix}_{key_suffix}"
        if st.session_state.get(key) not in column_names:
            st.session_state[key] = preferred_fits_column(column_names, keywords, fallback)
    cols = st.columns(3)
    cols[0].selectbox("Time column", options=column_names, key=f"{prefix}_fits_time_col")
    cols[1].selectbox("Flux column", options=column_names, key=f"{prefix}_fits_flux_col")
    error_options = ["None"] + column_names
    error_value = st.session_state.get(f"{prefix}_fits_error_col", "None")
    if error_value not in error_options:
        error_value = "None"
    cols[2].selectbox(
        "Error column",
        options=error_options,
        index=error_options.index(error_value),
        disabled=not bool(st.session_state.get(f"{prefix}_use_error", True)),
        key=f"{prefix}_fits_error_col",
    )
    st.dataframe(
        pd.DataFrame(column_rows),
        use_container_width=True,
        hide_index=True,
        height=180,
    )


def suggested_frequency_range_from_bytes(file_bytes: bytes | None, time_column: int, time_unit: str = "days") -> tuple[float, float, str] | None:
    if file_bytes is None:
        return None
    try:
        text = file_bytes.decode("ascii")
        numeric_text = "\n".join(
            line for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith(("#", "%"))
        )
        data = np.genfromtxt(io.StringIO(numeric_text.replace(",", " ")), invalid_raise=False)
    except Exception:
        return None
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2 or data.shape[0] < 3 or time_column < 1 or time_column > data.shape[1]:
        return None
    time = np.sort(np.asarray(data[:, time_column - 1], dtype=float))
    time = time[np.isfinite(time)]
    dt = np.diff(time)
    dt = dt[dt > 0]
    if len(time) < 3 or len(dt) == 0:
        return None
    baseline = float(time[-1] - time[0])
    median_dt = float(np.median(dt))
    if baseline <= 0 or median_dt <= 0:
        return None
    fmin = max(1.0 / baseline, 1e-5)
    fmax = min(0.5 / median_dt, 20.0)
    if fmax <= fmin:
        fmax = fmin * 10.0
    labels = axis_labels(time_unit)
    return fmin, fmax, f"Suggested from time span {baseline:.1f} {labels['baseline']} and median sampling {median_dt:.3g} {labels['baseline']}."


def suggested_frequency_range(uploaded_file, time_column: int, time_unit: str = "days") -> tuple[float, float, str] | None:
    return suggested_frequency_range_from_bytes(None if uploaded_file is None else uploaded_file.getvalue(), time_column, time_unit)


def file_preview_dataframe(file_bytes: bytes, filename: str, prefix: str, max_rows: int = 12) -> pd.DataFrame:
    if is_fits_upload(filename, file_bytes):
        extension, time_col, flux_col, error_col = selected_fits_columns(prefix)
        if extension is None or time_col is None or flux_col is None:
            return pd.DataFrame()
        time, flux, error = selected_light_curve(file_bytes, filename, prefix)
        data = {
            str(time_col): time[:max_rows],
            str(flux_col): flux[:max_rows],
        }
        if error_col is not None:
            data[str(error_col)] = error[:max_rows]
        return pd.DataFrame(data)
    text = validate_upload(filename, file_bytes)
    rows = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "%"))
    ][:max_rows]
    if not rows:
        return pd.DataFrame()
    data = np.genfromtxt(io.StringIO("\n".join(rows).replace(",", " ")), invalid_raise=False)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        return pd.DataFrame()
    return pd.DataFrame(data, columns=[f"Column {index}" for index in range(1, data.shape[1] + 1)])


def raw_preview_figure(
    file_bytes: bytes,
    filename: str,
    prefix: str,
) -> go.Figure:
    use_error = bool(st.session_state.get(f"{prefix}_use_error", True))
    time, flux, error = selected_light_curve(file_bytes, filename, prefix)
    good = np.ones_like(time, dtype=bool)
    for name, op in [("xmin", np.greater_equal), ("xmax", np.less_equal)]:
        value = optional_state_float(prefix, name)
        if value is not None:
            good &= op(time, value)
    for name, op in [("ymin", np.greater_equal), ("ymax", np.less_equal)]:
        value = optional_state_float(prefix, name)
        if value is not None:
            good &= op(flux, value)
    labels = axis_labels(st.session_state.get(f"{prefix}_time_unit", "days"))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=time[good],
            y=flux[good],
            error_y=dict(type="data", array=error[good], visible=use_error),
            mode="markers",
            marker=dict(color="#20242a", size=4),
            name="Selected data",
            hovertemplate="Time=%{x:.6g}<br>Value=%{y:.6g}<extra></extra>",
        )
    )
    if np.any(~good):
        fig.add_trace(
            go.Scatter(
                x=time[~good],
                y=flux[~good],
                mode="markers",
                marker=dict(color="#b8bec8", size=3, opacity=0.45),
                name="Outside limits",
                hovertemplate="Time=%{x:.6g}<br>Value=%{y:.6g}<extra></extra>",
            )
        )
    fig.update_layout(
        title="Uploaded light curve preview",
        xaxis_title=labels["time"],
        yaxis_title=flux_axis_title(),
        height=300,
        margin=dict(l=20, r=20, t=45, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    apply_flux_axis(fig)
    return frame(fig)


def suggested_frequency_range_from_time(time_values: list[float] | np.ndarray, time_unit: str = "days") -> tuple[float, float, str] | None:
    time = np.sort(np.asarray(time_values, dtype=float))
    time = time[np.isfinite(time)]
    dt = np.diff(time)
    dt = dt[dt > 0]
    if len(time) < 3 or len(dt) == 0:
        return None
    baseline = float(time[-1] - time[0])
    median_dt = float(np.median(dt))
    if baseline <= 0 or median_dt <= 0:
        return None
    fmin = max(1.0 / baseline, 1e-5)
    fmax = min(0.5 / median_dt, 20.0)
    if fmax <= fmin:
        fmax = fmin * 10.0
    labels = axis_labels(time_unit)
    return fmin, fmax, f"Suggested from time span {baseline:.1f} {labels['baseline']} and median sampling {median_dt:.3g} {labels['baseline']}."


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace({np.nan: None})


def parallel_series_dataframe(columns: dict[str, object]) -> pd.DataFrame:
    """Build a table from parallel arrays, omitting unavailable optional series."""
    if not columns:
        return pd.DataFrame()
    first_name, first_values = next(iter(columns.items()))
    reference = [] if first_values is None else first_values
    target_length = len(reference)
    aligned = {first_name: reference}
    for name, values in list(columns.items())[1:]:
        series = [] if values is None else values
        if len(series) == target_length:
            aligned[name] = series
    return pd.DataFrame(aligned)


def peaks_dataframe(peaks: list[dict]) -> pd.DataFrame:
    return clean_dataframe(pd.DataFrame([
        {
            "label": peak["label"],
            "period": peak["period"],
            "period_error": peak["period_error"],
            "frequency": peak["frequency"],
            "frequency_error": peak["frequency_error"],
            "power": peak["power"],
            "FAP": f"{float(peak['fap']):.5f}",
            "type": peak["kind"],
            "window_period": peak["window_period"],
        }
        for peak in peaks
    ]))


def harmonic_diagnostics_dataframe(rows: list[dict]) -> pd.DataFrame:
    return clean_dataframe(pd.DataFrame([
        {
            "relation": row.get("relation"),
            "family": row.get("family"),
            "target_period": row.get("target_period"),
            "nearest_period": row.get("nearest_period"),
            "delta_%": row.get("period_delta_percent"),
            "power": row.get("power"),
            "FAP": None if row.get("fap") is None else f"{float(row['fap']):.5f}",
            "window_power": row.get("window_power"),
            "detected": row.get("detected"),
            "note": row.get("note"),
        }
        for row in rows
    ]))


def prewhitening_summary_dataframe(summary: dict) -> pd.DataFrame:
    if not summary:
        return pd.DataFrame()
    return clean_dataframe(pd.DataFrame([
        {
            "terms": summary.get("n_terms"),
            "parameters": summary.get("n_parameters"),
            "RMS before": summary.get("rms_before"),
            "RMS after": summary.get("rms_after"),
            "RMS reduction [%]": summary.get("rms_reduction_percent"),
            "LS peak before": summary.get("max_power_before"),
            "LS peak after": summary.get("max_power_after"),
            "LS peak reduction [%]": summary.get("max_power_reduction_percent"),
            "AIC": summary.get("AIC"),
            "BIC": summary.get("BIC"),
        }
    ]))


def period_options(result: dict | None, key: str) -> dict[float, str]:
    if not result:
        return {}
    period_unit = result.get("period_unit", "d")
    options = {}
    for peak in result.get(key, []):
        period = float(peak["period"])
        options[period] = f"{peak['label']} - {period:.4g} {period_unit} ({peak['kind']})"
    return options


def exclusion_options(result: dict | None) -> dict[float, str]:
    if not result:
        return {}
    period_unit = result.get("period_unit", "d")
    options = {}
    for group, key in [("detected", "peaks"), ("prewhitened", "residual_peaks")]:
        for peak in result.get(key, []):
            period = float(peak["period"])
            options[period] = f"{group}: {peak['label']} - {period:.4g} {period_unit} ({peak['kind']})"
    return options


def dataframe_download(label: str, df: pd.DataFrame, filename: str, key: str) -> None:
    st.download_button(
        label,
        data=df.to_csv(sep="\t", index=False).encode("utf-8"),
        file_name=filename,
        mime="text/plain",
        key=key,
        use_container_width=True,
    )


def y_range(values) -> list[float] | None:
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return None
    ymax = float(np.max(arr))
    if ymax <= 0:
        return None
    return [0, ymax * 1.25]


def frame(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(showline=True, linewidth=1, linecolor="#20242a", mirror=True, ticks="outside")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="#20242a", mirror=True, ticks="outside")
    return fig


def add_markers(fig: go.Figure, peaks: list[dict], y_key: str = "power", period_unit: str = "d") -> None:
    for peak in peaks:
        color = "#b13b32" if "artefact" in peak["kind"] else "#777777" if "excluded" in peak["kind"] else "#2457a6"
        if peak.get(y_key) is None:
            continue
        fig.add_vline(x=peak["period"], line_dash="dash", line_color=color, opacity=0.7)
        fig.add_trace(go.Scatter(
            x=[peak["period"]],
            y=[peak[y_key]],
            mode="markers+text",
            marker=dict(color=color, size=7),
            text=[f"{peak['period']:.3g} {period_unit}"],
            textposition="top right",
            showlegend=False,
        ))


def periodogram(result: dict, key: str, peaks_key: str, title: str) -> go.Figure:
    series = result["series"]
    fig = go.Figure(go.Scatter(x=series["period"], y=series[key], mode="lines", line=dict(color="#20242a", width=1.2)))
    add_markers(fig, result.get(peaks_key, []), period_unit=result.get("period_unit", "d"))
    yr = y_range([peak.get("power") for peak in result.get(peaks_key, [])]) or y_range(series[key])
    fig.update_layout(title=title, xaxis_title=result.get("period_label", "Period [d]"), yaxis_title="Lomb-Scargle power", height=430, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
    if yr:
        fig.update_yaxes(range=yr)
    return frame(fig)


def window_plot(result: dict) -> go.Figure:
    series = result["series"]
    peaks = result.get("window_peaks", [])
    fig = go.Figure(go.Scatter(x=series["period"], y=series["window_power"], mode="lines", line=dict(color="#20242a", width=1.2)))
    if peaks:
        fig.add_trace(go.Scatter(x=[p["period"] for p in peaks], y=[p["power"] for p in peaks], mode="markers", marker=dict(color="#b13b32", size=5), showlegend=False))
    for peak in peaks[:50]:
        fig.add_vline(x=peak["period"], line_dash="dash", line_color="#b13b32", opacity=0.2)
    yr = y_range([p["power"] for p in peaks]) or y_range(series["window_power"])
    fig.update_layout(title="Sampling window", xaxis_title=result.get("period_label", "Period [d]"), yaxis_title="Sampling-window power", height=430, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
    if yr:
        fig.update_yaxes(range=yr)
    return frame(fig)


def folded_plot(result: dict) -> go.Figure:
    series = result["series"]
    fig = go.Figure()
    has_error_column = bool(result.get("has_error_column", True))
    phase = np.asarray(series["fold_phase"], dtype=float)
    flux = np.asarray(series["fold_flux"], dtype=float)
    error = np.asarray(series["fold_error"], dtype=float)
    phase_2 = np.concatenate([phase, phase + 1.0]) if len(phase) else phase
    flux_2 = np.concatenate([flux, flux]) if len(flux) else flux
    error_2 = np.concatenate([error, error]) if len(error) else error
    model_phase = np.asarray(series["fold_model_phase"], dtype=float)
    model_flux = np.asarray(series["fold_model_flux"], dtype=float)
    if len(model_phase) and np.nanmax(model_phase) <= 1.01:
        model_phase = np.concatenate([model_phase, model_phase + 1.0])
        model_flux = np.concatenate([model_flux, model_flux])
    fig.add_trace(go.Scatter(
        x=phase_2,
        y=flux_2,
        error_y=dict(type="data", array=error_2, visible=has_error_column),
        mode="markers",
        marker=dict(color="#20242a", size=7),
    ))
    fig.add_trace(go.Scatter(x=model_phase, y=model_flux, mode="lines", line=dict(color="#2457a6", width=2, dash="dot")))
    for maximum in result.get("folded_maxima", []):
        for offset in [0.0, 1.0]:
            fig.add_vline(x=maximum["phase"] + offset, line_dash="dash", line_color="#2457a6", opacity=0.75)
    fig.update_layout(title="Folded profile", xaxis_title="Orbital phase", yaxis_title=f"Weighted mean {flux_axis_title(result).lower()}", height=430, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
    fig.update_xaxes(range=[0, 2])
    apply_flux_axis(fig, result)
    return frame(fig)


def folded_period_table(result: dict) -> pd.DataFrame:
    folded_period = result.get("folded_period")
    fit_periods = result.get("fold_fit_periods", [])
    ratios = result.get("fold_fit_ratios", [])
    rows = []
    for index, period in enumerate(fit_periods, start=1):
        rows.append(
            {
                "role": "folding period" if index == 1 and folded_period == period else f"fit period {index}",
                "period": period,
                "frequency_ratio_in_phase": ratios[index - 1] if index - 1 < len(ratios) else None,
            }
        )
    if folded_period is not None and not any(abs(float(folded_period) - float(row["period"])) < 1e-8 for row in rows):
        rows.insert(0, {"role": "folding period", "period": folded_period, "frequency_ratio_in_phase": 1.0})
    return clean_dataframe(pd.DataFrame(rows))


def folded_fit_summary(result: dict) -> pd.DataFrame:
    terms = result.get("fold_fit_terms", [])
    periods = [float(value) for value in result.get("fold_fit_periods", [])]
    if not terms:
        return pd.DataFrame()
    rows = []
    for idx, term in enumerate(terms):
        row = dict(term)
        row["period"] = periods[idx] if idx < len(periods) else None
        rows.append(row)
    return clean_dataframe(pd.DataFrame(rows))


def folded_periods_and_parameters(result: dict) -> pd.DataFrame:
    periods = folded_period_table(result)
    params = folded_fit_summary(result)
    if periods.empty:
        return params
    if params.empty:
        return periods
    merged = periods.reset_index(drop=True).copy()
    params = params.reset_index(drop=True)
    for column in params.columns:
        if column not in merged.columns:
            merged[column] = params[column]
    return clean_dataframe(merged)


def fit_equation_text(terms: list[dict], variable: str = "t") -> str:
    if not terms:
        return ""
    offset = float(terms[0].get("offset", 0.0) or 0.0)
    if len(terms) > 3:
        return f"y({variable}) = C + sum_i [a_i cos(2 pi f_i {variable}) + b_i sin(2 pi f_i {variable})]"
    pieces = [f"y({variable}) = {offset:.6g}"]
    for idx, term in enumerate(terms, start=1):
        freq = term.get("frequency", term.get("frequency_ratio"))
        cos_coeff = float(term.get("cos_coeff", 0.0) or 0.0)
        sin_coeff = float(term.get("sin_coeff", 0.0) or 0.0)
        pieces.append(f"{cos_coeff:+.6g} cos(2 pi * {float(freq):.6g} * {variable})")
        pieces.append(f"{sin_coeff:+.6g} sin(2 pi * {float(freq):.6g} * {variable})")
    return " ".join(pieces)


def global_fit_summary(terms: list[dict]) -> pd.DataFrame:
    if not terms:
        return pd.DataFrame()
    first = terms[0]
    return clean_dataframe(pd.DataFrame([{
        "method": first.get("fit_method"),
        "offset": first.get("offset"),
        "rms": first.get("rms"),
        "weighted_rms": first.get("weighted_rms"),
        "chi2_red": first.get("chi2_red"),
    }]))


def evidence_period_rows(prefix: str, key: str) -> list[dict[str, str]]:
    rows = []
    period = str(st.session_state.get(f"evidence_{key}_period", "")).strip()
    error = str(st.session_state.get(f"evidence_{key}_error", "")).strip()
    if period or error:
        rows.append({"period": period, "error": error})
    count = int(st.session_state.get(f"evidence_{key}_extra_count", 0))
    for index in range(count):
        period = str(st.session_state.get(f"evidence_{key}_extra_period_{index}", "")).strip()
        error = str(st.session_state.get(f"evidence_{key}_extra_error_{index}", "")).strip()
        if period or error:
            rows.append({"period": period, "error": error})
    return rows


def render_frequency_evidence(prefix: str, result: dict | None) -> None:
    with st.expander("Save evidences", expanded=False):
        st.caption("These notes are kept while the widget is closed and can be included in the PDF report.")
        cols = st.columns(2)
        default_period = "" if not result or result.get("primary_period") is None else f"{float(result['primary_period']):.8g}"
        if f"evidence_frequency_period" not in st.session_state and default_period:
            st.session_state["evidence_frequency_period"] = default_period
        cols[0].text_input("Period", key="evidence_frequency_period")
        cols[1].text_input("Error", key="evidence_frequency_error")
        count = int(st.session_state.get("evidence_frequency_extra_count", 0))
        for index in range(count):
            extra_cols = st.columns(2)
            extra_cols[0].text_input(f"Extra period {index + 1}", key=f"evidence_frequency_extra_period_{index}")
            extra_cols[1].text_input(f"Extra error {index + 1}", key=f"evidence_frequency_extra_error_{index}")
        if st.button("Add extra period", key="evidence_frequency_add_period"):
            st.session_state["evidence_frequency_extra_count"] = count + 1
            st.rerun()
        st.text_area(
            "Comments on period(s) found and comparison to literature:",
            key="evidence_frequency_comments",
            height=130,
        )


def render_prewhitening_evidence() -> None:
    with st.expander("Save evidences", expanded=False):
        chain = st.session_state.get("app_prewhitening_periods", [])
        if "evidence_prewhitening_periods" not in st.session_state and chain:
            st.session_state["evidence_prewhitening_periods"] = ", ".join(f"{float(period):.8g}" for period in chain)
        st.text_area("Periods used:", key="evidence_prewhitening_periods", height=90)
        st.text_area(
            "Comments on used periods and comparison to literature:",
            key="evidence_prewhitening_comments",
            height=130,
        )


def render_tomography_evidence(prefix: str) -> None:
    with st.expander("Save evidences", expanded=False):
        cols = st.columns(2)
        if "evidence_tomography_window_width" not in st.session_state:
            st.session_state["evidence_tomography_window_width"] = str(st.session_state.get(f"{prefix}_advanced_width", ""))
        if "evidence_tomography_window_step" not in st.session_state:
            st.session_state["evidence_tomography_window_step"] = str(st.session_state.get(f"{prefix}_advanced_step", ""))
        cols[0].text_input("Window width", key="evidence_tomography_window_width")
        cols[1].text_input("Window step", key="evidence_tomography_window_step")
        st.text_area("Comments on tomography results:", key="evidence_tomography_comments", height=130)


def render_model_lab_evidence(model_result: dict | None) -> None:
    with st.expander("Save evidences", expanded=False):
        family = str((model_result or {}).get("family", "selected model")).replace("_", " ")
        st.caption(f"Evidence for: {family}")
        st.text_area(
            "Comments and comparison to literature:",
            key="evidence_model_lab_comments",
            height=150,
        )


def dataframe_to_lines(df: pd.DataFrame, max_rows: int = 12) -> list[str]:
    if df is None or df.empty:
        return ["Not available."]
    small = df.head(max_rows).copy()
    for column in small.columns:
        small[column] = small[column].map(lambda value: "" if value is None else f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value))
    return small.to_string(index=False).splitlines()


def report_metadata() -> dict[str, str]:
    return {
        "object": str(st.session_state.get("report_object_name", "")).strip(),
        "student": str(st.session_state.get("report_student_name", "")).strip(),
        "course": str(st.session_state.get("report_course", "")).strip(),
        "filename": pdf_file_name(str(st.session_state.get("report_output_name", ""))),
    }


def report_selected_sections() -> dict[str, bool]:
    return {
        "frequency": bool(st.session_state.get("report_include_frequency", True)),
        "prewhitening": bool(st.session_state.get("report_include_prewhitening", True)),
        "tomography": bool(st.session_state.get("report_include_tomography", True)),
        "model_lab": bool(st.session_state.get("report_include_model_lab", True)),
    }


def add_wrapped_text(ax, lines: list[str] | str, y_start: float = 0.94, size: int = 10, width: int = 95) -> None:
    if isinstance(lines, str):
        raw_lines = lines.splitlines() or [lines]
    else:
        raw_lines = [str(line) for line in lines]
    y = y_start
    for raw in raw_lines:
        wrapped = textwrap.wrap(raw, width=width) or [""]
        for line in wrapped:
            if y < 0.04:
                return
            ax.text(0.04, y, line, transform=ax.transAxes, fontsize=size, va="top", family="monospace")
            y -= 0.035
        y -= 0.012


def add_text_page(pdf: PdfPages, title: str, lines: list[str] | str) -> None:
    fig = Figure(figsize=(8.27, 11.69))
    ax = fig.subplots()
    ax.axis("off")
    ax.text(0.04, 0.985, title, transform=ax.transAxes, fontsize=17, fontweight="bold", va="top")
    add_wrapped_text(ax, lines, y_start=0.93)
    pdf.savefig(fig, bbox_inches="tight")


def add_frequency_plot_page(pdf: PdfPages, result: dict, include_folded: bool = True) -> None:
    series = result["series"]
    fig = Figure(figsize=(11.69, 8.27))
    axes = fig.subplots(1, 2 if include_folded else 1)
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    ax = axes[0]
    ax.plot(series["period"], series["power"], color="#20242a", lw=1.0)
    for peak in result.get("peaks", [])[:10]:
        ax.axvline(float(peak["period"]), color="#2457a6", ls="--", alpha=0.45)
    ax.set_title("Lomb-Scargle periodogram")
    ax.set_xlabel(result.get("period_label", "Period"))
    ax.set_ylabel("Power")
    ax.grid(alpha=0.25)
    if include_folded:
        ax = axes[1]
        phase = np.asarray(series.get("fold_phase", []), dtype=float)
        flux = np.asarray(series.get("fold_flux", []), dtype=float)
        error = np.asarray(series.get("fold_error", []), dtype=float)
        model_phase = np.asarray(series.get("fold_model_phase", []), dtype=float)
        model_flux = np.asarray(series.get("fold_model_flux", []), dtype=float)
        if len(phase):
            ax.errorbar(np.r_[phase, phase + 1], np.r_[flux, flux], yerr=np.r_[error, error], fmt="o", ms=3, color="#20242a", ecolor="#8b93a5", lw=0.7)
        if len(model_phase):
            ax.plot(np.r_[model_phase, model_phase + 1], np.r_[model_flux, model_flux], color="#2457a6", ls=":", lw=1.6)
        ax.set_xlim(0, 2)
        ax.set_title("Folded profile")
        ax.set_xlabel("Phase")
        ax.set_ylabel(flux_axis_title(result))
        ax.grid(alpha=0.25)
        if flux_is_magnitude(result):
            ax.invert_yaxis()
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")


def add_prewhitening_plot_page(pdf: PdfPages, result: dict) -> None:
    series = result["series"]
    fig = Figure(figsize=(11.69, 8.27))
    axes = fig.subplots(2, 1)
    axes[0].plot(series["period"], series["residual_power"], color="#20242a", lw=1.0)
    axes[0].set_title("Residual Lomb-Scargle after prewhitening")
    axes[0].set_xlabel(result.get("period_label", "Period"))
    axes[0].set_ylabel("Power")
    axes[0].grid(alpha=0.25)
    time = np.asarray(series.get("time", []), dtype=float)
    flux = np.asarray(series.get("flux", []), dtype=float)
    model = np.asarray(series.get("prewhitening_display_model_flux", series.get("prewhitening_model_flux", [])), dtype=float)
    if len(time) and len(model) == len(time):
        axes[1].scatter(time, flux, s=6, color="#20242a", label="Original data")
        axes[1].plot(time, model, color="#2457a6", lw=1.0, label="Model at data")
        axes[1].scatter(time, flux - model, s=6, color="#b13b32", label="Residuals")
    axes[1].set_title("Prewhitening model and residuals")
    axes[1].set_xlabel(result.get("time_label", "Time"))
    axes[1].set_ylabel(flux_axis_title(result))
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    if flux_is_magnitude(result):
        axes[1].invert_yaxis()
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")


def add_tomography_plot_page(pdf: PdfPages, advanced: dict) -> None:
    fig = Figure(figsize=(11.69, 8.27))
    ax = fig.subplots()
    period = np.asarray(advanced.get("period", []), dtype=float)
    centers = np.asarray(advanced.get("centers", []), dtype=float)
    values = np.asarray(advanced.get("values", []), dtype=float)
    if len(period) and len(centers) and values.size:
        mesh = ax.pcolormesh(period, centers, values, shading="auto", cmap="viridis")
        fig.colorbar(mesh, ax=ax, label=str(advanced.get("metric", "power")))
        if advanced.get("show_best_track", True) and advanced.get("best_period"):
            ax.plot(advanced["best_period"], centers, color="white", lw=1.2, alpha=0.9)
    ax.set_title("Period tomography")
    ax.set_xlabel(advanced.get("period_label", "Period"))
    ax.set_ylabel("Time")
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")


def add_model_lab_plot_page(pdf: PdfPages, model_result: dict, app_result: dict) -> None:
    fig = Figure(figsize=(11.69, 8.27))
    axes = fig.subplots(1, 2)
    phase = np.asarray(model_result.get("phase", []), dtype=float)
    flux = np.asarray(model_result.get("flux", []), dtype=float)
    error = np.asarray(model_result.get("error", []), dtype=float)
    model_phase = np.asarray(model_result.get("model_phase", []), dtype=float)
    model_flux = np.asarray(model_result.get("model_flux", []), dtype=float)
    if len(phase):
        axes[0].errorbar(np.r_[phase, phase + 1], np.r_[flux, flux], yerr=np.r_[error, error], fmt="o", ms=3, color="#20242a", ecolor="#8b93a5", lw=0.7)
    if len(model_phase):
        axes[0].plot(np.r_[model_phase, model_phase + 1], np.r_[model_flux, model_flux], color="#2457a6", lw=1.4)
    axes[0].set_xlim(0, 2)
    axes[0].set_title("Model laboratory folded fit")
    axes[0].set_xlabel("Phase")
    axes[0].set_ylabel(flux_axis_title(app_result))
    axes[0].grid(alpha=0.25)
    time = np.asarray(model_result.get("model_time", []), dtype=float)
    model_time_flux = np.asarray(model_result.get("model_time_flux", []), dtype=float)
    data_time = np.asarray(app_result.get("series", {}).get("time", []), dtype=float)
    data_flux = np.asarray(app_result.get("series", {}).get("flux", []), dtype=float)
    if len(data_time):
        axes[1].scatter(data_time, data_flux, s=5, color="#20242a", alpha=0.65, label="Data")
    if len(time):
        axes[1].plot(time, model_time_flux, color="#2457a6", lw=1.0, label="Model")
    axes[1].set_title("Time-domain model")
    axes[1].set_xlabel(app_result.get("time_label", "Time"))
    axes[1].set_ylabel(flux_axis_title(app_result))
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.25)
    if flux_is_magnitude(app_result):
        axes[0].invert_yaxis()
        axes[1].invert_yaxis()
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")


def pdf_table_frame(df: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 8) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({"info": ["Not available."]})
    table = df.copy()
    if columns:
        selected = [column for column in columns if column in table.columns]
        if selected:
            table = table[selected]
    table = table.head(max_rows)
    for column in table.columns:
        table[column] = table[column].map(
            lambda value: ""
            if value is None
            else f"{float(value):.5g}"
            if isinstance(value, (float, int, np.floating, np.integer)) and np.isfinite(float(value))
            else str(value)
        )
    return table


def estimate_text_height(text: str, width: int = 92, line_height: float = 0.023) -> float:
    lines = []
    for raw in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(raw, width=width) or [""])
    return min(0.36, 0.07 + line_height * max(1, len(lines)))


def draw_wrapped_lines(
    fig: Figure,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    size: float = 8.5,
    wrap_width: int = 92,
    line_gap: float = 0.09,
    paragraph_gap: float = 0.03,
) -> None:
    ax = fig.add_axes([x, y, width, height])
    ax.axis("off")
    cursor = 0.98
    for raw in str(text).splitlines() or [""]:
        for line in textwrap.wrap(raw, width=wrap_width) or [""]:
            if cursor < 0.02:
                return
            ax.text(0.0, cursor, line, transform=ax.transAxes, fontsize=size, va="top")
            cursor -= line_gap
        cursor -= paragraph_gap


def draw_table(fig: Figure, x: float, y: float, width: float, height: float, df: pd.DataFrame, font_size: float = 6.2) -> None:
    ax = fig.add_axes([x, y, width, height])
    ax.axis("off")
    table = pdf_table_frame(df)
    mpl_table = ax.table(
        cellText=table.values,
        colLabels=list(table.columns),
        loc="upper left",
        cellLoc="left",
        colLoc="left",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    for (row, _), cell in mpl_table.get_celld().items():
        cell.PAD = 0.02
        cell.set_edgecolor("#c9ced8")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#eef1f6")
            cell.set_text_props(weight="bold")


class CompactPdfReport:
    def __init__(self, pdf: PdfPages) -> None:
        self.pdf = pdf
        self.fig: Figure | None = None
        self.y = 0.95
        self.page_has_content = False
        self.new_page()

    def new_page(self) -> None:
        if self.fig is not None and self.page_has_content:
            self.pdf.savefig(self.fig)
        self.fig = Figure(figsize=(8.27, 11.69))
        self.y = 0.95
        self.page_has_content = False

    def finish(self) -> None:
        if self.fig is not None and self.page_has_content:
            self.pdf.savefig(self.fig)

    def block(self, title: str, height: float, drawer) -> None:
        height = min(height, 0.86)
        if self.y - height < 0.06 and self.page_has_content:
            self.new_page()
        assert self.fig is not None
        top = self.y
        bottom = top - height
        self.fig.text(0.06, top, title, fontsize=12, fontweight="bold", va="top")
        drawer(self.fig, 0.06, bottom + 0.02, 0.88, max(0.04, height - 0.065))
        self.y = bottom - 0.025
        self.page_has_content = True

    def text_block(
        self,
        title: str,
        text: str,
        min_height: float = 0.12,
        size: float = 8.5,
        wrap_width: int = 92,
        line_height: float = 0.023,
        line_gap: float = 0.09,
    ) -> None:
        height = max(min_height, estimate_text_height(text, width=wrap_width, line_height=line_height))
        self.block(title, height, lambda fig, x, y, w, h: draw_wrapped_lines(fig, x, y, w, h, text, size=size, wrap_width=wrap_width, line_gap=line_gap))

    def table_block(self, title: str, df: pd.DataFrame, columns: list[str] | None = None, max_rows: int = 8) -> None:
        table = pdf_table_frame(df, columns=columns, max_rows=max_rows)
        height = min(0.28, 0.065 + 0.026 * (len(table) + 1))
        self.block(title, height, lambda fig, x, y, w, h: draw_table(fig, x, y, w, h, table, font_size=5.8))


def draw_frequency_summary_plot(fig: Figure, x: float, y: float, width: float, height: float, result: dict) -> None:
    series = result["series"]
    ax1 = fig.add_axes([x, y, width * 0.48, height])
    ax1.plot(series["period"], series["power"], color="#20242a", lw=0.9)
    for peak in result.get("peaks", [])[:8]:
        ax1.axvline(float(peak["period"]), color="#2457a6", ls="--", alpha=0.35, lw=0.8)
    ax1.set_title("Lomb-Scargle", fontsize=9)
    ax1.set_xlabel(result.get("period_label", "Period"), fontsize=8)
    ax1.set_ylabel("Power", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.grid(alpha=0.22)

    ax2 = fig.add_axes([x + width * 0.53, y, width * 0.47, height])
    phase = np.asarray(series.get("fold_phase", []), dtype=float)
    flux = np.asarray(series.get("fold_flux", []), dtype=float)
    error = np.asarray(series.get("fold_error", []), dtype=float)
    model_phase = np.asarray(series.get("fold_model_phase", []), dtype=float)
    model_flux = np.asarray(series.get("fold_model_flux", []), dtype=float)
    if len(phase):
        ax2.errorbar(np.r_[phase, phase + 1], np.r_[flux, flux], yerr=np.r_[error, error], fmt="o", ms=2.4, color="#20242a", ecolor="#8b93a5", lw=0.45)
    if len(model_phase):
        ax2.plot(np.r_[model_phase, model_phase + 1], np.r_[model_flux, model_flux], color="#2457a6", ls=":", lw=1.2)
    ax2.set_xlim(0, 2)
    ax2.set_title("Folded profile", fontsize=9)
    ax2.set_xlabel("Phase", fontsize=8)
    ax2.set_ylabel(flux_axis_title(result), fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(alpha=0.22)
    if flux_is_magnitude(result):
        ax2.invert_yaxis()


def draw_prewhitening_summary_plot(fig: Figure, x: float, y: float, width: float, height: float, result: dict) -> None:
    series = result["series"]
    ax1 = fig.add_axes([x, y, width * 0.48, height])
    ax1.plot(series["period"], series["residual_power"], color="#20242a", lw=0.9)
    ax1.set_title("Residual LS", fontsize=9)
    ax1.set_xlabel(result.get("period_label", "Period"), fontsize=8)
    ax1.set_ylabel("Power", fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.grid(alpha=0.22)

    ax2 = fig.add_axes([x + width * 0.53, y, width * 0.47, height])
    time = np.asarray(series.get("time", []), dtype=float)
    flux = np.asarray(series.get("flux", []), dtype=float)
    model = np.asarray(series.get("prewhitening_display_model_flux", series.get("prewhitening_model_flux", [])), dtype=float)
    if len(time):
        ax2.scatter(time, flux, s=4, color="#20242a", alpha=0.55, label="Data")
    if len(time) and len(model) == len(time):
        ax2.plot(time, model, color="#2457a6", lw=0.8, label="Model")
    ax2.set_title("Model at data", fontsize=9)
    ax2.set_xlabel(result.get("time_label", "Time"), fontsize=8)
    ax2.set_ylabel(flux_axis_title(result), fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(alpha=0.22)
    ax2.legend(loc="best", fontsize=6)
    if flux_is_magnitude(result):
        ax2.invert_yaxis()


def draw_tomography_summary_plot(fig: Figure, x: float, y: float, width: float, height: float, advanced: dict) -> None:
    ax = fig.add_axes([x, y, width, height])
    period = np.asarray(advanced.get("period", []), dtype=float)
    centers = np.asarray(advanced.get("time", []), dtype=float)
    values = np.asarray(advanced.get("values", []), dtype=float)
    if values.size:
        values = np.where(np.isfinite(values), values, np.nan)
    finite = values[np.isfinite(values)]
    if len(period) and len(centers) and values.size and finite.size:
        vmin = float(np.nanpercentile(finite, 2))
        vmax = float(np.nanpercentile(finite, 98))
        if vmax <= vmin:
            pad = max(abs(vmin) * 0.05, 1e-12)
            vmin -= pad
            vmax += pad
        mesh = ax.pcolormesh(period, centers, values, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax, label=str(advanced.get("metric", "power")), fraction=0.025, pad=0.015)
        if advanced.get("show_best_track", True) and advanced.get("best_period"):
            ax.plot(advanced["best_period"], centers, color="white", lw=1.0, alpha=0.9)
    else:
        ax.text(
            0.5,
            0.5,
            advanced.get("message") or "No finite tomogram values for the selected settings.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#5f6673",
        )
    ax.set_title("Period tomography", fontsize=9)
    ax.set_xlabel(advanced.get("period_label", "Period"), fontsize=8)
    ax.set_ylabel("Time", fontsize=8)
    ax.tick_params(labelsize=7)


def draw_model_lab_summary_plot(fig: Figure, x: float, y: float, width: float, height: float, model_result: dict, app_result: dict) -> None:
    ax1 = fig.add_axes([x, y, width * 0.48, height])
    phase = np.asarray(model_result.get("phase", []), dtype=float)
    flux = np.asarray(model_result.get("flux", []), dtype=float)
    error = np.asarray(model_result.get("error", []), dtype=float)
    model_phase = np.asarray(model_result.get("model_phase", []), dtype=float)
    model_flux = np.asarray(model_result.get("model_flux", []), dtype=float)
    if len(phase):
        ax1.errorbar(np.r_[phase, phase + 1], np.r_[flux, flux], yerr=np.r_[error, error], fmt="o", ms=2.4, color="#20242a", ecolor="#8b93a5", lw=0.45)
    if len(model_phase):
        ax1.plot(np.r_[model_phase, model_phase + 1], np.r_[model_flux, model_flux], color="#2457a6", lw=1.1)
    ax1.set_xlim(0, 2)
    ax1.set_title("Folded model", fontsize=9)
    ax1.set_xlabel("Phase", fontsize=8)
    ax1.set_ylabel(flux_axis_title(app_result), fontsize=8)
    ax1.tick_params(labelsize=7)
    ax1.grid(alpha=0.22)

    ax2 = fig.add_axes([x + width * 0.53, y, width * 0.47, height])
    time = np.asarray(model_result.get("model_time", []), dtype=float)
    model_time_flux = np.asarray(model_result.get("model_time_flux", []), dtype=float)
    data_time = np.asarray(app_result.get("series", {}).get("time", []), dtype=float)
    data_flux = np.asarray(app_result.get("series", {}).get("flux", []), dtype=float)
    if len(data_time):
        ax2.scatter(data_time, data_flux, s=4, color="#20242a", alpha=0.55)
    if len(time):
        ax2.plot(time, model_time_flux, color="#2457a6", lw=0.8)
    ax2.set_title("Time-domain model", fontsize=9)
    ax2.set_xlabel(app_result.get("time_label", "Time"), fontsize=8)
    ax2.set_ylabel(flux_axis_title(app_result), fontsize=8)
    ax2.tick_params(labelsize=7)
    ax2.grid(alpha=0.22)
    if flux_is_magnitude(app_result):
        ax1.invert_yaxis()
        ax2.invert_yaxis()


def input_summary_frame(result: dict) -> pd.DataFrame:
    filename = str(st.session_state.get("app_filename", "Not provided"))
    raw = st.session_state.get("app_file_bytes")
    fields = st.session_state.get("app_fields", {})
    series = result.get("series", {})
    time = np.asarray(series.get("time", []), dtype=float)
    flux = np.asarray(series.get("flux", []), dtype=float)
    if raw is not None and is_fits_upload(filename, raw):
        file_type = "FITS table"
    elif raw is not None:
        file_type = "ASCII table"
    else:
        file_type = "Unknown"
    object_name = report_metadata().get("object") or fields.get("target_name") or "Not provided"
    rows = [
        ("Object", object_name),
        ("File", filename),
        ("File type", file_type),
        ("Rows used", result.get("n_points", len(time))),
        ("Baseline", f"{float(result.get('baseline', 0.0)):.6g} {result.get('baseline_unit', '')}"),
        ("Time range", "Not available" if len(time) == 0 else f"{float(np.nanmin(time)):.6g} - {float(np.nanmax(time)):.6g}"),
        ("Flux range", "Not available" if len(flux) == 0 else f"{float(np.nanmin(flux)):.6g} - {float(np.nanmax(flux)):.6g}"),
        ("Error column", "yes" if result.get("has_error_column", True) else "no"),
    ]
    target_ra = str(fields.get("target_ra_deg", "")).strip()
    target_dec = str(fields.get("target_dec_deg", "")).strip()
    if target_ra and target_dec:
        rows.append(("Target coordinates", f"RA={target_ra} deg, Dec={target_dec} deg"))
    if file_type == "FITS table":
        rows.extend([
            ("FITS extension", fields.get("fits_extension", "Not recorded")),
            ("Time column", fields.get("fits_time_col", "Not recorded")),
            ("Flux column", fields.get("fits_flux_col", "Not recorded")),
            ("Error column name", fields.get("fits_error_col", "None") or "None"),
        ])
    else:
        rows.extend([
            ("Time column", fields.get("time_col", "Not recorded")),
            ("Flux column", fields.get("flux_col", "Not recorded")),
            ("Error column", fields.get("error_col", "None") or "None"),
        ])
    rows.extend([
        ("Frequency range", f"{fields.get('fmin', 'Not recorded')} - {fields.get('fmax', 'Not recorded')} {result.get('frequency_unit', '')}"),
        ("Min. considered period", f"{fields.get('min_considered_period', 'Not recorded')} {result.get('period_unit', '')}"),
    ])
    return pd.DataFrame(rows, columns=["item", "value"])


def draw_input_overview(fig: Figure, x: float, y: float, width: float, height: float, result: dict) -> None:
    table_width = width * 0.39
    draw_table(
        fig,
        x,
        y,
        table_width,
        height,
        input_summary_frame(result),
        font_size=5.6,
    )
    ax = fig.add_axes([x + table_width + width * 0.055, y, width * 0.555, height])
    series = result.get("series", {})
    time = np.asarray(series.get("time", []), dtype=float)
    flux = np.asarray(series.get("flux", []), dtype=float)
    error = np.asarray(series.get("error", []), dtype=float)
    valid = np.isfinite(time) & np.isfinite(flux)
    if len(error) == len(time):
        valid &= np.isfinite(error)
    time = time[valid]
    flux = flux[valid]
    error = error[valid] if len(error) == len(valid) else np.asarray([], dtype=float)
    if len(time):
        if len(time) > 2500:
            step = int(np.ceil(len(time) / 2500))
            time = time[::step]
            flux = flux[::step]
            error = error[::step] if len(error) else error
        if len(error) and result.get("has_error_column", True):
            ax.errorbar(time, flux, yerr=error, fmt="o", ms=1.9, color="#20242a", ecolor="#9aa2b1", elinewidth=0.35, alpha=0.75)
        else:
            ax.scatter(time, flux, s=5, color="#20242a", alpha=0.72)
    ax.set_title("Original light curve", fontsize=9)
    ax.set_xlabel(result.get("time_label", "Time"), fontsize=8)
    ax.set_ylabel(flux_axis_title(result), fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.22)
    if flux_is_magnitude(result):
        ax.invert_yaxis()


def fmt_value(value, digits: int = 5, default: str = "not available") -> str:
    if value is None:
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return default
    return f"{number:.{digits}g}"


def model_lab_pedagogical_hint(model_result: dict) -> str:
    family = model_result.get("family")
    if family == "fourier":
        return (
            "Multi-harmonic Fourier models show how non-sinusoidal profiles can be decomposed into a "
            "fundamental period plus harmonics. Extra harmonics improve shape flexibility, but they should "
            "be justified by a clear reduction in residuals or information criteria."
        )
    if family == "binary":
        model_kind = model_result.get("model_kind")
        if model_kind == "physical_eclipse_toy":
            return (
                "Masses set the orbital scale through Kepler's law, radii and inclination set whether "
                "eclipses occur, and luminosities, surface brightness, limb darkening, and third light "
                "dilute or deepen the eclipses."
            )
        if model_kind == "eccentric_harmonic":
            return (
                "Harmonic eccentric-binary fits are useful to explore asymmetric orbital modulation, but "
                "inclination, radii, and masses require a physical binary model or external constraints."
            )
        return (
            "Depth and width ratios trace relative brightness/radius geometry, while separation away from "
            "0.5 can suggest eccentric geometry or phase-reference effects."
        )
    if family == "binary_assistant":
        return (
            "This assistant is morphological: it highlights minima, phase separation, depth ratio, and possible "
            "half-period ambiguity, but does not derive a unique mass, radius, or inclination solution."
        )
    if family == "bondi_hoyle":
        return (
            "This toy model tests whether denser or slower wind near periastron can explain the modulation. "
            "Fitted parameters are phenomenological unless masses, wind law, and geometry are externally constrained."
        )
    if family == "pulse":
        return (
            "Epoch folding refines the pulse period, the profile fit estimates pulse shape and pulsed fraction, "
            "and O-C trends can flag Doppler delays before attempting a physical orbital timing solution."
        )
    return "No pedagogical hint is available for this model family."


def model_lab_report_summary(model_result: dict, app_result: dict) -> str:
    family = model_result.get("family")
    summary = model_result.get("summary", {})
    period_unit = app_result.get("period_unit", "")
    time_unit = app_result.get("baseline_unit", period_unit)
    lines: list[str] = []
    if family == "fourier":
        lines.extend([
            f"Model family: Fourier multi-harmonic",
            f"Model period: {fmt_value(model_result.get('period'), 6)} {period_unit}",
            f"Selected harmonics: {model_result.get('selected_harmonics', 'not available')}",
            f"Selection mode: {str(model_result.get('selection', 'manual')).upper()}",
            f"RMS: {fmt_value(summary.get('rms'))}",
            f"Weighted RMS: {fmt_value(summary.get('weighted_rms'))}",
            f"Reduced chi2: {fmt_value(summary.get('chi2_red'))}",
        ])
        for index, term in enumerate(model_result.get("terms", [])[:6], start=1):
            lines.append(
                f"Term {index}: period={fmt_value(term.get('period'), 6)} {period_unit}, "
                f"amplitude={fmt_value(term.get('amplitude'))}, phase of maximum={fmt_value(term.get('phase_of_max'))}"
            )
        lines.append(f"Pedagogical hint: {model_lab_pedagogical_hint(model_result)}")
    elif family == "binary":
        model_kind = str(model_result.get("model_kind", "")).replace("_", " ")
        lines.extend([
            f"Model family: eclipsing / eccentric binary",
            f"Period: {fmt_value(model_result.get('period'), 6)} {period_unit}",
            f"T0 / reference epoch: {fmt_value(model_result.get('t0'), 6)} {time_unit}",
            f"Model: {model_kind}",
        ])
        if model_result.get("model_kind") == "physical_eclipse_toy":
            lines.extend([
                f"Eccentricity: {fmt_value(summary.get('eccentricity'), 4)}",
                f"Omega: {fmt_value(summary.get('omega_deg'))} deg",
                f"Inclination: {fmt_value(summary.get('inclination_deg'))} deg",
                f"Masses / radii: M1={fmt_value(summary.get('mass1_msun'))} Msun, M2={fmt_value(summary.get('mass2_msun'))} Msun, R1={fmt_value(summary.get('radius1_rsun'))} Rsun, R2={fmt_value(summary.get('radius2_rsun'))} Rsun",
                f"Mass ratio q=M2/M1: {fmt_value(summary.get('mass_ratio_q_m2_over_m1'))}",
                f"Semi-major axis: {fmt_value(summary.get('semi_major_axis_rsun'))} Rsun",
                f"Radii: R1/a={fmt_value(summary.get('radius1_over_a'))}, R2/a={fmt_value(summary.get('radius2_over_a'))}",
                f"Temperature ratio T2/T1: {fmt_value(summary.get('temperature_ratio_t2_over_t1'))}",
                f"Luminosity ratio L2/L1: {fmt_value(summary.get('bolometric_luminosity_ratio_l2_over_l1'))}",
                f"Surface brightness ratio S2/S1: {fmt_value(summary.get('surface_brightness_ratio_s2_over_s1'))}",
                f"Third light fraction: {fmt_value(summary.get('third_light_fraction'))}",
                f"Eclipse geometry: {summary.get('eclipse_possible', 'unknown')}",
                f"Pedagogical hint: {model_lab_pedagogical_hint(model_result)}",
            ])
        elif model_result.get("model_kind") == "eccentric_harmonic":
            lines.extend([
                f"Eccentricity: {fmt_value(summary.get('eccentricity'), 4)}",
                "Inclination: not constrained by this empirical fit",
                "Mass ratio: not constrained by this empirical fit",
                f"Pedagogical hint: {model_lab_pedagogical_hint(model_result)}",
            ])
        else:
            primary_phase = model_parameter_value(model_result, "primary_phase")
            secondary_phase = model_parameter_value(model_result, "secondary_phase")
            primary_depth = model_parameter_value(model_result, "primary_depth")
            secondary_depth = model_parameter_value(model_result, "secondary_depth")
            primary_width = model_parameter_value(model_result, "primary_width_phase")
            secondary_width = model_parameter_value(model_result, "secondary_width_phase")
            lines.extend([
                f"Primary eclipse phase: {fmt_value(primary_phase)}",
                f"Secondary eclipse phase: {fmt_value(secondary_phase)}",
            ])
            if primary_phase is not None and secondary_phase is not None:
                separation = (secondary_phase - primary_phase) % 1.0
                lines.append(f"Eclipse separation: {fmt_value(separation)} in phase; offset from 0.5: {fmt_value(separation - 0.5, 4)}")
            lines.extend([
                f"Primary depth/sign: {fmt_value(primary_depth)}",
                f"Secondary depth/sign: {fmt_value(secondary_depth)}",
            ])
            if primary_depth is not None and abs(primary_depth) > 0.0 and secondary_depth is not None:
                lines.append(f"Depth ratio |secondary/primary|: {fmt_value(abs(secondary_depth / primary_depth))}")
            if primary_width is not None and abs(primary_width) > 0.0 and secondary_width is not None:
                lines.append(f"Width ratio secondary/primary: {fmt_value(secondary_width / primary_width)}")
            lines.extend([
                "Eccentricity: not directly constrained by the empirical eclipse model",
                "Inclination / mass ratio: require a physical binary model such as ellc or external constraints",
                f"Pedagogical hint: {model_lab_pedagogical_hint(model_result)}",
            ])
        if summary.get("BIC") is not None:
            lines.append(f"BIC: {fmt_value(summary.get('BIC'))}")
        if summary.get("rms") is not None:
            lines.append(f"RMS: {fmt_value(summary.get('rms'))}")
    elif family == "binary_assistant":
        lines.extend([
            "Model family: binary interpretation assistant",
            f"Folded period: {fmt_value(model_result.get('period'), 6)} {period_unit}",
            f"T0 / reference epoch: {fmt_value(model_result.get('t0'), 6)} {time_unit}",
            f"Morphology: {summary.get('morphology', 'not available')}",
            f"Minimum separation: {fmt_value(summary.get('minimum_phase_separation'))}",
            f"Separation from 0.5: {fmt_value(summary.get('separation_from_0p5'))}",
            f"Depth ratio: {fmt_value(summary.get('depth_ratio'))}",
            f"Possible half-period ambiguity: {summary.get('half_period_warning', False)}",
            f"Pedagogical hint: {model_lab_pedagogical_hint(model_result)}",
        ])
    elif family == "bondi_hoyle":
        maxima = model_result.get("extrema", [])
        lines.extend([
            "Model family: Bondi-Hoyle accretion toy model",
            f"Orbital period: {fmt_value(model_result.get('period'), 6)} {period_unit}",
            f"T0 / periastron epoch: {fmt_value(model_result.get('t0'), 6)} {time_unit}",
            f"Eccentricity: {fmt_value(summary.get('eccentricity'), 4)}",
            f"Effective v_wind / v_orb: {fmt_value(summary.get('wind_speed_ratio'), 4)}",
            f"Wind beta: {fmt_value(summary.get('wind_beta'), 4)}",
            f"Wind launch radius R_donor/a: {fmt_value(summary.get('donor_radius_over_a'))}",
            f"Phase lag: {fmt_value(summary.get('phase_lag'), 5)}",
        ])
        if summary.get("donor_spectral_type"):
            lines.append(f"Donor preset: {summary.get('donor_spectral_type')} {summary.get('donor_luminosity_class', '')}")
        if summary.get("wind_input_mode") == "v_inf":
            lines.extend([
                f"v_inf: {fmt_value(summary.get('v_inf_km_s'))} km/s",
                f"Masses: M_donor={fmt_value(summary.get('donor_mass_msun'))} Msun, M_compact={fmt_value(summary.get('compact_mass_msun'))} Msun",
                f"Donor radius: {fmt_value(summary.get('donor_radius_rsun'))} Rsun",
                f"Semi-major axis: {fmt_value(summary.get('semi_major_axis_rsun'))} Rsun",
                f"Orbital speed scale: {fmt_value(summary.get('orbital_speed_km_s'))} km/s",
            ])
        if maxima:
            lines.append(f"Model accretion maximum phase: {fmt_value(maxima[0].get('phase'))}")
        lines.extend([
            f"Accretion scale/sign: {fmt_value(summary.get('scale'))}",
            f"BIC: {fmt_value(summary.get('BIC'))}",
            f"RMS: {fmt_value(summary.get('rms'))}",
            f"Pedagogical hint: {model_lab_pedagogical_hint(model_result)}",
        ])
    elif family == "pulse":
        maxima = model_result.get("maxima", [])
        epoch_stat = summary.get("epoch_folding_statistic")
        lines.extend([
            "Model family: pulse-period analysis",
            f"Best pulse period: {fmt_value(model_result.get('period'), 8)} {period_unit}",
            f"Input period guess: {fmt_value(model_result.get('period_guess'), 8)} {period_unit}",
            f"T0: {fmt_value(model_result.get('t0'), 6)} {time_unit}",
            f"Epoch-folding statistic: {fmt_value(epoch_stat)}",
            f"Pulse arrivals used: {int(summary.get('n_arrivals', 0))}",
            f"Peak-to-peak pulsed fraction: {fmt_value(summary.get('peak_to_peak_pulsed_fraction'))}",
            f"RMS pulsed fraction: {fmt_value(summary.get('rms_pulsed_fraction'))}",
        ])
        if maxima:
            lines.append(f"Main pulse maximum phase: {fmt_value(maxima[0].get('phase'))}")
        oc_summary = model_result.get("oc_summary", {})
        if oc_summary:
            lines.append(f"O-C sinusoid period: {fmt_value(oc_summary.get('orbital_period'), 6)} {period_unit}")
            lines.append(f"Projected a sin i proxy: {fmt_value(oc_summary.get('oc_amplitude_time'))} {time_unit}")
        lines.append(f"Pedagogical hint: {model_lab_pedagogical_hint(model_result)}")
    else:
        lines.append("Model summary is not available for this model family.")
    return "\n".join(lines)


def append_pdf_report(previous_pdf: bytes, current_report_pdf: bytes) -> bytes:
    if PdfReader is None or PdfWriter is None:
        raise RuntimeError("pypdf is required to append a previous PDF report.")
    writer = PdfWriter()
    for pdf_bytes in (previous_pdf, current_report_pdf):
        reader = PdfReader(io.BytesIO(pdf_bytes))
        for page in reader.pages:
            writer.add_page(page)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def build_periodicity_pdf_report(result: dict | None) -> bytes:
    metadata = report_metadata()
    sections = report_selected_sections()
    advanced = st.session_state.get("app_advanced_result")
    model_result = st.session_state.get("app_model_lab_result")
    buffer = io.BytesIO()
    with PdfPages(buffer) as pdf:
        report = CompactPdfReport(pdf)
        title_lines = "\n".join([
            f"Object: {metadata['object'] or 'Not provided'}",
            f"Student: {metadata['student'] or 'Not provided'}",
            f"Course: {metadata['course'] or 'Not provided'}",
            f"Input file: {st.session_state.get('app_filename', 'Not provided')}",
            f"Application: Periodicity Workbench {APP_VERSION}",
        ])
        report.text_block("Periodicity Workbench Report", title_lines, min_height=0.20)
        if result:
            report.block(
                "Input data overview",
                0.31,
                lambda fig, x, y, w, h: draw_input_overview(fig, x, y, w, h, result),
            )
        if result and sections["frequency"]:
            report.block(
                "Frequency search plots",
                0.31,
                lambda fig, x, y, w, h: draw_frequency_summary_plot(fig, x, y, w, h, result),
            )
            report.text_block(
                "Folded fit equation",
                fit_equation_text(result.get("fold_fit_terms", []), variable="phase") or "Not available.",
                min_height=0.17,
                size=8.0,
                wrap_width=78,
                line_height=0.036,
                line_gap=0.15,
            )
            folded_terms = result.get("fold_fit_terms", [])
            summary_df = global_fit_summary(folded_terms)
            folded_df = folded_periods_and_parameters(result)
            report.block(
                "Global fit summary and folded parameters",
                0.19,
                lambda fig, x, y, w, h: (
                    draw_table(fig, x, y, w * 0.36, h, pdf_table_frame(summary_df, max_rows=3), font_size=6.0),
                    draw_table(
                        fig,
                        x + w * 0.40,
                        y,
                        w * 0.60,
                        h,
                        pdf_table_frame(
                            folded_df,
                            columns=["role", "period", "frequency_ratio_in_phase", "amplitude", "amplitude_error", "phase_of_max", "rms", "chi2_red"],
                            max_rows=6,
                        ),
                        font_size=5.2,
                    ),
                ),
            )
            report.table_block(
                "Detected peaks",
                peaks_dataframe(result.get("peaks", [])),
                columns=["period", "period_error", "frequency", "frequency_error", "power", "FAP", "type", "window_period"],
                max_rows=8,
            )
            period_lines = evidence_period_rows("l1", "frequency")
            evidence_text = (
                f"Periods reported by the student: {period_lines or 'Not provided.'}\n"
                f"{st.session_state.get('evidence_frequency_comments', 'No comments provided.')}"
            )
            report.text_block("Frequency-search evidence", evidence_text, min_height=0.14)
        if result and sections["prewhitening"] and result.get("has_prewhitening"):
            report.block(
                "Iterative prewhitening plots",
                0.30,
                lambda fig, x, y, w, h: draw_prewhitening_summary_plot(fig, x, y, w, h, result),
            )
            report.text_block(
                "Prewhitening fit equation",
                fit_equation_text(result.get("prewhitening_terms", []), variable="t_shifted") or "Not available.",
                min_height=0.17,
                size=8.0,
                wrap_width=78,
                line_height=0.036,
                line_gap=0.15,
            )
            report.table_block(
                "Prewhitening steps",
                clean_dataframe(pd.DataFrame(result.get("prewhitening_terms", []))),
                columns=["label", "main_period", "period_error", "frequency", "frequency_error", "amplitude", "amplitude_error", "rms"],
                max_rows=7,
            )
            report.table_block(
                "Remaining LS peaks after prewhitening",
                peaks_dataframe(result.get("residual_peaks", [])),
                columns=["period", "period_error", "frequency", "frequency_error", "FAP", "type"],
                max_rows=7,
            )
            evidence_text = (
                f"Periods used: {st.session_state.get('evidence_prewhitening_periods', 'Not provided.')}\n"
                f"{st.session_state.get('evidence_prewhitening_comments', 'No comments provided.')}"
            )
            report.text_block("Prewhitening evidence", evidence_text, min_height=0.14)
        if sections["tomography"] and advanced:
            report.block(
                "Period tomography",
                0.32,
                lambda fig, x, y, w, h: draw_tomography_summary_plot(fig, x, y, w, h, advanced),
            )
            tomography_text = "\n".join([
                f"Method: {advanced.get('method_label', advanced.get('method', 'Not available'))}",
                f"Window width: {st.session_state.get('evidence_tomography_window_width', advanced.get('window_width', 'Not provided'))}",
                f"Window step: {st.session_state.get('evidence_tomography_window_step', advanced.get('window_step', 'Not provided'))}",
                f"Comments: {st.session_state.get('evidence_tomography_comments', 'No comments provided.')}",
            ])
            report.text_block("Tomography evidence", tomography_text, min_height=0.16)
        if result and sections["model_lab"] and model_result:
            report.block(
                "Model laboratory plots",
                0.30,
                lambda fig, x, y, w, h: draw_model_lab_summary_plot(fig, x, y, w, h, model_result, result),
            )
            formula = str(model_result.get("formula", fit_equation_text(model_result.get("terms", []), variable="phase") or "Not available."))
            report.text_block(
                "Model formula",
                f"Model family: {model_result.get('family', 'Not available')}\n{formula}",
                min_height=0.19,
                size=8.0,
                wrap_width=78,
                line_height=0.036,
                line_gap=0.15,
            )
            report.text_block(
                "Model laboratory summary",
                model_lab_report_summary(model_result, result),
                min_height=0.36,
                size=8.0,
                wrap_width=88,
                line_height=0.027,
                line_gap=0.055,
            )
            report.text_block(
                "Model-laboratory evidence",
                str(st.session_state.get("evidence_model_lab_comments", "No comments provided.")),
                min_height=0.13,
            )
        report.finish()
    return buffer.getvalue()


def prewhitening_model_plot(result: dict, show_errors: bool = True) -> go.Figure:
    series = result["series"]
    show_errors = show_errors and bool(result.get("has_error_column", True))
    time = np.asarray(series["time"], dtype=float)
    flux = np.asarray(series["flux"], dtype=float)
    error = np.asarray(series["error"], dtype=float)
    model_key = "prewhitening_display_model_flux" if "prewhitening_display_model_flux" in series else "prewhitening_model_flux"
    model_at_data = np.asarray(series[model_key], dtype=float)
    residual = flux - model_at_data
    terms = result.get("prewhitening_display_terms") or result.get("prewhitening_terms", [])
    display_model = bool(result.get("prewhitening_display_terms"))
    frequencies = [float(term["frequency"]) for term in terms]
    dense_time = np.linspace(float(np.nanmin(time)), float(np.nanmax(time)), 2500)
    dense_model = None
    if frequencies and terms and "cos_coeff" in terms[0]:
        offset = float(terms[0].get("offset", 0.0) or 0.0)
        dense_shifted = dense_time - float(np.nanmin(time))
        dense_model = np.full_like(dense_time, offset, dtype=float)
        for term, freq in zip(terms, frequencies):
            phase = 2.0 * np.pi * freq * dense_shifted
            dense_model += float(term.get("cos_coeff", 0.0) or 0.0) * np.cos(phase)
            dense_model += float(term.get("sin_coeff", 0.0) or 0.0) * np.sin(phase)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.68, 0.32],
        subplot_titles=("Original data and prewhitening model", "Residuals"),
    )
    fig.add_trace(go.Scatter(
        x=time,
        y=flux,
        error_y=dict(type="data", array=error, visible=show_errors),
        mode="markers",
        line=dict(width=0),
        marker=dict(color="#20242a", size=5),
        name="Original data",
        hovertemplate="Time=%{x:.5f}<br>Value=%{y:.6g}<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=time,
        y=model_at_data,
        mode="markers",
        marker=dict(color="#2457a6", size=5, symbol="diamond"),
        name="Display model at data" if display_model else "Prewhitening model at data",
        hovertemplate="Time=%{x:.5f}<br>Model=%{y:.6g}<extra></extra>",
    ), row=1, col=1)
    if dense_model is not None:
        fig.add_trace(go.Scatter(
            x=dense_time,
            y=dense_model,
            mode="lines",
            line=dict(color="#2457a6", width=2),
            name="Full display model" if display_model else "Full model",
            hovertemplate="Time=%{x:.5f}<br>Full model=%{y:.6g}<extra></extra>",
        ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=time,
        y=residual,
        error_y=dict(type="data", array=error, visible=show_errors),
        mode="markers",
        marker=dict(color="#b13b32", size=5),
        name="Residuals",
        hovertemplate="Time=%{x:.5f}<br>Residual=%{y:.6g}<extra></extra>",
    ), row=2, col=1)
    fig.add_hline(y=0.0, line_dash="dash", line_color="#777777", opacity=0.7, row=2, col=1)
    fig.update_layout(
        title="Original light curve with prewhitening model",
        height=620,
        margin=dict(l=20, r=20, t=50, b=80),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.85)",
        ),
    )
    fig.update_yaxes(title_text=flux_axis_title(result), row=1, col=1)
    apply_flux_axis(fig, result, row=1, col=1)
    fig.update_yaxes(title_text="Residuals", row=2, col=1)
    fig.update_xaxes(title_text=result.get("time_label", "Time [d]"), row=2, col=1)
    return frame(fig)


def advanced_plot(advanced: dict) -> go.Figure:
    metric = advanced.get("metric", "power")
    period = np.asarray(advanced.get("period", []), dtype=float)
    time = np.asarray(advanced.get("time", []), dtype=float)
    values = np.asarray(advanced.get("values", []), dtype=float)
    if values.ndim == 1 and len(time) == 1 and len(period) == values.size:
        values = values.reshape(1, -1)
    if values.ndim == 2 and values.shape == (len(period), len(time)) and values.shape != (len(time), len(period)):
        values = values.T
    if values.size:
        values = np.where(np.isfinite(values), values, np.nan)
    finite = values[np.isfinite(values)] if values.size else np.asarray([])
    if not len(period) or not len(time) or values.shape != (len(time), len(period)) or finite.size == 0:
        fig = go.Figure()
        message = advanced.get("message") or "No finite tomogram values for the selected settings."
        if values.size and values.shape != (len(time), len(period)):
            message = f"Tomogram shape mismatch: values={values.shape}, time={len(time)}, period={len(period)}."
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color="#5f6673"),
        )
    else:
        zmin = float(np.nanpercentile(finite, 2))
        zmax = float(np.nanpercentile(finite, 98))
        if not np.isfinite(zmin) or not np.isfinite(zmax):
            zmin, zmax = float(np.nanmin(finite)), float(np.nanmax(finite))
        if zmax <= zmin:
            pad = max(abs(zmin) * 0.05, 1e-12)
            zmin -= pad
            zmax += pad
        fig = go.Figure(go.Heatmap(
            x=period,
            y=time,
            z=values,
            zmin=zmin,
            zmax=zmax,
            colorscale="Viridis",
            colorbar=dict(title=metric),
            hovertemplate="Period=%{x:.6g}<br>Time=%{y:.6g}<br>" + str(metric) + "=%{z:.6g}<extra></extra>",
        ))
    if advanced.get("track_min_period") is not None and advanced.get("track_max_period") is not None:
        fig.add_vrect(
            x0=advanced["track_min_period"],
            x1=advanced["track_max_period"],
            fillcolor="white",
            opacity=0.14,
            line_width=0,
        )
    if advanced.get("show_best_track", True) and advanced.get("best_period"):
        fig.add_trace(go.Scatter(x=advanced["best_period"], y=advanced["time"], mode="lines+markers", line=dict(color="white", width=2), marker=dict(color="white", size=4), showlegend=False))
    fig.update_layout(title=advanced.get("method_label", "Advanced map"), xaxis_title=advanced.get("period_label", "Period [d]"), yaxis_title="Window center time", height=520, margin=dict(l=20, r=20, t=50, b=20), showlegend=False)
    return frame(fig)


def model_lab_fourier_plot(model_result: dict, app_result: dict) -> go.Figure:
    phase = np.asarray(model_result.get("phase", []), dtype=float)
    flux = np.asarray(model_result.get("flux", []), dtype=float)
    error = np.asarray(model_result.get("error", []), dtype=float)
    phase_2 = np.concatenate([phase, phase + 1.0]) if len(phase) else phase
    flux_2 = np.concatenate([flux, flux]) if len(flux) else flux
    error_2 = np.concatenate([error, error]) if len(error) else error
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=phase_2,
        y=flux_2,
        error_y=dict(type="data", array=error_2, visible=bool(app_result.get("has_error_column", True))),
        mode="markers",
        marker=dict(color="#20242a", size=7),
        name="Binned folded data",
        hovertemplate="Phase=%{x:.5f}<br>Value=%{y:.6g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=model_result.get("model_phase", []),
        y=model_result.get("model_flux", []),
        mode="lines",
        line=dict(color="#2457a6", width=2),
        name="Fourier model",
        hovertemplate="Phase=%{x:.5f}<br>Model=%{y:.6g}<extra></extra>",
    ))
    for maximum in model_result.get("maxima", []):
        for offset in [0.0, 1.0]:
            fig.add_vline(x=float(maximum["phase"]) + offset, line_dash="dash", line_color="#2457a6", opacity=0.65)
    fig.update_layout(
        title="Fourier multi-harmonic model",
        xaxis_title="Phase",
        yaxis_title=f"Weighted mean {flux_axis_title(app_result).lower()}",
        height=480,
        margin=dict(l=20, r=20, t=50, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(range=[0, 2])
    apply_flux_axis(fig, app_result)
    return frame(fig)


def model_lab_binary_plot(model_result: dict, app_result: dict) -> go.Figure:
    phase = np.asarray(model_result.get("phase", []), dtype=float)
    flux = np.asarray(model_result.get("flux", []), dtype=float)
    error = np.asarray(model_result.get("error", []), dtype=float)
    phase_2 = np.concatenate([phase, phase + 1.0]) if len(phase) else phase
    flux_2 = np.concatenate([flux, flux]) if len(flux) else flux
    error_2 = np.concatenate([error, error]) if len(error) else error
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=phase_2,
        y=flux_2,
        error_y=dict(type="data", array=error_2, visible=bool(app_result.get("has_error_column", True))),
        mode="markers",
        marker=dict(color="#20242a", size=7),
        name="Binned folded data",
        hovertemplate="Phase=%{x:.5f}<br>Value=%{y:.6g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=model_result.get("model_phase", []),
        y=model_result.get("model_flux", []),
        mode="lines",
        line=dict(color="#2457a6", width=2),
        name="Binary model",
        hovertemplate="Phase=%{x:.5f}<br>Model=%{y:.6g}<extra></extra>",
    ))
    for item in model_result.get("extrema", []):
        color = "#b13b32" if "eclipse" in str(item.get("kind", "")) else "#2457a6"
        for offset in [0.0, 1.0]:
            fig.add_vline(x=float(item["phase"]) + offset, line_dash="dash", line_color=color, opacity=0.65)
    fig.update_layout(
        title="Eclipsing / eccentric binary model",
        xaxis_title="Orbital phase",
        yaxis_title=f"Weighted mean {flux_axis_title(app_result).lower()}",
        height=480,
        margin=dict(l=20, r=20, t=50, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(range=[0, 2])
    apply_flux_axis(fig, app_result)
    return frame(fig)


def circular_phase_distance(a: float, b: float) -> float:
    delta = abs((a - b) % 1.0)
    return min(delta, 1.0 - delta)


def binary_interpretation_assistant(result: dict) -> dict:
    phase = np.asarray(result.get("series", {}).get("fold_phase", []), dtype=float)
    flux = np.asarray(result.get("series", {}).get("fold_flux", []), dtype=float)
    error = np.asarray(result.get("series", {}).get("fold_error", []), dtype=float)
    valid = np.isfinite(phase) & np.isfinite(flux)
    phase = phase[valid]
    flux = flux[valid]
    error = error[valid] if len(error) == len(valid) else np.asarray([], dtype=float)
    if len(phase) < 4:
        raise ValueError("The binary assistant needs at least four folded bins.")
    order = np.argsort(phase)
    phase = phase[order]
    flux = flux[order]
    error = error[order] if len(error) == len(order) else error
    brightness = -flux if flux_is_magnitude(result) else flux
    baseline = float(np.nanpercentile(brightness, 90))
    primary_idx = int(np.nanargmin(brightness))
    separated = np.asarray([circular_phase_distance(float(ph), float(phase[primary_idx])) >= 0.20 for ph in phase], dtype=bool)
    if np.any(separated):
        separated_indices = np.flatnonzero(separated)
        secondary_idx = int(separated_indices[int(np.nanargmin(brightness[separated]))])
    else:
        secondary_idx = int(np.nanargmin(np.where(np.arange(len(phase)) == primary_idx, np.inf, brightness)))
    primary_phase = float(phase[primary_idx])
    secondary_phase = float(phase[secondary_idx])
    separation = circular_phase_distance(secondary_phase, primary_phase)
    primary_depth = max(0.0, baseline - float(brightness[primary_idx]))
    secondary_depth = max(0.0, baseline - float(brightness[secondary_idx]))
    depth_ratio = None if primary_depth <= 0.0 else float(secondary_depth / primary_depth)
    asymmetry = abs(separation - 0.5)
    harmonic_rows = result.get("harmonic_diagnostics", [])
    half_period_warning = any(row.get("relation") == "2P" and row.get("detected") for row in harmonic_rows)
    if depth_ratio is not None and depth_ratio > 0.75 and asymmetry < 0.08:
        morphology = "W UMa / contact-like candidate"
    elif depth_ratio is not None and depth_ratio < 0.45:
        morphology = "Algol-like detached candidate"
    else:
        morphology = "Beta Lyrae / ellipsoidal or mixed candidate"
    if asymmetry >= 0.05:
        eccentricity_note = "secondary minimum is not exactly 0.5 in phase from the primary; eccentricity, apsidal orientation, or an imperfect T0 may matter"
    else:
        eccentricity_note = "minimum separation is close to 0.5; a circular or well-phased solution is plausible"
    rows = [
        {"quantity": "primary minimum phase", "value": primary_phase, "interpretation": "deeper/lowest-brightness minimum"},
        {"quantity": "secondary minimum phase", "value": secondary_phase, "interpretation": "next separated minimum"},
        {"quantity": "minimum separation", "value": separation, "interpretation": eccentricity_note},
        {"quantity": "primary depth proxy", "value": primary_depth, "interpretation": "relative to upper folded-profile envelope"},
        {"quantity": "secondary depth proxy", "value": secondary_depth, "interpretation": "relative to upper folded-profile envelope"},
        {"quantity": "secondary / primary depth", "value": depth_ratio, "interpretation": "similar depths favor contact-like systems; very different depths favor Algol-like systems"},
        {"quantity": "possible half-period ambiguity", "value": bool(half_period_warning), "interpretation": "try folding at 2P if LS selected nearly equal minima as one cycle"},
    ]
    summary = {
        "morphology": morphology,
        "minimum_phase_separation": separation,
        "separation_from_0p5": asymmetry,
        "depth_ratio": depth_ratio,
        "half_period_warning": bool(half_period_warning),
        "rms": None,
    }
    hints = [
        "Fold at 2P if the LS peak may represent two similar eclipses per orbit.",
        "If the secondary minimum is not separated by 0.5 in phase, inspect eccentricity or the chosen T0.",
        "Depth ratios alone are not unique mass or radius estimates; inclination, third light, temperatures, and radii are degenerate.",
    ]
    phase_2 = np.concatenate([phase, phase + 1.0])
    flux_2 = np.concatenate([flux, flux])
    return {
        "family": "binary_assistant",
        "period": result.get("folded_period") or result.get("primary_period"),
        "t0": result.get("t0"),
        "phase": phase.tolist(),
        "flux": flux.tolist(),
        "error": error.tolist(),
        "model_phase": phase_2.tolist(),
        "model_flux": flux_2.tolist(),
        "summary": summary,
        "parameters": rows,
        "hints": hints,
        "formula": "Morphological diagnostics from folded minima: phase separation, depth ratio, and harmonic/2P ambiguity checks.",
    }


def model_lab_binary_assistant_plot(model_result: dict, app_result: dict) -> go.Figure:
    phase = np.asarray(model_result.get("phase", []), dtype=float)
    flux = np.asarray(model_result.get("flux", []), dtype=float)
    error = np.asarray(model_result.get("error", []), dtype=float)
    phase_2 = np.concatenate([phase, phase + 1.0]) if len(phase) else phase
    flux_2 = np.concatenate([flux, flux]) if len(flux) else flux
    error_2 = np.concatenate([error, error]) if len(error) else error
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=phase_2,
        y=flux_2,
        error_y=dict(type="data", array=error_2, visible=bool(app_result.get("has_error_column", True))),
        mode="markers",
        marker=dict(color="#20242a", size=8),
        name="Folded data",
        hovertemplate="Phase=%{x:.5f}<br>Value=%{y:.6g}<extra></extra>",
    ))
    for row in model_result.get("parameters", []):
        if row.get("quantity") not in {"primary minimum phase", "secondary minimum phase"}:
            continue
        color = "#b13b32" if row["quantity"].startswith("primary") else "#2457a6"
        for offset in [0.0, 1.0]:
            fig.add_vline(x=float(row["value"]) + offset, line_dash="dash", line_color=color, opacity=0.72)
    fig.update_layout(
        title="Binary interpretation assistant",
        xaxis_title="Orbital phase",
        yaxis_title=f"Weighted mean {flux_axis_title(app_result).lower()}",
        height=440,
        margin=dict(l=20, r=20, t=50, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(range=[0, 2])
    apply_flux_axis(fig, app_result)
    return frame(fig)


def model_lab_bondi_hoyle_plot(model_result: dict, app_result: dict) -> go.Figure:
    phase = np.asarray(model_result.get("phase", []), dtype=float)
    flux = np.asarray(model_result.get("flux", []), dtype=float)
    error = np.asarray(model_result.get("error", []), dtype=float)
    phase_2 = np.concatenate([phase, phase + 1.0]) if len(phase) else phase
    flux_2 = np.concatenate([flux, flux]) if len(flux) else flux
    error_2 = np.concatenate([error, error]) if len(error) else error
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=phase_2,
        y=flux_2,
        error_y=dict(type="data", array=error_2, visible=bool(app_result.get("has_error_column", True))),
        mode="markers",
        marker=dict(color="#20242a", size=7),
        name="Binned folded data",
        hovertemplate="Phase=%{x:.5f}<br>Value=%{y:.6g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=model_result.get("model_phase", []),
        y=model_result.get("model_flux", []),
        mode="lines",
        line=dict(color="#2457a6", width=2),
        name="Bondi-Hoyle toy model",
        hovertemplate="Phase=%{x:.5f}<br>Model=%{y:.6g}<extra></extra>",
    ))
    for item in model_result.get("extrema", []):
        for offset in [0.0, 1.0]:
            fig.add_vline(x=float(item["phase"]) + offset, line_dash="dash", line_color="#2457a6", opacity=0.65)
    fig.update_layout(
        title="Bondi-Hoyle wind accretion toy model",
        xaxis_title="Orbital phase",
        yaxis_title=f"Weighted mean {flux_axis_title(app_result).lower()}",
        height=480,
        margin=dict(l=20, r=20, t=50, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(range=[0, 2])
    apply_flux_axis(fig, app_result)
    return frame(fig)


def model_lab_pulse_profile_plot(model_result: dict, app_result: dict) -> go.Figure:
    phase = np.asarray(model_result.get("phase", []), dtype=float)
    flux = np.asarray(model_result.get("flux", []), dtype=float)
    error = np.asarray(model_result.get("error", []), dtype=float)
    phase_2 = np.concatenate([phase, phase + 1.0]) if len(phase) else phase
    flux_2 = np.concatenate([flux, flux]) if len(flux) else flux
    error_2 = np.concatenate([error, error]) if len(error) else error
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=phase_2,
        y=flux_2,
        error_y=dict(type="data", array=error_2, visible=bool(app_result.get("has_error_column", True))),
        mode="markers",
        marker=dict(color="#20242a", size=7),
        name="Binned pulse profile",
        hovertemplate="Pulse phase=%{x:.5f}<br>Value=%{y:.6g}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=model_result.get("model_phase", []),
        y=model_result.get("model_flux", []),
        mode="lines",
        line=dict(color="#2457a6", width=2),
        name="Pulse-shape model",
        hovertemplate="Pulse phase=%{x:.5f}<br>Model=%{y:.6g}<extra></extra>",
    ))
    for maximum in model_result.get("maxima", []):
        for offset in [0.0, 1.0]:
            fig.add_vline(x=float(maximum["phase"]) + offset, line_dash="dash", line_color="#2457a6", opacity=0.65)
    fig.update_layout(
        title="Pulse profile model",
        xaxis_title="Pulse phase",
        yaxis_title=f"Weighted mean {flux_axis_title(app_result).lower()}",
        height=480,
        margin=dict(l=20, r=20, t=50, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_xaxes(range=[0, 2])
    apply_flux_axis(fig, app_result)
    return frame(fig)


def model_lab_pulse_epoch_plot(model_result: dict, app_result: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=model_result.get("trial_period", []),
        y=model_result.get("epoch_power", []),
        mode="lines",
        line=dict(color="#2457a6", width=2),
        name="Epoch-folding statistic",
        hovertemplate=f"Period=%{{x:.8g}} {app_result.get('period_unit', '')}<br>Statistic=%{{y:.6g}}<extra></extra>",
    ))
    if model_result.get("period") is not None:
        fig.add_vline(x=float(model_result["period"]), line_dash="dash", line_color="#b13b32", opacity=0.8)
    fig.update_layout(
        title="Epoch-folding period refinement",
        xaxis_title=f"Trial period [{app_result.get('period_unit', '')}]",
        yaxis_title="Epoch-folding statistic",
        height=380,
        margin=dict(l=20, r=20, t=50, b=25),
        showlegend=False,
    )
    return frame(fig)


def model_lab_pulse_oc_plot(model_result: dict, app_result: dict) -> go.Figure:
    arrivals = model_result.get("arrivals", [])
    time = [row.get("time") for row in arrivals]
    oc = [row.get("oc_time") for row in arrivals]
    err = [row.get("arrival_time_error") for row in arrivals]
    show_errors = bool(app_result.get("has_error_column", True)) and any(value is not None for value in err)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=oc,
        error_y=dict(type="data", array=err, visible=show_errors),
        mode="markers",
        marker=dict(color="#20242a", size=7),
        name="Pulse arrivals O-C",
        hovertemplate="Time=%{x:.6g}<br>O-C=%{y:.6g}<extra></extra>",
    ))
    if model_result.get("oc_model_time"):
        fig.add_trace(go.Scatter(
            x=model_result.get("oc_model_time", []),
            y=model_result.get("oc_model", []),
            mode="lines",
            line=dict(color="#2457a6", width=2),
            name="O-C sinusoid",
            hovertemplate="Time=%{x:.6g}<br>Model O-C=%{y:.6g}<extra></extra>",
        ))
    fig.add_hline(y=0.0, line_dash="dash", line_color="#777777", opacity=0.6)
    fig.update_layout(
        title="Pulse arrival times",
        xaxis_title=app_result.get("time_label", "Time"),
        yaxis_title=f"O-C [{app_result.get('baseline_unit', '')}]",
        height=380,
        margin=dict(l=20, r=20, t=50, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    return frame(fig)


def model_lab_time_plot(model_result: dict, app_result: dict, show_errors: bool = True) -> go.Figure:
    time = np.asarray(model_result.get("data_time", []), dtype=float)
    flux = np.asarray(model_result.get("data_flux", []), dtype=float)
    error = np.asarray(model_result.get("data_error", []), dtype=float)
    model_at_data = np.asarray(model_result.get("model_at_data", []), dtype=float)
    show_errors = show_errors and bool(app_result.get("has_error_column", True))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time,
        y=flux,
        error_y=dict(type="data", array=error, visible=show_errors),
        mode="markers",
        marker=dict(color="#20242a", size=5),
        name="Original data",
        hovertemplate="Time=%{x:.6g}<br>Value=%{y:.6g}<extra></extra>",
    ))
    if len(model_at_data):
        fig.add_trace(go.Scatter(
            x=time,
            y=model_at_data,
            mode="markers",
            marker=dict(color="#2457a6", size=5, symbol="diamond"),
            name="Model at data",
            hovertemplate="Time=%{x:.6g}<br>Model=%{y:.6g}<extra></extra>",
        ))
    fig.add_trace(go.Scatter(
        x=model_result.get("model_time", []),
        y=model_result.get("model_time_flux", []),
        mode="lines",
        line=dict(color="#2457a6", width=2),
        name="Full model",
        hovertemplate="Time=%{x:.6g}<br>Model=%{y:.6g}<extra></extra>",
    ))
    fig.update_layout(
        title="Full data set with model",
        xaxis_title=app_result.get("time_label", "Time"),
        yaxis_title=flux_axis_title(app_result),
        height=480,
        margin=dict(l=20, r=20, t=50, b=25),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    apply_flux_axis(fig, app_result)
    return frame(fig)


def fields_from_state(prefix: str, bootstrap_override: int | None = None) -> dict[str, str]:
    periods = st.session_state.get(f"{prefix}_selected_periods", [])
    excluded = st.session_state.get(f"{prefix}_excluded_periods", [])
    fields = {
        "time_col": str(st.session_state.get(f"{prefix}_time_col", 1)),
        "flux_col": str(st.session_state.get(f"{prefix}_flux_col", 2)),
        "error_col": str(st.session_state.get(f"{prefix}_error_col", 3)) if st.session_state.get(f"{prefix}_use_error", True) else "",
        "flux_is_magnitude": "true" if st.session_state.get(f"{prefix}_flux_is_magnitude", False) else "false",
        "time_unit": st.session_state.get(f"{prefix}_time_unit", "days"),
        "target_name": str(st.session_state.get(f"{prefix}_target_name", "")).strip(),
        "target_ra_deg": str(st.session_state.get(f"{prefix}_target_ra_deg", "")).strip(),
        "target_dec_deg": str(st.session_state.get(f"{prefix}_target_dec_deg", "")).strip(),
        "xmin": str(st.session_state.get(f"{prefix}_xmin", "")).strip(),
        "xmax": str(st.session_state.get(f"{prefix}_xmax", "")).strip(),
        "ymin": str(st.session_state.get(f"{prefix}_ymin", "")).strip(),
        "ymax": str(st.session_state.get(f"{prefix}_ymax", "")).strip(),
        "fmin": str(st.session_state.get(f"{prefix}_fmin", 0.01)),
        "fmax": str(st.session_state.get(f"{prefix}_fmax", 1.0)),
        "samples_per_peak": str(st.session_state.get(f"{prefix}_samples_per_peak", 10.0)),
        "max_peaks": str(st.session_state.get(f"{prefix}_max_peaks", 6)),
        "min_considered_period": str(st.session_state.get(f"{prefix}_min_period", 2.0)),
        "window_artifact_power": str(st.session_state.get(f"{prefix}_window_threshold", 0.01)),
        "window_artifact_tolerance": str(st.session_state.get(f"{prefix}_window_tolerance", 0.01)),
        "n_bootstrap": str(bootstrap_override if bootstrap_override is not None else st.session_state.get(f"{prefix}_bootstrap", 1000)),
        "bootstrap_width": str(st.session_state.get(f"{prefix}_bootstrap_width", 0.03)),
        "fold_bins": str(st.session_state.get(f"{prefix}_fold_bins", 10)),
        "fold_fit_mode": st.session_state.get(f"{prefix}_fold_mode", "harmonics"),
        "fold_fit_harmonics": str(st.session_state.get(f"{prefix}_fold_harmonics", 1)),
        "model_fit_method": st.session_state.get(f"{prefix}_model_fit_method", "standard"),
        "exclusion_tolerance": str(st.session_state.get(f"{prefix}_exclusion_tolerance", 0.015)),
        "advanced_method": st.session_state.get(f"{prefix}_advanced_method", "v1"),
        "advanced_fmin": str(st.session_state.get(f"{prefix}_advanced_fmin", st.session_state.get(f"{prefix}_fmin", 0.01))),
        "advanced_fmax": str(st.session_state.get(f"{prefix}_advanced_fmax", st.session_state.get(f"{prefix}_fmax", 1.0))),
        "advanced_period_bins": str(st.session_state.get(f"{prefix}_advanced_bins", 200)),
        "advanced_window_width": str(st.session_state.get(f"{prefix}_advanced_width", 100.0)),
        "advanced_window_step": str(st.session_state.get(f"{prefix}_advanced_step", 25.0)),
        "advanced_min_points": str(st.session_state.get(f"{prefix}_advanced_min_points", 30)),
        "advanced_metric": st.session_state.get(f"{prefix}_advanced_metric", "power"),
        "advanced_wwz_decay": str(st.session_state.get(f"{prefix}_advanced_wwz_decay", 0.0125)),
    }
    if st.session_state.get(f"{prefix}_advanced_constrain_track", False):
        track_period = st.session_state.get(f"{prefix}_advanced_track_period")
        if track_period is not None:
            fields["advanced_track_period"] = str(track_period)
            fields["advanced_track_width_fraction"] = str(
                float(st.session_state.get(f"{prefix}_advanced_track_width_percent", 10.0)) / 100.0
            )
    if periods:
        fields["fold_fit_periods"] = ",".join(f"{p:.12g}" for p in periods)
    if excluded:
        fields["excluded_periods"] = ",".join(f"{p:.12g}" for p in excluded)
    fits_extension, fits_time_col, fits_flux_col, fits_error_col = selected_fits_columns(prefix)
    if fits_extension is not None:
        fields["fits_extension"] = str(fits_extension)
        fields["fits_time_col"] = fits_time_col or ""
        fields["fits_flux_col"] = fits_flux_col or ""
        fields["fits_error_col"] = fits_error_col or ""
    chain = st.session_state.get("app_prewhitening_periods", [])
    if chain:
        fields["prewhiten_periods"] = ",".join(f"{p:.12g}" for p in chain)
    t0 = st.session_state.get(f"{prefix}_t0", "").strip()
    if t0:
        fields["t0"] = t0
    return fields


def run_with_current_file(fields: dict[str, str]) -> dict:
    if "app_file_bytes" not in st.session_state:
        raise ValueError("Upload a file and run the analysis first.")
    return run_analysis(fields, st.session_state["app_file_bytes"], st.session_state["app_filename"])


def run_analysis_action(prefix: str, location=st) -> None:
    fields = fields_from_state(prefix)
    with st.spinner("Running analysis..."):
        try:
            st.session_state["app_result"] = run_with_current_file(fields)
            st.session_state["app_fields"] = fields
            st.session_state.pop("app_advanced_result", None)
            st.session_state.pop("app_model_lab_result", None)
            st.session_state["app_show_model"] = False
            st.session_state["app_show_help"] = False
            st.session_state["app_show_bibliography"] = False
        except ValueError as exc:
            location.error(str(exc))
        else:
            st.rerun()


def input_controls(prefix: str, location=st) -> None:
    if "app_upload_key" not in st.session_state:
        st.session_state["app_upload_key"] = 0
    uploaded = location.file_uploader(
        "ASCII or FITS file",
        type=None,
        accept_multiple_files=False,
        key=f"app_upload_{st.session_state['app_upload_key']}",
    )
    if uploaded is not None:
        signature = file_signature(uploaded)
        if signature != st.session_state.get("app_file_signature"):
            clear_state()
            st.session_state["app_file_signature"] = signature
            suggestion = suggested_frequency_range(
                uploaded,
                int(st.session_state.get(f"{prefix}_time_col", 1)),
                st.session_state.get(f"{prefix}_time_unit", "days"),
            )
            if suggestion:
                st.session_state[f"{prefix}_fmin"] = suggestion[0]
                st.session_state[f"{prefix}_fmax"] = suggestion[1]
                st.session_state[f"{prefix}_frequency_suggestion"] = suggestion
        st.session_state["app_file_bytes"] = uploaded.getvalue()
        st.session_state["app_filename"] = uploaded.name
    current_bytes = st.session_state.get("app_file_bytes")
    current_filename = st.session_state.get("app_filename", "")
    if current_bytes is not None and is_fits_upload(current_filename, current_bytes):
        location.caption("FITS extension and column selectors are shown in the workspace preview.")
    else:
        cols = location.columns(3)
        cols[0].number_input("Time", min_value=1, value=st.session_state.get(f"{prefix}_time_col", 1), key=f"{prefix}_time_col")
        cols[1].number_input("Flux", min_value=1, value=st.session_state.get(f"{prefix}_flux_col", 2), key=f"{prefix}_flux_col")
        cols[2].number_input("Error", min_value=1, value=st.session_state.get(f"{prefix}_error_col", 3), key=f"{prefix}_error_col")
    option_cols = location.columns([0.40, 0.30, 0.30])
    option_cols[0].selectbox(
        "Time units",
        ["days", "seconds"],
        index=0 if st.session_state.get(f"{prefix}_time_unit", "days") == "days" else 1,
        key=f"{prefix}_time_unit",
    )
    option_cols[1].checkbox("Use error column", value=st.session_state.get(f"{prefix}_use_error", True), key=f"{prefix}_use_error")
    option_cols[2].checkbox("Flux is magnitude", value=st.session_state.get(f"{prefix}_flux_is_magnitude", False), key=f"{prefix}_flux_is_magnitude")
    location.caption("Target identification")
    target_cols = location.columns([0.62, 0.38])
    target_name = target_cols[0].text_input("Object name", key=f"{prefix}_target_name", placeholder="e.g. 4U 2206+54")
    target_cols[1].markdown("<div style='height: 1.78rem'></div>", unsafe_allow_html=True)
    if target_cols[1].button("SIMBAD search", use_container_width=True, key=f"{prefix}_resolve_target"):
        try:
            ra_deg, dec_deg, resolved = resolve_target_name(str(st.session_state.get(f"{prefix}_target_name", "")))
        except Exception as exc:
            st.session_state[f"{prefix}_target_resolved"] = False
            st.session_state[f"{prefix}_target_resolved_name"] = ""
            location.error(f"Could not resolve target name: {exc}")
        else:
            st.session_state[f"{prefix}_target_ra_deg"] = f"{ra_deg:.10f}"
            st.session_state[f"{prefix}_target_dec_deg"] = f"{dec_deg:.10f}"
            st.session_state[f"{prefix}_target_resolved"] = True
            st.session_state[f"{prefix}_target_resolved_name"] = str(st.session_state.get(f"{prefix}_target_name", "")).strip()
            if not st.session_state.get("report_object_name"):
                st.session_state["report_object_name"] = st.session_state.get(f"{prefix}_target_name", "")
            location.success(f"Resolved coordinates: {resolved}")
    coord_cols = location.columns(2)
    coord_cols[0].text_input("RA [deg]", key=f"{prefix}_target_ra_deg", placeholder="auto/manual")
    coord_cols[1].text_input("Dec [deg]", key=f"{prefix}_target_dec_deg", placeholder="auto/manual")
    resolved_name = str(st.session_state.get(f"{prefix}_target_resolved_name", "")).strip()
    target_ready = (
        bool(st.session_state.get(f"{prefix}_target_resolved", False))
        and str(target_name).strip()
        and str(target_name).strip() == resolved_name
        and str(st.session_state.get(f"{prefix}_target_ra_deg", "")).strip()
        and str(st.session_state.get(f"{prefix}_target_dec_deg", "")).strip()
    )
    if location.button(
        "Bibliographic Search",
        use_container_width=True,
        key=f"{prefix}_show_bibliography_input",
        disabled=not target_ready,
        help="Resolve the object with SIMBAD before searching bibliography." if not target_ready else "Search literature mentioning time series, periods, periodicity, timing, or variability.",
    ):
        st.session_state["app_show_bibliography"] = True
        st.session_state["app_show_help"] = False
        st.rerun()
    location.caption("Analysis limits")
    limit_cols = location.columns(4)
    limit_cols[0].text_input("xmin", value=st.session_state.get(f"{prefix}_xmin", ""), key=f"{prefix}_xmin", placeholder="auto")
    limit_cols[1].text_input("xmax", value=st.session_state.get(f"{prefix}_xmax", ""), key=f"{prefix}_xmax", placeholder="auto")
    limit_cols[2].text_input("ymin", value=st.session_state.get(f"{prefix}_ymin", ""), key=f"{prefix}_ymin", placeholder="auto")
    limit_cols[3].text_input("ymax", value=st.session_state.get(f"{prefix}_ymax", ""), key=f"{prefix}_ymax", placeholder="auto")
    if location.button("Clear workspace", use_container_width=True, key=f"{prefix}_clear"):
        clear_state(reset_file=True)
        st.rerun()


def search_controls(prefix: str, location=st) -> None:
    if location.button("Run analysis", type="primary", use_container_width=True, key=f"{prefix}_run"):
        run_analysis_action(prefix, location)
    time_unit = st.session_state.get(f"{prefix}_time_unit", "days")
    suggestion = None
    file_bytes = st.session_state.get("app_file_bytes")
    filename = st.session_state.get("app_filename", "")
    if file_bytes is not None and is_fits_upload(filename, file_bytes):
        try:
            time, _, _ = selected_light_curve(file_bytes, filename, prefix)
            suggestion = suggested_frequency_range_from_time(time, time_unit)
        except ValueError:
            suggestion = None
    else:
        suggestion = suggested_frequency_range_from_bytes(
            file_bytes,
            int(st.session_state.get(f"{prefix}_time_col", 1)),
            time_unit,
        )
    if suggestion is None and st.session_state.get("app_result"):
        suggestion = suggested_frequency_range_from_time(st.session_state["app_result"]["series"]["time"], time_unit)
    if suggestion is not None:
        st.session_state[f"{prefix}_frequency_suggestion"] = suggestion
    else:
        suggestion = st.session_state.get(f"{prefix}_frequency_suggestion")
    if suggestion:
        labels = axis_labels(time_unit)
        frequency_unit = "Hz" if labels["baseline"] == "s" else "cycles/day"
        location.info(f"{suggestion[2]}\n\nSuggested range: {suggestion[0]:.6g} - {suggestion[1]:.6g} {frequency_unit}.")
        if location.button("Use suggested frequency range", use_container_width=True, key=f"{prefix}_use_suggested_frequency"):
            st.session_state[f"{prefix}_fmin"] = float(suggestion[0])
            st.session_state[f"{prefix}_fmax"] = float(suggestion[1])
            st.rerun()
    cols = location.columns(2)
    frequency_label = axis_labels(time_unit)["frequency"]
    cols[0].number_input(f"Min {frequency_label}", min_value=0.0, step=0.001, format="%.6f", key=f"{prefix}_fmin", value=st.session_state.get(f"{prefix}_fmin", 0.01))
    cols[1].number_input(f"Max {frequency_label}", min_value=0.0, step=0.01, format="%.6f", key=f"{prefix}_fmax", value=st.session_state.get(f"{prefix}_fmax", 1.0))
    location.number_input("Max considered peaks", min_value=1, max_value=20, value=st.session_state.get(f"{prefix}_max_peaks", 6), step=1, key=f"{prefix}_max_peaks")
    cols = location.columns(2)
    cols[0].number_input("Sampling-window threshold", min_value=0.0, value=st.session_state.get(f"{prefix}_window_threshold", 0.01), step=0.005, format="%.4f", key=f"{prefix}_window_threshold")
    cols[1].number_input("Sampling-window tolerance", min_value=0.001, value=st.session_state.get(f"{prefix}_window_tolerance", 0.01), step=0.001, format="%.3f", key=f"{prefix}_window_tolerance")
    cols = location.columns(2)
    cols[0].number_input("Samples per peak", min_value=1.0, value=st.session_state.get(f"{prefix}_samples_per_peak", 10.0), step=1.0, key=f"{prefix}_samples_per_peak")
    cols[1].number_input(f"Min. considered {axis_labels(time_unit)['period']}", min_value=0.0, value=st.session_state.get(f"{prefix}_min_period", 2.0), step=0.1, key=f"{prefix}_min_period")


def uncertainty_controls(prefix: str, location=st) -> None:
    cols = location.columns(2)
    cols[0].number_input("Bootstrap iterations", min_value=0, value=st.session_state.get(f"{prefix}_bootstrap", 1000), step=50, key=f"{prefix}_bootstrap")
    cols[1].number_input("Bootstrap local width", min_value=0.001, value=st.session_state.get(f"{prefix}_bootstrap_width", 0.03), step=0.001, format="%.3f", key=f"{prefix}_bootstrap_width")
    if location.button("Update uncertainties", use_container_width=True, key=f"{prefix}_update_uncertainties"):
        if "app_file_bytes" not in st.session_state:
            location.error("Upload a file and run the analysis first.")
        else:
            fields = fields_from_state(prefix)
            with st.spinner("Updating bootstrap uncertainties..."):
                st.session_state["app_result"] = run_with_current_file(fields)
                st.session_state["app_fields"] = fields
                st.session_state.pop("app_model_lab_result", None)
            st.rerun()


def manual_exclusion_controls(prefix: str, location=st) -> None:
    result = st.session_state.get("app_result")
    period_unit = (result or {}).get("period_unit", axis_labels(st.session_state.get(f"{prefix}_time_unit", "days"))["baseline"])
    options = exclusion_options(result)
    selected = location.multiselect(
        "Exclude periods from primary selection",
        options=list(options.keys()),
        format_func=lambda period: options[period],
        key=f"{prefix}_selected_exclusions",
    )
    manual = location.text_input(f"User-specified period(s) [{period_unit}]", key=f"{prefix}_manual_exclusions")
    excluded = list(selected)
    if manual.strip():
        try:
            excluded.extend(parse_periods(manual))
        except ValueError as exc:
            location.error(str(exc))
    st.session_state[f"{prefix}_excluded_periods"] = unique_periods(excluded)
    location.number_input("Manual exclusion tolerance", min_value=0.001, value=st.session_state.get(f"{prefix}_exclusion_tolerance", 0.015), step=0.001, format="%.3f", key=f"{prefix}_exclusion_tolerance")
    if excluded:
        location.caption("Excluded: " + ", ".join(f"{period:.4g} {period_unit}" for period in st.session_state[f"{prefix}_excluded_periods"]))
    if location.button("Apply exclusions", use_container_width=True, key=f"{prefix}_apply_exclusions"):
        if "app_file_bytes" not in st.session_state:
            location.error("Upload a file and run the analysis first.")
        else:
            fields = fields_from_state(prefix, bootstrap_override=0)
            with st.spinner("Applying manual exclusions..."):
                st.session_state["app_result"] = run_with_current_file(fields)
                st.session_state["app_fields"] = fields
                st.session_state.pop("app_model_lab_result", None)
            st.rerun()


def folded_controls(prefix: str, location=st) -> None:
    result = st.session_state.get("app_result")
    period_unit = (result or {}).get("period_unit", axis_labels(st.session_state.get(f"{prefix}_time_unit", "days"))["baseline"])
    time_label = (result or {}).get("time_label", axis_labels(st.session_state.get(f"{prefix}_time_unit", "days"))["time"])
    cols = location.columns(2)
    cols[0].number_input("Phase bins", min_value=4, max_value=80, value=st.session_state.get(f"{prefix}_fold_bins", 10), key=f"{prefix}_fold_bins")
    cols[1].text_input(f"T0 ({time_label})", value=st.session_state.get(f"{prefix}_t0", ""), key=f"{prefix}_t0")
    mode = location.radio("Folded-fit frequencies", ["harmonics", "selected"], horizontal=True, key=f"{prefix}_fold_mode")
    if mode == "harmonics":
        location.number_input("Number of harmonics", min_value=1, max_value=8, value=st.session_state.get(f"{prefix}_fold_harmonics", 1), key=f"{prefix}_fold_harmonics")
        st.session_state[f"{prefix}_selected_periods"] = []
        if result and result.get("folded_period"):
            location.caption(
                "Folding period: "
                + f"{float(result['folded_period']):.6g} {period_unit}; fitted periods: "
                + ", ".join(f"{float(period):.6g} {period_unit}" for period in result.get("fold_fit_periods", []))
            )
    else:
        options = period_options(result, "peaks") | period_options(result, "residual_peaks")
        selected = location.multiselect("Use detected periods", options=list(options.keys()), format_func=lambda p: options[p], key=f"{prefix}_detected_fold_periods")
        manual = location.text_input(f"Additional periods [{period_unit}]", key=f"{prefix}_manual_periods")
        periods = list(selected)
        if manual.strip():
            try:
                periods.extend(parse_periods(manual))
            except ValueError as exc:
                location.error(str(exc))
        st.session_state[f"{prefix}_selected_periods"] = unique_periods(periods)
        if st.session_state[f"{prefix}_selected_periods"]:
            location.caption(
                "Folding on first selected period; fitted periods: "
                + ", ".join(f"{period:.6g} {period_unit}" for period in st.session_state[f"{prefix}_selected_periods"])
            )
    fit_options = ["standard", "robust", "display-optimized"]
    current_fit_method = st.session_state.get(f"{prefix}_model_fit_method", "standard")
    if current_fit_method not in fit_options:
        current_fit_method = "standard"
    fit_method = location.selectbox(
        "Sinusoidal fitting",
        fit_options,
        index=fit_options.index(current_fit_method),
        key=f"{prefix}_model_fit_method",
    )
    fit_help = {
        "standard": "Weighted least-squares fit using the provided errors.",
        "robust": "Robust soft-L1 fit that downweights outliers and over-dominant points.",
        "display-optimized": "Folded-display fit with amplitude and offset matched to an adaptive outer data range; prewhitening subtraction remains standard.",
    }
    location.caption(fit_help[fit_method])
    if location.button("Update folded profile", use_container_width=True, key=f"{prefix}_update_fold"):
        current_result = st.session_state.get("app_result")
        if not current_result:
            location.error("Run an analysis first.")
        else:
            fields = fields_from_state(prefix, bootstrap_override=0)
            st.session_state["app_result"] = update_folded_profile(current_result, fields)
            st.session_state["app_fields"] = fields
            st.session_state.pop("app_model_lab_result", None)
            st.rerun()


def prewhitening_controls(prefix: str, location=st) -> None:
    result = st.session_state.get("app_result")
    frequency_label = (result or {}).get("frequency_unit", axis_labels(st.session_state.get(f"{prefix}_time_unit", "days"))["frequency"])
    period_unit = (result or {}).get("period_unit", axis_labels(st.session_state.get(f"{prefix}_time_unit", "days"))["baseline"])
    location.caption(
        "Residual LS uses the global Frequency Search range: "
        f"{float(st.session_state.get(f'{prefix}_fmin', 0.01)):.6g} - "
        f"{float(st.session_state.get(f'{prefix}_fmax', 1.0)):.6g} {frequency_label}."
    )
    options = period_options(result, "residual_peaks") or period_options(result, "peaks")
    selected_period = None
    if options:
        option_labels = [options[period] for period in options]
        label_to_period = {options[period]: period for period in options}
        select_version = st.session_state.get(f"{prefix}_next_prewhitening_select_version", 0)
        selected_label = location.selectbox(
            "Next period to remove",
            options=option_labels,
            index=None,
            placeholder="Choose a detected period",
            key=f"{prefix}_next_prewhitening_label_{select_version}",
        )
        selected_period = label_to_period.get(selected_label)
        if selected_period is not None:
            location.caption(f"Selected for next step: {float(selected_period):.6g} {period_unit}")
    else:
        location.caption("Run an analysis to populate candidates.")
    manual_period_text = location.text_input(
        f"Manual period for next step [{period_unit}]",
        value=st.session_state.get(f"{prefix}_manual_prewhitening_period", ""),
        key=f"{prefix}_manual_prewhitening_period",
        placeholder="Optional; overrides dropdown",
    )
    if st.session_state.get("app_prewhitening_periods"):
        location.caption("Chain: " + ", ".join(f"{p:.4g} {period_unit}" for p in st.session_state["app_prewhitening_periods"]))
    cols = location.columns(3)
    if cols[0].button("Next step", use_container_width=True, key=f"{prefix}_next_step"):
        period_to_add = selected_period
        if manual_period_text.strip():
            try:
                manual_periods = parse_periods(manual_period_text)
            except ValueError as exc:
                location.error(str(exc))
                return
            if len(manual_periods) != 1:
                location.error("Enter one manual period for the next prewhitening step.")
                return
            period_to_add = manual_periods[0]
        if period_to_add is None:
            location.error("Choose a detected period or enter one manual period first.")
        else:
            previous_chain = list(st.session_state.get("app_prewhitening_periods", []))
            st.session_state["app_prewhitening_periods"] = unique_periods(
                previous_chain + [float(period_to_add)]
            )
            fields = fields_from_state(prefix, bootstrap_override=0)
            with st.spinner("Running prewhitening step..."):
                try:
                    st.session_state["app_result"] = run_with_current_file(fields)
                    st.session_state["app_fields"] = fields
                    st.session_state.pop("app_model_lab_result", None)
                    st.session_state["app_show_model"] = False
                except ValueError as exc:
                    st.session_state["app_prewhitening_periods"] = previous_chain
                    location.error(str(exc))
                    return
            st.session_state[f"{prefix}_next_prewhitening_select_version"] = (
                st.session_state.get(f"{prefix}_next_prewhitening_select_version", 0) + 1
            )
            st.rerun()
    if cols[1].button("Show model", use_container_width=True, key=f"{prefix}_show_model"):
        st.session_state["app_show_model"] = True
        st.rerun()
    location.checkbox("Show errors", value=True, key=f"{prefix}_show_model_errors")
    if cols[2].button("Clear chain", use_container_width=True, key=f"{prefix}_clear_chain"):
        st.session_state["app_prewhitening_periods"] = []
        st.session_state["app_show_model"] = False
        st.session_state[f"{prefix}_next_prewhitening_select_version"] = (
            st.session_state.get(f"{prefix}_next_prewhitening_select_version", 0) + 1
        )
        if "app_file_bytes" in st.session_state:
            st.session_state["app_result"] = run_with_current_file(fields_from_state(prefix, bootstrap_override=0))
            st.session_state.pop("app_model_lab_result", None)
        st.rerun()


def advanced_controls(prefix: str, location=st) -> None:
    result = st.session_state.get("app_result")
    labels = axis_labels(st.session_state.get(f"{prefix}_time_unit", "days"))
    frequency_unit = (result or {}).get("frequency_unit", "Hz" if labels["baseline"] == "s" else "cycles/day")
    time_unit = (result or {}).get("baseline_unit", labels["baseline"])
    method_label = location.radio("Method", ["v1 Sliding LS", "v2 WWZ"], horizontal=True, key=f"{prefix}_advanced_method_label")
    st.session_state[f"{prefix}_advanced_method"] = "v2" if method_label == "v2 WWZ" else "v1"
    cols = location.columns(2)
    advanced_fmin_value = max(
        float(st.session_state.get(f"{prefix}_advanced_fmin", st.session_state.get(f"{prefix}_fmin", 0.01))),
        1e-12,
    )
    advanced_fmax_value = max(
        float(st.session_state.get(f"{prefix}_advanced_fmax", st.session_state.get(f"{prefix}_fmax", 1.0))),
        1e-12,
    )
    cols[0].number_input(
        f"Min. frequency [{frequency_unit}]",
        min_value=1e-12,
        value=advanced_fmin_value,
        step=0.001,
        format="%.6f",
        key=f"{prefix}_advanced_fmin",
    )
    cols[1].number_input(
        f"Max. frequency [{frequency_unit}]",
        min_value=1e-12,
        value=advanced_fmax_value,
        step=0.01,
        format="%.6f",
        key=f"{prefix}_advanced_fmax",
    )
    cols = location.columns(3)
    cols[0].number_input("Period bins", min_value=20, max_value=1000, value=st.session_state.get(f"{prefix}_advanced_bins", 200), step=20, key=f"{prefix}_advanced_bins")
    cols[1].number_input(f"Window width [{time_unit}]", min_value=0.0, value=st.session_state.get(f"{prefix}_advanced_width", 100.0), step=10.0, key=f"{prefix}_advanced_width")
    cols[2].number_input(f"Window step [{time_unit}]", min_value=0.0, value=st.session_state.get(f"{prefix}_advanced_step", 25.0), step=5.0, key=f"{prefix}_advanced_step")
    location.number_input("Min. points per window", min_value=3, value=st.session_state.get(f"{prefix}_advanced_min_points", 30), step=5, key=f"{prefix}_advanced_min_points")
    if st.session_state[f"{prefix}_advanced_method"] == "v1":
        location.selectbox("Color metric", ["power", "amplitude"], key=f"{prefix}_advanced_metric")
        constrain_track = location.checkbox(
            "Keep best-period track near a detected period",
            value=st.session_state.get(f"{prefix}_advanced_constrain_track", False),
            key=f"{prefix}_advanced_constrain_track",
        )
        if constrain_track:
            options = period_options(result, "peaks") | period_options(result, "residual_peaks")
            if options:
                selected_track_period = location.selectbox(
                    "Track reference period",
                    options=list(options.keys()),
                    format_func=lambda period: options[period],
                    key=f"{prefix}_advanced_track_period",
                )
                period_unit = (result or {}).get("period_unit", labels["baseline"])
                location.number_input(
                    "Track half-width [%]",
                    min_value=0.1,
                    max_value=100.0,
                    value=float(st.session_state.get(f"{prefix}_advanced_track_width_percent", 10.0)),
                    step=1.0,
                    format="%.1f",
                    key=f"{prefix}_advanced_track_width_percent",
                )
                half_width = float(st.session_state.get(f"{prefix}_advanced_track_width_percent", 10.0)) / 100.0
                location.caption(
                    f"Track will search around {float(selected_track_period):.6g} {period_unit} "
                    f"within +/- {100.0 * half_width:.1f}%."
                )
            else:
                st.session_state.pop(f"{prefix}_advanced_track_period", None)
                location.caption("Run an analysis with detected periods before constraining the track.")
    else:
        st.session_state[f"{prefix}_advanced_metric"] = "WWZ"
        location.number_input("WWZ decay", min_value=0.0001, value=st.session_state.get(f"{prefix}_advanced_wwz_decay", 0.0125), step=0.0025, format="%.5f", key=f"{prefix}_advanced_wwz_decay")
    show_track = location.checkbox("Show best period track", value=True, key=f"{prefix}_advanced_track")
    if location.button("Build tomogram", use_container_width=True, key=f"{prefix}_run_advanced"):
        if not result:
            location.error("Run an analysis first.")
        else:
            fields = fields_from_state(prefix, bootstrap_override=0)
            advanced_fmin = float(fields.get("advanced_fmin", fields.get("fmin", "0.01")))
            advanced_fmax = float(fields.get("advanced_fmax", fields.get("fmax", "1.0")))
            if advanced_fmin <= 0 or advanced_fmax <= advanced_fmin:
                location.error("Advanced frequency range must satisfy 0 < min frequency < max frequency.")
                return
            with st.spinner("Running advanced map..."):
                try:
                    advanced = advanced_time_frequency_map(result, fields)
                    advanced["show_best_track"] = show_track
                    advanced["period_label"] = result.get("period_label", labels["period"])
                    st.session_state["app_advanced_result"] = advanced
                except ValueError as exc:
                    location.error(str(exc))
                else:
                    st.rerun()


def model_lab_controls(prefix: str, location=st) -> None:
    result = st.session_state.get("app_result")
    family = location.selectbox(
        "Model family",
        [
            "Fourier multi-harmonic",
            "Binary interpretation assistant",
            "Eclipsing / eccentric binaries",
            "Bondi-Hoyle accretion",
            "X-ray pulsation timing",
        ],
        key=f"{prefix}_model_lab_family",
    )
    if not result:
        location.caption("Run an analysis first, then fit models here.")
        return
    period_unit = result.get("period_unit", axis_labels(st.session_state.get(f"{prefix}_time_unit", "days"))["baseline"])
    default_period = result.get("folded_period") or result.get("primary_period") or ""
    if family == "Binary interpretation assistant":
        location.caption("Compact folded-profile diagnostics for eclipsing-binary morphology, half-period ambiguity, and minimum spacing.")
        if location.button("Run binary assistant", use_container_width=True, key=f"{prefix}_run_binary_assistant"):
            try:
                st.session_state["app_model_lab_result"] = binary_interpretation_assistant(result)
            except ValueError as exc:
                location.error(str(exc))
                return
            st.rerun()
        return

    if family == "X-ray pulsation timing":
        if f"{prefix}_model_lab_pulse_period" not in st.session_state and default_period != "":
            st.session_state[f"{prefix}_model_lab_pulse_period"] = f"{float(default_period):.12g}"
        cols = location.columns(2)
        cols[0].text_input(
            f"Pulse period [{period_unit}]",
            key=f"{prefix}_model_lab_pulse_period",
            placeholder="Default: folded/primary",
        )
        cols[1].text_input(
            f"T0 [{result.get('baseline_unit', period_unit)}]",
            value=st.session_state.get(f"{prefix}_model_lab_pulse_t0", ""),
            key=f"{prefix}_model_lab_pulse_t0",
            placeholder=f"{float(result.get('t0', 0.0)):.6g}",
        )
        cols = location.columns(2)
        cols[0].number_input("Profile bins", min_value=4, max_value=120, value=st.session_state.get(f"{prefix}_model_lab_pulse_bins", 24), key=f"{prefix}_model_lab_pulse_bins")
        cols[1].number_input("Profile harmonics", min_value=1, max_value=20, value=st.session_state.get(f"{prefix}_model_lab_pulse_harmonics", 3), key=f"{prefix}_model_lab_pulse_harmonics")
        cols = location.columns(2)
        cols[0].number_input("Epoch search half-width [%]", min_value=0.0, max_value=50.0, value=st.session_state.get(f"{prefix}_model_lab_pulse_search_width", 2.0), step=0.25, format="%.2f", key=f"{prefix}_model_lab_pulse_search_width")
        cols[1].number_input("Trial periods", min_value=30, max_value=5000, value=st.session_state.get(f"{prefix}_model_lab_pulse_trials", 300), step=10, key=f"{prefix}_model_lab_pulse_trials")
        location.number_input("Epoch-folding bins", min_value=4, max_value=80, value=st.session_state.get(f"{prefix}_model_lab_pulse_epoch_bins", 16), key=f"{prefix}_model_lab_pulse_epoch_bins")
        cols = location.columns(2)
        cols[0].number_input("Pulse-arrival segments", min_value=2, max_value=80, value=st.session_state.get(f"{prefix}_model_lab_pulse_segments", 8), key=f"{prefix}_model_lab_pulse_segments")
        cols[1].number_input("Min points per segment", min_value=4, max_value=500, value=st.session_state.get(f"{prefix}_model_lab_pulse_min_points", 20), key=f"{prefix}_model_lab_pulse_min_points")
        location.text_input(
            f"O-C sinusoid period [{period_unit}]",
            value=st.session_state.get(f"{prefix}_model_lab_pulse_orbital_period", ""),
            key=f"{prefix}_model_lab_pulse_orbital_period",
            placeholder="optional orbital trial period",
        )
        fit_options = ["standard", "robust", "display-optimized"]
        global_fit = st.session_state.get(f"{prefix}_model_fit_method", "standard")
        if global_fit not in fit_options:
            global_fit = "standard"
        current_fit = st.session_state.get(f"{prefix}_model_lab_pulse_fit_method", global_fit)
        if current_fit not in fit_options:
            current_fit = global_fit
        location.selectbox(
            "Pulse-shape fit method",
            fit_options,
            index=fit_options.index(current_fit),
            key=f"{prefix}_model_lab_pulse_fit_method",
        )
        location.checkbox(
            "Show full data set with model",
            value=st.session_state.get(f"{prefix}_model_lab_show_time_model", True),
            key=f"{prefix}_model_lab_show_time_model",
        )
        if location.button("Run pulse analysis", use_container_width=True, key=f"{prefix}_fit_pulse_model"):
            fields = fields_from_state(prefix, bootstrap_override=0)
            fields.update({
                "model_lab_pulse_period": str(st.session_state.get(f"{prefix}_model_lab_pulse_period", "")).strip(),
                "model_lab_pulse_t0": str(st.session_state.get(f"{prefix}_model_lab_pulse_t0", "")).strip(),
                "model_lab_pulse_bins": str(st.session_state.get(f"{prefix}_model_lab_pulse_bins", 24)),
                "model_lab_pulse_harmonics": str(st.session_state.get(f"{prefix}_model_lab_pulse_harmonics", 3)),
                "model_lab_pulse_search_width": str(st.session_state.get(f"{prefix}_model_lab_pulse_search_width", 2.0)),
                "model_lab_pulse_trials": str(st.session_state.get(f"{prefix}_model_lab_pulse_trials", 300)),
                "model_lab_pulse_epoch_bins": str(st.session_state.get(f"{prefix}_model_lab_pulse_epoch_bins", 16)),
                "model_lab_pulse_segments": str(st.session_state.get(f"{prefix}_model_lab_pulse_segments", 8)),
                "model_lab_pulse_min_points": str(st.session_state.get(f"{prefix}_model_lab_pulse_min_points", 20)),
                "model_lab_pulse_orbital_period": str(st.session_state.get(f"{prefix}_model_lab_pulse_orbital_period", "")).strip(),
                "model_lab_pulse_fit_method": st.session_state.get(f"{prefix}_model_lab_pulse_fit_method", global_fit),
            })
            with st.spinner("Running pulse period analysis..."):
                try:
                    st.session_state["app_model_lab_result"] = pulse_period_model_lab(result, fields)
                except ValueError as exc:
                    location.error(str(exc))
                    return
            st.rerun()
        return

    if family == "Fourier multi-harmonic":
        if f"{prefix}_model_lab_period" not in st.session_state and default_period != "":
            st.session_state[f"{prefix}_model_lab_period"] = f"{float(default_period):.12g}"
        cols = location.columns(2)
        cols[0].text_input(
            f"Model period [{period_unit}]",
            key=f"{prefix}_model_lab_period",
            placeholder="Default: folded/primary",
        )
        cols[1].text_input(
            f"T0 [{result.get('baseline_unit', period_unit)}]",
            value=st.session_state.get(f"{prefix}_model_lab_t0", ""),
            key=f"{prefix}_model_lab_t0",
            placeholder=f"{float(result.get('t0', 0.0)):.6g}",
        )
        location.number_input("Display phase bins", min_value=4, max_value=120, value=st.session_state.get(f"{prefix}_model_lab_bins", 20), key=f"{prefix}_model_lab_bins")
        selection = location.radio("Harmonic selection", ["manual", "AIC", "BIC"], horizontal=True, key=f"{prefix}_model_lab_selection")
        harmonic_cols = location.columns(2)
        harmonic_cols[0].number_input("Manual harmonics", min_value=1, max_value=20, value=st.session_state.get(f"{prefix}_model_lab_harmonics", 3), key=f"{prefix}_model_lab_harmonics")
        harmonic_cols[1].number_input("Max harmonics for AIC/BIC", min_value=1, max_value=20, value=st.session_state.get(f"{prefix}_model_lab_max_harmonics", 8), key=f"{prefix}_model_lab_max_harmonics")
        fit_options = ["standard", "robust", "display-optimized"]
        global_fit = st.session_state.get(f"{prefix}_model_fit_method", "standard")
        if global_fit not in fit_options:
            global_fit = "standard"
        current_fit = st.session_state.get(f"{prefix}_model_lab_fit_method", global_fit)
        if current_fit not in fit_options:
            current_fit = global_fit
        location.selectbox(
            "Fourier fit method",
            fit_options,
            index=fit_options.index(current_fit),
            key=f"{prefix}_model_lab_fit_method",
        )
        location.checkbox(
            "Show full data set with model",
            value=st.session_state.get(f"{prefix}_model_lab_show_time_model", True),
            key=f"{prefix}_model_lab_show_time_model",
        )
        if location.button("Fit Fourier model", use_container_width=True, key=f"{prefix}_fit_fourier_model"):
            fields = fields_from_state(prefix, bootstrap_override=0)
            fields.update({
                "model_lab_period": str(st.session_state.get(f"{prefix}_model_lab_period", "")).strip(),
                "model_lab_t0": str(st.session_state.get(f"{prefix}_model_lab_t0", "")).strip(),
                "model_lab_bins": str(st.session_state.get(f"{prefix}_model_lab_bins", 20)),
                "model_lab_fourier_selection": selection.lower(),
                "model_lab_fourier_harmonics": str(st.session_state.get(f"{prefix}_model_lab_harmonics", 3)),
                "model_lab_fourier_max_harmonics": str(st.session_state.get(f"{prefix}_model_lab_max_harmonics", 8)),
                "model_lab_fit_method": st.session_state.get(f"{prefix}_model_lab_fit_method", global_fit),
            })
            with st.spinner("Fitting Fourier model..."):
                try:
                    st.session_state["app_model_lab_result"] = fourier_model_lab(result, fields)
                except ValueError as exc:
                    location.error(str(exc))
                    return
            st.rerun()
        return

    if family == "Bondi-Hoyle accretion":
        if f"{prefix}_model_lab_bh_period" not in st.session_state and default_period != "":
            st.session_state[f"{prefix}_model_lab_bh_period"] = f"{float(default_period):.12g}"
        cols = location.columns(2)
        cols[0].text_input(
            f"Orbital period [{period_unit}]",
            key=f"{prefix}_model_lab_bh_period",
            placeholder="Default: folded/primary",
        )
        cols[1].text_input(
            f"T0 / periastron epoch [{result.get('baseline_unit', period_unit)}]",
            value=st.session_state.get(f"{prefix}_model_lab_bh_t0", ""),
            key=f"{prefix}_model_lab_bh_t0",
            placeholder=f"{float(result.get('t0', 0.0)):.6g}",
        )
        location.number_input("Display phase bins", min_value=4, max_value=120, value=st.session_state.get(f"{prefix}_model_lab_bh_bins", 24), key=f"{prefix}_model_lab_bh_bins")
        cols = location.columns(2)
        cols[0].number_input("Initial/fixed eccentricity", min_value=0.0, max_value=0.9, value=st.session_state.get(f"{prefix}_model_lab_bh_eccentricity", 0.3), step=0.05, format="%.3f", key=f"{prefix}_model_lab_bh_eccentricity")
        wind_mode = location.radio("Wind speed input", ["v_inf [km/s]", "v_wind / v_orb"], horizontal=True, key=f"{prefix}_model_lab_bh_wind_input_label")
        st.session_state[f"{prefix}_model_lab_bh_wind_input_mode"] = "ratio" if wind_mode == "v_wind / v_orb" else "v_inf"
        if st.session_state[f"{prefix}_model_lab_bh_wind_input_mode"] == "ratio":
            cols[1].number_input("v_wind / v_orb", min_value=0.05, max_value=20.0, value=st.session_state.get(f"{prefix}_model_lab_bh_wind_speed_ratio", 3.0), step=0.1, format="%.3f", key=f"{prefix}_model_lab_bh_wind_speed_ratio")
            location.number_input("Donor radius / a", min_value=0.001, max_value=0.95, value=st.session_state.get(f"{prefix}_model_lab_bh_donor_radius_over_a", 0.15), step=0.01, format="%.3f", key=f"{prefix}_model_lab_bh_donor_radius_over_a")
        else:
            cols[1].number_input("v_inf [km/s]", min_value=1.0, max_value=5000.0, value=st.session_state.get(f"{prefix}_model_lab_bh_vinf", 1000.0), step=50.0, format="%.1f", key=f"{prefix}_model_lab_bh_vinf")
            donor_options = ["Manual"] + list(BHL_DONOR_PRESETS.keys())
            preset_cols = location.columns(2)
            preset_cols[0].selectbox(
                "Donor spectral type",
                donor_options,
                index=donor_options.index(st.session_state.get(f"{prefix}_model_lab_bh_donor_spectral_type", "Manual"))
                if st.session_state.get(f"{prefix}_model_lab_bh_donor_spectral_type", "Manual") in donor_options
                else 0,
                key=f"{prefix}_model_lab_bh_donor_spectral_type",
            )
            preset_cols[1].selectbox(
                "Luminosity class",
                ["V", "III", "I"],
                index=["V", "III", "I"].index(st.session_state.get(f"{prefix}_model_lab_bh_donor_luminosity_class", "V"))
                if st.session_state.get(f"{prefix}_model_lab_bh_donor_luminosity_class", "V") in ["V", "III", "I"]
                else 0,
                key=f"{prefix}_model_lab_bh_donor_luminosity_class",
            )
            preset = bhl_donor_preset_values(
                st.session_state.get(f"{prefix}_model_lab_bh_donor_spectral_type", "Manual"),
                st.session_state.get(f"{prefix}_model_lab_bh_donor_luminosity_class", "V"),
            )
            if preset is not None:
                st.session_state[f"{prefix}_model_lab_bh_donor_mass"] = float(preset["mass_msun"])
                st.session_state[f"{prefix}_model_lab_bh_donor_radius"] = float(preset["radius_rsun"])
                location.caption(
                    "Typical donor values loaded: "
                    f"M={float(preset['mass_msun']):.4g} Msun, R={float(preset['radius_rsun']):.4g} Rsun. "
                    "Switch spectral type to Manual to edit them freely."
                )
            physical_cols = location.columns(3)
            preset_active = preset is not None
            physical_cols[0].number_input("Donor mass [Msun]", min_value=0.1, max_value=150.0, value=st.session_state.get(f"{prefix}_model_lab_bh_donor_mass", 18.0), step=0.5, format="%.2f", key=f"{prefix}_model_lab_bh_donor_mass", disabled=preset_active)
            physical_cols[1].number_input("Compact mass [Msun]", min_value=0.1, max_value=50.0, value=st.session_state.get(f"{prefix}_model_lab_bh_compact_mass", 1.4), step=0.1, format="%.2f", key=f"{prefix}_model_lab_bh_compact_mass")
            physical_cols[2].number_input("Donor radius [Rsun]", min_value=0.1, max_value=2000.0, value=st.session_state.get(f"{prefix}_model_lab_bh_donor_radius", 8.0), step=0.5, format="%.2f", key=f"{prefix}_model_lab_bh_donor_radius", disabled=preset_active)
        location.number_input("Wind beta", min_value=0.0, max_value=5.0, value=st.session_state.get(f"{prefix}_model_lab_bh_wind_beta", 0.8), step=0.1, format="%.2f", key=f"{prefix}_model_lab_bh_wind_beta")
        cols = location.columns(3)
        cols[0].checkbox("Fit eccentricity", value=st.session_state.get(f"{prefix}_model_lab_bh_fit_eccentricity", True), key=f"{prefix}_model_lab_bh_fit_eccentricity")
        cols[1].checkbox("Fit wind speed", value=st.session_state.get(f"{prefix}_model_lab_bh_fit_wind_speed", True), disabled=st.session_state[f"{prefix}_model_lab_bh_wind_input_mode"] == "v_inf", key=f"{prefix}_model_lab_bh_fit_wind_speed")
        cols[2].checkbox("Fit phase lag", value=st.session_state.get(f"{prefix}_model_lab_bh_phase_lag", True), key=f"{prefix}_model_lab_bh_phase_lag")
        location.checkbox(
            "Show full data set with model",
            value=st.session_state.get(f"{prefix}_model_lab_show_time_model", True),
            key=f"{prefix}_model_lab_show_time_model",
        )
        if location.button("Fit Bondi-Hoyle model", use_container_width=True, key=f"{prefix}_fit_bh_model"):
            fields = fields_from_state(prefix, bootstrap_override=0)
            fields.update({
                "model_lab_bh_period": str(st.session_state.get(f"{prefix}_model_lab_bh_period", "")).strip(),
                "model_lab_bh_t0": str(st.session_state.get(f"{prefix}_model_lab_bh_t0", "")).strip(),
                "model_lab_bh_bins": str(st.session_state.get(f"{prefix}_model_lab_bh_bins", 24)),
                "model_lab_bh_eccentricity": str(st.session_state.get(f"{prefix}_model_lab_bh_eccentricity", 0.3)),
                "model_lab_bh_wind_input_mode": st.session_state.get(f"{prefix}_model_lab_bh_wind_input_mode", "v_inf"),
                "model_lab_bh_wind_speed_ratio": str(st.session_state.get(f"{prefix}_model_lab_bh_wind_speed_ratio", 3.0)),
                "model_lab_bh_donor_radius_over_a": str(st.session_state.get(f"{prefix}_model_lab_bh_donor_radius_over_a", 0.15)),
                "model_lab_bh_vinf": str(st.session_state.get(f"{prefix}_model_lab_bh_vinf", 1000.0)),
                "model_lab_bh_donor_spectral_type": st.session_state.get(f"{prefix}_model_lab_bh_donor_spectral_type", "Manual"),
                "model_lab_bh_donor_luminosity_class": st.session_state.get(f"{prefix}_model_lab_bh_donor_luminosity_class", "V"),
                "model_lab_bh_donor_mass": str(st.session_state.get(f"{prefix}_model_lab_bh_donor_mass", 18.0)),
                "model_lab_bh_compact_mass": str(st.session_state.get(f"{prefix}_model_lab_bh_compact_mass", 1.4)),
                "model_lab_bh_donor_radius": str(st.session_state.get(f"{prefix}_model_lab_bh_donor_radius", 8.0)),
                "model_lab_bh_wind_beta": str(st.session_state.get(f"{prefix}_model_lab_bh_wind_beta", 0.8)),
                "model_lab_bh_fit_eccentricity": "true" if st.session_state.get(f"{prefix}_model_lab_bh_fit_eccentricity", True) else "false",
                "model_lab_bh_fit_wind_speed": "true" if st.session_state.get(f"{prefix}_model_lab_bh_fit_wind_speed", True) else "false",
                "model_lab_bh_phase_lag": "true" if st.session_state.get(f"{prefix}_model_lab_bh_phase_lag", True) else "false",
            })
            with st.spinner("Fitting Bondi-Hoyle toy model..."):
                try:
                    st.session_state["app_model_lab_result"] = bondi_hoyle_model_lab(result, fields)
                except ValueError as exc:
                    location.error(str(exc))
                    return
            st.rerun()
        return

    if f"{prefix}_model_lab_binary_period" not in st.session_state and default_period != "":
        st.session_state[f"{prefix}_model_lab_binary_period"] = f"{float(default_period):.12g}"
    cols = location.columns(2)
    cols[0].text_input(
        f"Binary period [{period_unit}]",
        key=f"{prefix}_model_lab_binary_period",
        placeholder="Default: folded/primary",
    )
    cols[1].text_input(
        f"T0 / periastron epoch [{result.get('baseline_unit', period_unit)}]",
        value=st.session_state.get(f"{prefix}_model_lab_binary_t0", ""),
        key=f"{prefix}_model_lab_binary_t0",
        placeholder=f"{float(result.get('t0', 0.0)):.6g}",
    )
    location.number_input("Display phase bins", min_value=4, max_value=120, value=st.session_state.get(f"{prefix}_model_lab_binary_bins", 24), key=f"{prefix}_model_lab_binary_bins")
    binary_kind = location.radio(
        "Binary model",
        ["eccentric harmonic", "empirical eclipses", "physical eclipse toy"],
        horizontal=True,
        key=f"{prefix}_model_lab_binary_kind_label",
    )
    if binary_kind == "physical eclipse toy":
        st.session_state[f"{prefix}_model_lab_binary_kind"] = "physical_eclipse_toy"
    elif binary_kind == "empirical eclipses":
        st.session_state[f"{prefix}_model_lab_binary_kind"] = "empirical_eclipses"
    else:
        st.session_state[f"{prefix}_model_lab_binary_kind"] = "eccentric_harmonic"
    if st.session_state[f"{prefix}_model_lab_binary_kind"] == "eccentric_harmonic":
        cols = location.columns(2)
        cols[0].number_input("True-anomaly harmonics", min_value=1, max_value=8, value=st.session_state.get(f"{prefix}_model_lab_binary_harmonics", 2), key=f"{prefix}_model_lab_binary_harmonics")
        cols[1].number_input("Initial/fixed eccentricity", min_value=0.0, max_value=0.9, value=st.session_state.get(f"{prefix}_model_lab_binary_eccentricity", 0.2), step=0.05, format="%.3f", key=f"{prefix}_model_lab_binary_eccentricity")
        location.checkbox("Fit eccentricity", value=st.session_state.get(f"{prefix}_model_lab_binary_fit_eccentricity", True), key=f"{prefix}_model_lab_binary_fit_eccentricity")
    elif st.session_state[f"{prefix}_model_lab_binary_kind"] == "empirical_eclipses":
        location.checkbox("Include secondary eclipse", value=st.session_state.get(f"{prefix}_model_lab_binary_secondary", True), key=f"{prefix}_model_lab_binary_secondary")
        cols = location.columns(2)
        cols[0].text_input("Primary phase guess", value=st.session_state.get(f"{prefix}_model_lab_binary_primary_phase", ""), key=f"{prefix}_model_lab_binary_primary_phase", placeholder="auto")
        cols[1].text_input("Secondary phase guess", value=st.session_state.get(f"{prefix}_model_lab_binary_secondary_phase", ""), key=f"{prefix}_model_lab_binary_secondary_phase", placeholder="auto")
    else:
        cols = location.columns(3)
        cols[0].number_input("Eccentricity", min_value=0.0, max_value=0.9, value=st.session_state.get(f"{prefix}_model_lab_binary_physical_eccentricity", 0.0), step=0.05, format="%.3f", key=f"{prefix}_model_lab_binary_physical_eccentricity")
        cols[1].number_input("Omega [deg]", min_value=0.0, max_value=360.0, value=st.session_state.get(f"{prefix}_model_lab_binary_omega", 90.0), step=5.0, format="%.1f", key=f"{prefix}_model_lab_binary_omega")
        cols[2].number_input("Inclination [deg]", min_value=0.0, max_value=90.0, value=st.session_state.get(f"{prefix}_model_lab_binary_inclination", 85.0), step=1.0, format="%.1f", key=f"{prefix}_model_lab_binary_inclination")
        spectral_options = ["Manual"] + list(BHL_DONOR_PRESETS.keys())
        class_options = ["V", "III", "I"]
        location.caption("Optional stellar presets provide typical pedagogical M, R, L, and Teff values.")
        cols = location.columns(2)
        cols[0].selectbox(
            "Primary spectral type",
            spectral_options,
            index=spectral_options.index(st.session_state.get(f"{prefix}_model_lab_binary_primary_spectral_type", "Manual"))
            if st.session_state.get(f"{prefix}_model_lab_binary_primary_spectral_type", "Manual") in spectral_options
            else 0,
            key=f"{prefix}_model_lab_binary_primary_spectral_type",
        )
        cols[1].selectbox(
            "Primary luminosity class",
            class_options,
            index=class_options.index(st.session_state.get(f"{prefix}_model_lab_binary_primary_luminosity_class", "V"))
            if st.session_state.get(f"{prefix}_model_lab_binary_primary_luminosity_class", "V") in class_options
            else 0,
            key=f"{prefix}_model_lab_binary_primary_luminosity_class",
        )
        cols = location.columns(2)
        cols[0].selectbox(
            "Secondary spectral type",
            spectral_options,
            index=spectral_options.index(st.session_state.get(f"{prefix}_model_lab_binary_secondary_spectral_type", "Manual"))
            if st.session_state.get(f"{prefix}_model_lab_binary_secondary_spectral_type", "Manual") in spectral_options
            else 0,
            key=f"{prefix}_model_lab_binary_secondary_spectral_type",
        )
        cols[1].selectbox(
            "Secondary luminosity class",
            class_options,
            index=class_options.index(st.session_state.get(f"{prefix}_model_lab_binary_secondary_luminosity_class", "V"))
            if st.session_state.get(f"{prefix}_model_lab_binary_secondary_luminosity_class", "V") in class_options
            else 0,
            key=f"{prefix}_model_lab_binary_secondary_luminosity_class",
        )
        primary_preset = bhl_donor_preset_values(
            st.session_state.get(f"{prefix}_model_lab_binary_primary_spectral_type", "Manual"),
            st.session_state.get(f"{prefix}_model_lab_binary_primary_luminosity_class", "V"),
        )
        secondary_preset = bhl_donor_preset_values(
            st.session_state.get(f"{prefix}_model_lab_binary_secondary_spectral_type", "Manual"),
            st.session_state.get(f"{prefix}_model_lab_binary_secondary_luminosity_class", "V"),
        )
        if primary_preset is not None:
            st.session_state[f"{prefix}_model_lab_binary_mass1"] = float(primary_preset["mass_msun"])
            st.session_state[f"{prefix}_model_lab_binary_radius1"] = float(primary_preset["radius_rsun"])
            st.session_state[f"{prefix}_model_lab_binary_temperature1"] = float(primary_preset["teff_k"])
            st.session_state[f"{prefix}_model_lab_binary_luminosity1"] = float(primary_preset["luminosity_lsun"])
        if secondary_preset is not None:
            st.session_state[f"{prefix}_model_lab_binary_mass2"] = float(secondary_preset["mass_msun"])
            st.session_state[f"{prefix}_model_lab_binary_radius2"] = float(secondary_preset["radius_rsun"])
            st.session_state[f"{prefix}_model_lab_binary_temperature2"] = float(secondary_preset["teff_k"])
            st.session_state[f"{prefix}_model_lab_binary_luminosity2"] = float(secondary_preset["luminosity_lsun"])
        active_presets = []
        if primary_preset is not None:
            active_presets.append(
                f"primary M={float(primary_preset['mass_msun']):.4g} Msun, R={float(primary_preset['radius_rsun']):.4g} Rsun, "
                f"L={float(primary_preset['luminosity_lsun']):.4g} Lsun, Teff={float(primary_preset['teff_k']):.5g} K"
            )
        if secondary_preset is not None:
            active_presets.append(
                f"secondary M={float(secondary_preset['mass_msun']):.4g} Msun, R={float(secondary_preset['radius_rsun']):.4g} Rsun, "
                f"L={float(secondary_preset['luminosity_lsun']):.4g} Lsun, Teff={float(secondary_preset['teff_k']):.5g} K"
            )
        if active_presets:
            location.caption("Loaded preset values: " + "; ".join(active_presets) + ". Switch the corresponding spectral type to Manual to edit freely.")
        cols = location.columns(2)
        cols[0].number_input("M1 [Msun]", min_value=0.05, max_value=150.0, value=st.session_state.get(f"{prefix}_model_lab_binary_mass1", 10.0), step=0.5, format="%.2f", key=f"{prefix}_model_lab_binary_mass1", disabled=primary_preset is not None)
        cols[1].number_input("M2 [Msun]", min_value=0.05, max_value=150.0, value=st.session_state.get(f"{prefix}_model_lab_binary_mass2", 1.4), step=0.1, format="%.2f", key=f"{prefix}_model_lab_binary_mass2", disabled=secondary_preset is not None)
        cols = location.columns(2)
        cols[0].number_input("R1 [Rsun]", min_value=0.01, max_value=2000.0, value=st.session_state.get(f"{prefix}_model_lab_binary_radius1", 6.0), step=0.5, format="%.2f", key=f"{prefix}_model_lab_binary_radius1", disabled=primary_preset is not None)
        cols[1].number_input("R2 [Rsun]", min_value=0.01, max_value=2000.0, value=st.session_state.get(f"{prefix}_model_lab_binary_radius2", 1.0), step=0.1, format="%.2f", key=f"{prefix}_model_lab_binary_radius2", disabled=secondary_preset is not None)
        cols = location.columns(2)
        cols[0].number_input("T1 [K]", min_value=1.0, max_value=100000.0, value=st.session_state.get(f"{prefix}_model_lab_binary_temperature1", 20000.0), step=500.0, format="%.0f", key=f"{prefix}_model_lab_binary_temperature1", disabled=primary_preset is not None)
        cols[1].number_input("T2 [K]", min_value=1.0, max_value=100000.0, value=st.session_state.get(f"{prefix}_model_lab_binary_temperature2", 8000.0), step=500.0, format="%.0f", key=f"{prefix}_model_lab_binary_temperature2", disabled=secondary_preset is not None)
        cols = location.columns(2)
        cols[0].number_input("L1 [Lsun]", min_value=1e-6, max_value=1e7, value=st.session_state.get(f"{prefix}_model_lab_binary_luminosity1", 10000.0), step=10.0, format="%.6g", key=f"{prefix}_model_lab_binary_luminosity1", disabled=primary_preset is not None)
        cols[1].number_input("L2 [Lsun]", min_value=1e-6, max_value=1e7, value=st.session_state.get(f"{prefix}_model_lab_binary_luminosity2", 10.0), step=1.0, format="%.6g", key=f"{prefix}_model_lab_binary_luminosity2", disabled=secondary_preset is not None)
        cols = location.columns(3)
        cols[0].number_input("u1", min_value=0.0, max_value=1.0, value=st.session_state.get(f"{prefix}_model_lab_binary_limb_u1", 0.4), step=0.05, format="%.2f", key=f"{prefix}_model_lab_binary_limb_u1")
        cols[1].number_input("u2", min_value=0.0, max_value=1.0, value=st.session_state.get(f"{prefix}_model_lab_binary_limb_u2", 0.4), step=0.05, format="%.2f", key=f"{prefix}_model_lab_binary_limb_u2")
        cols[2].number_input("Third light", min_value=0.0, max_value=20.0, value=st.session_state.get(f"{prefix}_model_lab_binary_third_light", 0.0), step=0.05, format="%.2f", key=f"{prefix}_model_lab_binary_third_light")
        cols = location.columns(2)
        cols[0].checkbox("Ellipsoidal", value=st.session_state.get(f"{prefix}_model_lab_binary_include_ellipsoidal", True), key=f"{prefix}_model_lab_binary_include_ellipsoidal")
        cols[1].number_input("Ellip. amp", min_value=-1.0, max_value=1.0, value=st.session_state.get(f"{prefix}_model_lab_binary_ellipsoidal_amp", 0.02), step=0.005, format="%.4f", key=f"{prefix}_model_lab_binary_ellipsoidal_amp")
        cols = location.columns(2)
        cols[0].checkbox("Reflection", value=st.session_state.get(f"{prefix}_model_lab_binary_include_reflection", True), key=f"{prefix}_model_lab_binary_include_reflection")
        cols[1].number_input("Refl. amp", min_value=-1.0, max_value=1.0, value=st.session_state.get(f"{prefix}_model_lab_binary_reflection_amp", 0.01), step=0.005, format="%.4f", key=f"{prefix}_model_lab_binary_reflection_amp")
        cols = location.columns(2)
        cols[0].checkbox("Beaming", value=st.session_state.get(f"{prefix}_model_lab_binary_include_beaming", False), key=f"{prefix}_model_lab_binary_include_beaming")
        cols[1].number_input("Beam amp", min_value=-1.0, max_value=1.0, value=st.session_state.get(f"{prefix}_model_lab_binary_beaming_amp", 0.0), step=0.001, format="%.4f", key=f"{prefix}_model_lab_binary_beaming_amp")
    location.checkbox(
        "Show full data set with model",
        value=st.session_state.get(f"{prefix}_model_lab_show_time_model", True),
        key=f"{prefix}_model_lab_show_time_model",
    )
    if location.button("Fit binary model", use_container_width=True, key=f"{prefix}_fit_binary_model"):
        fields = fields_from_state(prefix, bootstrap_override=0)
        fields.update({
            "model_lab_binary_period": str(st.session_state.get(f"{prefix}_model_lab_binary_period", "")).strip(),
            "model_lab_binary_t0": str(st.session_state.get(f"{prefix}_model_lab_binary_t0", "")).strip(),
            "model_lab_binary_bins": str(st.session_state.get(f"{prefix}_model_lab_binary_bins", 24)),
            "model_lab_binary_kind": st.session_state.get(f"{prefix}_model_lab_binary_kind", "eccentric_harmonic"),
            "model_lab_binary_harmonics": str(st.session_state.get(f"{prefix}_model_lab_binary_harmonics", 2)),
            "model_lab_binary_eccentricity": str(st.session_state.get(f"{prefix}_model_lab_binary_eccentricity", 0.2)),
            "model_lab_binary_fit_eccentricity": "true" if st.session_state.get(f"{prefix}_model_lab_binary_fit_eccentricity", True) else "false",
            "model_lab_binary_secondary": "true" if st.session_state.get(f"{prefix}_model_lab_binary_secondary", True) else "false",
            "model_lab_binary_primary_phase": str(st.session_state.get(f"{prefix}_model_lab_binary_primary_phase", "")).strip(),
            "model_lab_binary_secondary_phase": str(st.session_state.get(f"{prefix}_model_lab_binary_secondary_phase", "")).strip(),
            "model_lab_binary_physical_eccentricity": str(st.session_state.get(f"{prefix}_model_lab_binary_physical_eccentricity", 0.0)),
            "model_lab_binary_omega": str(st.session_state.get(f"{prefix}_model_lab_binary_omega", 90.0)),
            "model_lab_binary_inclination": str(st.session_state.get(f"{prefix}_model_lab_binary_inclination", 85.0)),
            "model_lab_binary_primary_spectral_type": st.session_state.get(f"{prefix}_model_lab_binary_primary_spectral_type", "Manual"),
            "model_lab_binary_primary_luminosity_class": st.session_state.get(f"{prefix}_model_lab_binary_primary_luminosity_class", "V"),
            "model_lab_binary_secondary_spectral_type": st.session_state.get(f"{prefix}_model_lab_binary_secondary_spectral_type", "Manual"),
            "model_lab_binary_secondary_luminosity_class": st.session_state.get(f"{prefix}_model_lab_binary_secondary_luminosity_class", "V"),
            "model_lab_binary_mass1": str(st.session_state.get(f"{prefix}_model_lab_binary_mass1", 10.0)),
            "model_lab_binary_mass2": str(st.session_state.get(f"{prefix}_model_lab_binary_mass2", 1.4)),
            "model_lab_binary_radius1": str(st.session_state.get(f"{prefix}_model_lab_binary_radius1", 6.0)),
            "model_lab_binary_radius2": str(st.session_state.get(f"{prefix}_model_lab_binary_radius2", 1.0)),
            "model_lab_binary_temperature1": str(st.session_state.get(f"{prefix}_model_lab_binary_temperature1", 20000.0)),
            "model_lab_binary_temperature2": str(st.session_state.get(f"{prefix}_model_lab_binary_temperature2", 8000.0)),
            "model_lab_binary_luminosity1": str(st.session_state.get(f"{prefix}_model_lab_binary_luminosity1", 10000.0)),
            "model_lab_binary_luminosity2": str(st.session_state.get(f"{prefix}_model_lab_binary_luminosity2", 10.0)),
            "model_lab_binary_limb_u1": str(st.session_state.get(f"{prefix}_model_lab_binary_limb_u1", 0.4)),
            "model_lab_binary_limb_u2": str(st.session_state.get(f"{prefix}_model_lab_binary_limb_u2", 0.4)),
            "model_lab_binary_third_light": str(st.session_state.get(f"{prefix}_model_lab_binary_third_light", 0.0)),
            "model_lab_binary_include_ellipsoidal": "true" if st.session_state.get(f"{prefix}_model_lab_binary_include_ellipsoidal", True) else "false",
            "model_lab_binary_ellipsoidal_amp": str(st.session_state.get(f"{prefix}_model_lab_binary_ellipsoidal_amp", 0.02)),
            "model_lab_binary_include_reflection": "true" if st.session_state.get(f"{prefix}_model_lab_binary_include_reflection", True) else "false",
            "model_lab_binary_reflection_amp": str(st.session_state.get(f"{prefix}_model_lab_binary_reflection_amp", 0.01)),
            "model_lab_binary_include_beaming": "true" if st.session_state.get(f"{prefix}_model_lab_binary_include_beaming", False) else "false",
            "model_lab_binary_beaming_amp": str(st.session_state.get(f"{prefix}_model_lab_binary_beaming_amp", 0.0)),
        })
        with st.spinner("Fitting binary model..."):
            try:
                st.session_state["app_model_lab_result"] = binary_model_lab(result, fields)
            except ValueError as exc:
                location.error(str(exc))
                return
        st.rerun()


def report_controls(prefix: str, location=st) -> None:
    result = st.session_state.get("app_result")
    location.text_input("Object name", key="report_object_name")
    location.text_input("Student name", key="report_student_name")
    location.text_input("Course", key="report_course")
    location.text_input(
        "Output PDF file name",
        key="report_output_name",
        value=st.session_state.get("report_output_name", "periodicity_workbench_report.pdf"),
    )
    previous_report_pdf = location.file_uploader(
        "Previous PDF report",
        type=["pdf"],
        key=f"{prefix}_previous_report_pdf",
        help="Optional. If provided, the current report is appended as additional pages.",
    )
    location.caption("Sections to include")
    section_cols = location.columns(2)
    section_cols[0].checkbox("Frequency search", value=st.session_state.get("report_include_frequency", True), key="report_include_frequency")
    section_cols[1].checkbox("Iterative prewhitening", value=st.session_state.get("report_include_prewhitening", True), key="report_include_prewhitening")
    section_cols = location.columns(2)
    section_cols[0].checkbox("Period tomography", value=st.session_state.get("report_include_tomography", True), key="report_include_tomography")
    section_cols[1].checkbox("Model laboratory", value=st.session_state.get("report_include_model_lab", True), key="report_include_model_lab")
    if not result:
        location.info("Run an analysis before generating a report.")
        return
    try:
        pdf_report = build_periodicity_pdf_report(result)
        pdf_download = pdf_report
        button_label = "Generate PDF report"
        if previous_report_pdf is not None:
            pdf_download = append_pdf_report(previous_report_pdf.getvalue(), pdf_report)
            button_label = "Generate appended PDF report"
        location.download_button(
            button_label,
            data=pdf_download,
            file_name=report_metadata()["filename"],
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            key=f"{prefix}_download_pdf_report",
        )
    except Exception as exc:
        location.error(f"Could not build the PDF report: {exc}")


def render_search_outputs(result: dict | None) -> None:
    if not result:
        st.info("Run an analysis to show plots.")
        return
    st.divider()
    st.subheader("Frequency analysis")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows used", f"{result['n_points']}")
    baseline_unit = result.get("baseline_unit", "d")
    period_unit = result.get("period_unit", "d")
    metric_cols[1].metric("Baseline", f"{result['baseline']:.4g} {baseline_unit}")
    primary = result.get("primary_period")
    metric_cols[2].metric("Primary period", "" if primary is None else f"{float(primary):.6g} {period_unit}")
    metric_cols[3].metric("T0", f"{float(result.get('t0', 0.0)):.4f}")
    cols = st.columns(3)
    cols[0].plotly_chart(periodogram(result, "power", "peaks", "Lomb-Scargle periodogram"), use_container_width=True)
    cols[1].plotly_chart(window_plot(result), use_container_width=True)
    with cols[2]:
        st.plotly_chart(folded_plot(result), use_container_width=True)
    folded_terms = result.get("fold_fit_terms", [])
    if folded_terms:
        st.caption("Folded fit equation")
        st.code(fit_equation_text(folded_terms, variable="phase"), language="text")
        fit_cols = st.columns([0.8, 1.2])
        with fit_cols[0]:
            st.caption("Global fit summary")
            st.dataframe(global_fit_summary(folded_terms), use_container_width=True, hide_index=True)
        with fit_cols[1]:
            st.caption("Folded periods and fit parameters")
            st.dataframe(folded_periods_and_parameters(result), use_container_width=True, hide_index=True)
    st.subheader("Detected peaks")
    st.dataframe(peaks_dataframe(result.get("peaks", [])), use_container_width=True, hide_index=True)
    harmonic_rows = result.get("harmonic_diagnostics", [])
    if harmonic_rows:
        with st.expander("Harmonic diagnostics", expanded=False):
            st.caption("Checks the fundamental, harmonics, and longer multiples of the selected primary period. The 2P row is useful for eclipsing-binary half-period ambiguity.")
            st.dataframe(harmonic_diagnostics_dataframe(harmonic_rows), use_container_width=True, hide_index=True)
    dl_cols = st.columns(3)
    with dl_cols[0]:
        dataframe_download("Download LS data", pd.DataFrame({"period": result["series"]["period"], "frequency": result["series"]["frequency"], "power": result["series"]["power"]}), "lomb_scargle_periodogram.txt", "app_download_ls")
    with dl_cols[1]:
        dataframe_download("Download window data", pd.DataFrame({"period": result["series"]["period"], "window_power": result["series"]["window_power"]}), "sampling_window.txt", "app_download_window")
    with dl_cols[2]:
        dataframe_download("Download folded data", pd.DataFrame({"phase": result["series"]["fold_phase"], "flux": result["series"]["fold_flux"], "error": result["series"]["fold_error"]}), "folded_profile.txt", "app_download_folded")
    render_frequency_evidence("l1", result)


def render_file_preview(prefix: str) -> None:
    if "app_file_bytes" not in st.session_state:
        return
    st.subheader("Uploaded data preview")
    try:
        file_bytes = st.session_state["app_file_bytes"]
        filename = st.session_state["app_filename"]
        render_fits_selector(prefix, file_bytes, filename)
        preview_cols = st.columns([1, 2])
        preview = file_preview_dataframe(file_bytes, filename, prefix)
        if not preview.empty:
            with preview_cols[0]:
                st.caption("File preview")
                st.dataframe(preview, use_container_width=True, hide_index=True, height=300)
        with preview_cols[1]:
            st.plotly_chart(
                raw_preview_figure(file_bytes, filename, prefix),
                use_container_width=True,
            )
    except ValueError as exc:
        st.error(str(exc))


def render_secondary_outputs(result: dict | None) -> None:
    if not result:
        return
    if result.get("has_prewhitening"):
        st.divider()
        st.subheader("Iterative prewhitening")
        summary = result.get("prewhitening_summary", {})
        if summary:
            metric_cols = st.columns(4)
            metric_cols[0].metric("RMS reduction", f"{float(summary.get('rms_reduction_percent', 0.0)):.3g}%")
            metric_cols[1].metric("LS peak reduction", f"{float(summary.get('max_power_reduction_percent', 0.0)):.3g}%")
            metric_cols[2].metric("AIC", f"{float(summary.get('AIC', 0.0)):.5g}")
            metric_cols[3].metric("BIC", f"{float(summary.get('BIC', 0.0)):.5g}")
        cols = st.columns([1.2, 1.0])
        cols[0].plotly_chart(periodogram(result, "residual_power", "residual_peaks", "After prewhitening"), use_container_width=True)
        with cols[1]:
            st.caption("Prewhitening steps")
            st.dataframe(clean_dataframe(pd.DataFrame(result.get("prewhitening_terms", []))), use_container_width=True, hide_index=True)
            if result.get("prewhitening_terms"):
                st.caption("Prewhitening fit equation")
                st.code(fit_equation_text(result.get("prewhitening_terms", []), variable="t_shifted"), language="text")
                st.caption("Global prewhitening fit summary")
                st.dataframe(global_fit_summary(result.get("prewhitening_terms", [])), use_container_width=True, hide_index=True)
                with st.expander("Prewhitening quality diagnostics", expanded=False):
                    st.dataframe(prewhitening_summary_dataframe(result.get("prewhitening_summary", {})), use_container_width=True, hide_index=True)
            st.caption("Remaining LS peaks after prewhitening")
            st.dataframe(peaks_dataframe(result.get("residual_peaks", [])), use_container_width=True, hide_index=True)
        render_prewhitening_evidence()
    if st.session_state.get("app_show_model") and result.get("has_prewhitening"):
        st.plotly_chart(
            prewhitening_model_plot(result, st.session_state.get("l1_show_model_errors", True)),
            use_container_width=True,
        )
    advanced = st.session_state.get("app_advanced_result")
    if advanced:
        st.divider()
        st.subheader("Period tomography")
        if advanced.get("message"):
            st.warning(str(advanced["message"]))
        st.plotly_chart(advanced_plot(advanced), use_container_width=True)
        render_tomography_evidence("l1")


def model_parameter_value(model_result: dict, name: str) -> float | None:
    for row in model_result.get("parameters", []):
        if row.get("parameter") == name:
            value = row.get("value")
            return None if value is None else float(value)
    return None


def render_binary_orbital_highlight(model_result: dict, app_result: dict) -> None:
    summary = model_result.get("summary", {})
    period_unit = app_result.get("period_unit", "")
    time_unit = app_result.get("baseline_unit", period_unit)
    model_kind = str(model_result.get("model_kind", "")).replace("_", " ")
    lines = [
        f"<strong>Period:</strong> {float(model_result.get('period', 0.0)):.6g} {period_unit}",
        f"<strong>T0 / reference epoch:</strong> {float(model_result.get('t0', 0.0)):.6g} {time_unit}",
        f"<strong>Model:</strong> {model_kind}",
    ]
    if model_result.get("model_kind") == "physical_eclipse_toy":
        lines.append(f"<strong>Eccentricity:</strong> {float(summary.get('eccentricity', 0.0)):.4g}")
        lines.append(f"<strong>Omega:</strong> {float(summary.get('omega_deg', 0.0)):.5g} deg")
        lines.append(f"<strong>Inclination:</strong> {float(summary.get('inclination_deg', 0.0)):.5g} deg")
        if summary.get("primary_spectral_type") != "manual" or summary.get("secondary_spectral_type") != "manual":
            lines.append(
                "<strong>Stellar presets:</strong> "
                f"primary {summary.get('primary_spectral_type', 'manual')} {summary.get('primary_luminosity_class', '')}, "
                f"secondary {summary.get('secondary_spectral_type', 'manual')} {summary.get('secondary_luminosity_class', '')}"
            )
        lines.append(
            "<strong>Masses / radii:</strong> "
            f"M1={float(summary.get('mass1_msun', 0.0)):.5g} Msun, M2={float(summary.get('mass2_msun', 0.0)):.5g} Msun, "
            f"R1={float(summary.get('radius1_rsun', 0.0)):.5g} Rsun, R2={float(summary.get('radius2_rsun', 0.0)):.5g} Rsun"
        )
        lines.append(f"<strong>Mass ratio q=M2/M1:</strong> {float(summary.get('mass_ratio_q_m2_over_m1', 0.0)):.5g}")
        lines.append(f"<strong>Semi-major axis:</strong> {float(summary.get('semi_major_axis_rsun', 0.0)):.5g} Rsun")
        lines.append(f"<strong>Radii:</strong> R1/a={float(summary.get('radius1_over_a', 0.0)):.5g}, R2/a={float(summary.get('radius2_over_a', 0.0)):.5g}")
        lines.append(f"<strong>Temperature ratio T2/T1:</strong> {float(summary.get('temperature_ratio_t2_over_t1', 0.0)):.5g}")
        lines.append(f"<strong>Luminosity ratio L2/L1:</strong> {float(summary.get('bolometric_luminosity_ratio_l2_over_l1', 0.0)):.5g}")
        lines.append(f"<strong>Surface brightness ratio S2/S1:</strong> {float(summary.get('surface_brightness_ratio_s2_over_s1', 0.0)):.5g}")
        lines.append(f"<strong>Third light fraction:</strong> {float(summary.get('third_light_fraction', 0.0)):.5g}")
        lines.append(f"<strong>Eclipse geometry:</strong> {summary.get('eclipse_possible', 'unknown')}")
        lines.append("<strong>Pedagogical hint:</strong> masses set the orbital scale through Kepler's law, radii and inclination set whether eclipses occur, and luminosities, surface brightness, limb darkening, and third light dilute or deepen the eclipses.")
    elif model_result.get("model_kind") == "eccentric_harmonic":
        eccentricity = summary.get("eccentricity")
        if eccentricity is not None:
            lines.append(f"<strong>Eccentricity:</strong> {float(eccentricity):.4g}")
        lines.append("<strong>Inclination:</strong> not constrained by this empirical fit")
        lines.append("<strong>Mass ratio:</strong> not constrained by this empirical fit")
    else:
        primary_phase = model_parameter_value(model_result, "primary_phase")
        secondary_phase = model_parameter_value(model_result, "secondary_phase")
        primary_depth = model_parameter_value(model_result, "primary_depth")
        secondary_depth = model_parameter_value(model_result, "secondary_depth")
        primary_width = model_parameter_value(model_result, "primary_width_phase")
        secondary_width = model_parameter_value(model_result, "secondary_width_phase")
        if primary_phase is not None:
            lines.append(f"<strong>Primary eclipse phase:</strong> {primary_phase:.5g}")
        if secondary_phase is not None:
            separation = (secondary_phase - primary_phase) % 1.0 if primary_phase is not None else None
            lines.append(f"<strong>Secondary eclipse phase:</strong> {secondary_phase:.5g}")
            if separation is not None:
                lines.append(f"<strong>Eclipse separation:</strong> {separation:.5g} in phase")
                lines.append(f"<strong>Offset from 0.5 separation:</strong> {separation - 0.5:+.5g} in phase")
        if primary_depth is not None:
            lines.append(f"<strong>Primary depth/sign:</strong> {primary_depth:.5g}")
        if secondary_depth is not None:
            lines.append(f"<strong>Secondary depth/sign:</strong> {secondary_depth:.5g}")
        if primary_depth is not None and abs(primary_depth) > 0.0 and secondary_depth is not None:
            lines.append(f"<strong>Depth ratio |secondary/primary|:</strong> {abs(secondary_depth / primary_depth):.5g}")
        if primary_width is not None and abs(primary_width) > 0.0 and secondary_width is not None:
            lines.append(f"<strong>Width ratio secondary/primary:</strong> {secondary_width / primary_width:.5g}")
        lines.append("<strong>Pedagogical hint:</strong> depth and width ratios trace relative brightness/radius geometry, while separation away from 0.5 can suggest eccentric geometry or phase-reference effects.")
        lines.append("<strong>Eccentricity:</strong> not directly constrained by the empirical eclipse model")
        lines.append("<strong>Inclination / mass ratio:</strong> require a physical binary model such as ellc or external constraints")
    if summary.get("BIC") is not None:
        lines.append(f"<strong>BIC:</strong> {float(summary['BIC']):.5g}")
    if summary.get("rms") is not None:
        lines.append(f"<strong>RMS:</strong> {float(summary['rms']):.5g}")
    st.markdown(
        """
        <div style="border: 1px solid #b7c7e6; background: #f3f7ff; padding: 0.85rem 1rem; border-radius: 0.45rem; margin: 0.35rem 0 1rem 0;">
          <div style="font-weight: 800; font-size: 1.02rem; margin-bottom: 0.35rem;">Basic orbital information from this fit</div>
          <div style="line-height: 1.65;">""" + "<br>".join(lines) + """</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bondi_hoyle_highlight(model_result: dict, app_result: dict) -> None:
    summary = model_result.get("summary", {})
    period_unit = app_result.get("period_unit", "")
    time_unit = app_result.get("baseline_unit", period_unit)
    maxima = model_result.get("extrema", [])
    lines = [
        f"<strong>Orbital period:</strong> {float(model_result.get('period', 0.0)):.6g} {period_unit}",
        f"<strong>T0 / periastron epoch:</strong> {float(model_result.get('t0', 0.0)):.6g} {time_unit}",
        f"<strong>Eccentricity:</strong> {float(summary.get('eccentricity', 0.0)):.4g}",
        f"<strong>Effective v_wind / v_orb:</strong> {float(summary.get('wind_speed_ratio', 0.0)):.4g}",
        f"<strong>Wind beta:</strong> {float(summary.get('wind_beta', 0.0)):.4g}",
        f"<strong>Wind launch radius R_donor/a:</strong> {float(summary.get('donor_radius_over_a', 0.0)):.5g}",
        f"<strong>Phase lag:</strong> {float(summary.get('phase_lag', 0.0)):+.5g}",
    ]
    if summary.get("donor_spectral_type"):
        lines.append(
            f"<strong>Donor preset:</strong> {summary.get('donor_spectral_type')} {summary.get('donor_luminosity_class', '')}"
        )
    if summary.get("wind_input_mode") == "v_inf":
        lines.extend([
            f"<strong>v_inf:</strong> {float(summary.get('v_inf_km_s', 0.0)):.5g} km/s",
            f"<strong>Masses:</strong> M_donor={float(summary.get('donor_mass_msun', 0.0)):.5g} Msun, M_compact={float(summary.get('compact_mass_msun', 0.0)):.5g} Msun",
            f"<strong>Donor radius:</strong> {float(summary.get('donor_radius_rsun', 0.0)):.5g} Rsun",
            f"<strong>Semi-major axis:</strong> {float(summary.get('semi_major_axis_rsun', 0.0)):.5g} Rsun",
            f"<strong>Orbital speed scale:</strong> {float(summary.get('orbital_speed_km_s', 0.0)):.5g} km/s",
        ])
        if summary.get("radius_periastron_fraction") is not None:
            lines.append(f"<strong>R_donor / r_periastron:</strong> {float(summary.get('radius_periastron_fraction', 0.0)):.5g}")
    if maxima:
        lines.append(f"<strong>Model accretion maximum phase:</strong> {float(maxima[0].get('phase', 0.0)):.5g}")
    if summary.get("scale") is not None:
        lines.append(f"<strong>Accretion scale/sign:</strong> {float(summary['scale']):.5g}")
    if summary.get("BIC") is not None:
        lines.append(f"<strong>BIC:</strong> {float(summary['BIC']):.5g}")
    if summary.get("rms") is not None:
        lines.append(f"<strong>RMS:</strong> {float(summary['rms']):.5g}")
    lines.append("<strong>Pedagogical hint:</strong> this toy model tests whether denser/slower wind near periastron can explain the modulation; fitted parameters are phenomenological unless masses, wind law, and geometry are externally constrained.")
    st.markdown(
        """
        <div style="border: 1px solid #d4be8a; background: #fff8e8; padding: 0.85rem 1rem; border-radius: 0.45rem; margin: 0.35rem 0 1rem 0;">
          <div style="font-weight: 800; font-size: 1.02rem; margin-bottom: 0.35rem;">Bondi-Hoyle toy accretion summary</div>
          <div style="line-height: 1.65;">""" + "<br>".join(lines) + """</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pulse_highlight(model_result: dict, app_result: dict) -> None:
    summary = model_result.get("summary", {})
    period_unit = app_result.get("period_unit", "")
    time_unit = app_result.get("baseline_unit", period_unit)
    maxima = model_result.get("maxima", [])
    epoch_stat = summary.get("epoch_folding_statistic")
    lines = [
        f"<strong>Best pulse period:</strong> {float(model_result.get('period', 0.0)):.8g} {period_unit}",
        f"<strong>Input period guess:</strong> {float(model_result.get('period_guess', 0.0)):.8g} {period_unit}",
        f"<strong>T0:</strong> {float(model_result.get('t0', 0.0)):.6g} {time_unit}",
        f"<strong>Epoch-folding statistic:</strong> {float(epoch_stat):.5g}" if epoch_stat is not None else "<strong>Epoch-folding statistic:</strong> unavailable",
        f"<strong>Pulse arrivals used:</strong> {int(summary.get('n_arrivals', 0))}",
    ]
    if summary.get("peak_to_peak_pulsed_fraction") is not None:
        lines.append(f"<strong>Peak-to-peak pulsed fraction:</strong> {float(summary['peak_to_peak_pulsed_fraction']):.5g}")
    if summary.get("rms_pulsed_fraction") is not None:
        lines.append(f"<strong>RMS pulsed fraction:</strong> {float(summary['rms_pulsed_fraction']):.5g}")
    if maxima:
        lines.append(f"<strong>Main pulse maximum phase:</strong> {float(maxima[0].get('phase', 0.0)):.5g}")
    oc_summary = model_result.get("oc_summary", {})
    if oc_summary:
        lines.append(f"<strong>O-C sinusoid period:</strong> {float(oc_summary.get('orbital_period', 0.0)):.6g} {period_unit}")
        lines.append(f"<strong>Projected a sin i proxy:</strong> {float(oc_summary.get('oc_amplitude_time', 0.0)):.5g} {time_unit}")
    lines.append("<strong>Pedagogical hint:</strong> epoch folding refines the pulse period, the profile fit estimates pulse shape and pulsed fraction, and O-C trends can flag Doppler delays before attempting a physical orbital timing solution.")
    st.markdown(
        """
        <div style="border: 1px solid #b8bfd4; background: #f6f7fb; padding: 0.85rem 1rem; border-radius: 0.45rem; margin: 0.35rem 0 1rem 0;">
          <div style="font-weight: 800; font-size: 1.02rem; margin-bottom: 0.35rem;">Pulse timing summary</div>
          <div style="line-height: 1.65;">""" + "<br>".join(lines) + """</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_lab_outputs(result: dict | None) -> None:
    model_result = st.session_state.get("app_model_lab_result")
    if not result or not model_result:
        return
    st.divider()
    st.subheader("Model laboratory")
    if model_result.get("family") == "fourier":
        st.plotly_chart(model_lab_fourier_plot(model_result, result), use_container_width=True)
        if st.session_state.get("l1_model_lab_show_time_model", True):
            st.plotly_chart(
                model_lab_time_plot(model_result, result, st.session_state.get("l1_show_model_errors", True)),
                use_container_width=True,
            )
        info_cols = st.columns(4)
        info_cols[0].metric("Model period", f"{float(model_result['period']):.6g} {result.get('period_unit', '')}")
        info_cols[1].metric("Harmonics", f"{int(model_result['selected_harmonics'])}")
        info_cols[2].metric("Selection", str(model_result.get("selection", "manual")).upper())
        info_cols[3].metric("RMS", f"{float(model_result.get('summary', {}).get('rms', 0.0)):.5g}")
        cols = st.columns([1.05, 1.0])
        with cols[0]:
            st.caption("Fourier model equation")
            st.code(fit_equation_text(model_result.get("terms", []), variable="phase"), language="text")
            st.caption("Fit terms")
            st.dataframe(clean_dataframe(pd.DataFrame(model_result.get("terms", []))), use_container_width=True, hide_index=True)
        with cols[1]:
            st.caption("Global Fourier fit summary")
            st.dataframe(clean_dataframe(pd.DataFrame([model_result.get("summary", {})])), use_container_width=True, hide_index=True)
            st.caption("Harmonic-order comparison")
            st.dataframe(clean_dataframe(pd.DataFrame(model_result.get("trials", []))), use_container_width=True, hide_index=True)
            st.caption("Model maxima")
            st.dataframe(clean_dataframe(pd.DataFrame(model_result.get("maxima", []))), use_container_width=True, hide_index=True)
        prefix_name = "fourier"
    elif model_result.get("family") == "binary":
        st.plotly_chart(model_lab_binary_plot(model_result, result), use_container_width=True)
        if st.session_state.get("l1_model_lab_show_time_model", True):
            st.plotly_chart(
                model_lab_time_plot(model_result, result, st.session_state.get("l1_show_model_errors", True)),
                use_container_width=True,
            )
        info_cols = st.columns(4)
        info_cols[0].metric("Model period", f"{float(model_result['period']):.6g} {result.get('period_unit', '')}")
        info_cols[1].metric("Model", str(model_result.get("model_kind", "")).replace("_", " "))
        summary = model_result.get("summary", {})
        info_cols[2].metric("BIC", f"{float(summary.get('BIC', 0.0)):.5g}")
        info_cols[3].metric("RMS", f"{float(summary.get('rms', 0.0)):.5g}")
        render_binary_orbital_highlight(model_result, result)
        cols = st.columns([1.05, 1.0])
        with cols[0]:
            st.caption("Binary model formula")
            st.code(str(model_result.get("formula", "")), language="text")
            st.caption("Fit parameters")
            st.dataframe(clean_dataframe(pd.DataFrame(model_result.get("parameters", []))), use_container_width=True, hide_index=True)
        with cols[1]:
            st.caption("Global binary fit summary")
            st.dataframe(clean_dataframe(pd.DataFrame([summary])), use_container_width=True, hide_index=True)
            st.caption("Model extrema / eclipse phases")
            st.dataframe(clean_dataframe(pd.DataFrame(model_result.get("extrema", []))), use_container_width=True, hide_index=True)
        prefix_name = "binary"
    elif model_result.get("family") == "binary_assistant":
        st.plotly_chart(model_lab_binary_assistant_plot(model_result, result), use_container_width=True)
        summary = model_result.get("summary", {})
        info_cols = st.columns(4)
        period_value = model_result.get("period")
        period_label = "" if period_value is None else f"{float(period_value):.6g} {result.get('period_unit', '')}"
        info_cols[0].metric("Folded period", period_label)
        info_cols[1].metric("Morphology", str(summary.get("morphology", "not available")))
        info_cols[2].metric("Min. separation", f"{float(summary.get('minimum_phase_separation', 0.0)):.4g}")
        ratio = summary.get("depth_ratio")
        info_cols[3].metric("Depth ratio", "" if ratio is None else f"{float(ratio):.4g}")
        st.caption("Binary interpretation diagnostics")
        st.dataframe(clean_dataframe(pd.DataFrame(model_result.get("parameters", []))), use_container_width=True, hide_index=True)
        st.caption("Pedagogical hints")
        for hint in model_result.get("hints", []):
            st.markdown(f"- {hint}")
        prefix_name = "binary_assistant"
    elif model_result.get("family") == "bondi_hoyle":
        st.plotly_chart(model_lab_bondi_hoyle_plot(model_result, result), use_container_width=True)
        if st.session_state.get("l1_model_lab_show_time_model", True):
            st.plotly_chart(
                model_lab_time_plot(model_result, result, st.session_state.get("l1_show_model_errors", True)),
                use_container_width=True,
            )
        summary = model_result.get("summary", {})
        info_cols = st.columns(4)
        info_cols[0].metric("Model period", f"{float(model_result['period']):.6g} {result.get('period_unit', '')}")
        info_cols[1].metric("Eccentricity", f"{float(summary.get('eccentricity', 0.0)):.4g}")
        info_cols[2].metric("BIC", f"{float(summary.get('BIC', 0.0)):.5g}")
        info_cols[3].metric("RMS", f"{float(summary.get('rms', 0.0)):.5g}")
        render_bondi_hoyle_highlight(model_result, result)
        cols = st.columns([1.05, 1.0])
        with cols[0]:
            st.caption("Bondi-Hoyle toy model formula")
            st.code(str(model_result.get("formula", "")), language="text")
            st.caption("Fit parameters")
            st.dataframe(clean_dataframe(pd.DataFrame(model_result.get("parameters", []))), use_container_width=True, hide_index=True)
        with cols[1]:
            st.caption("Global Bondi-Hoyle fit summary")
            st.dataframe(clean_dataframe(pd.DataFrame([summary])), use_container_width=True, hide_index=True)
            st.caption("Accretion maxima")
            st.dataframe(clean_dataframe(pd.DataFrame(model_result.get("extrema", []))), use_container_width=True, hide_index=True)
        prefix_name = "bondi_hoyle"
    elif model_result.get("family") == "pulse":
        st.plotly_chart(model_lab_pulse_profile_plot(model_result, result), use_container_width=True)
        if st.session_state.get("l1_model_lab_show_time_model", True):
            st.plotly_chart(
                model_lab_time_plot(model_result, result, st.session_state.get("l1_show_model_errors", True)),
                use_container_width=True,
            )
        cols_plots = st.columns(2)
        with cols_plots[0]:
            st.plotly_chart(model_lab_pulse_epoch_plot(model_result, result), use_container_width=True)
        with cols_plots[1]:
            st.plotly_chart(model_lab_pulse_oc_plot(model_result, result), use_container_width=True)
        summary = model_result.get("summary", {})
        info_cols = st.columns(4)
        info_cols[0].metric("Pulse period", f"{float(model_result['period']):.8g} {result.get('period_unit', '')}")
        info_cols[1].metric("Pulsed fraction", "" if summary.get("peak_to_peak_pulsed_fraction") is None else f"{float(summary['peak_to_peak_pulsed_fraction']):.5g}")
        info_cols[2].metric("Arrivals", f"{int(summary.get('n_arrivals', 0))}")
        info_cols[3].metric("RMS", f"{float(summary.get('rms', 0.0)):.5g}")
        render_pulse_highlight(model_result, result)
        cols = st.columns([1.05, 1.0])
        with cols[0]:
            st.caption("Pulse-shape formula")
            st.code(str(model_result.get("formula", "")), language="text")
            st.caption("Pulse-shape terms")
            st.dataframe(clean_dataframe(pd.DataFrame(model_result.get("terms", []))), use_container_width=True, hide_index=True)
            if model_result.get("oc_formula"):
                st.caption("Pulse-arrival O-C formula")
                st.code(str(model_result.get("oc_formula", "")), language="text")
                st.caption("O-C sinusoid parameters")
                st.dataframe(clean_dataframe(pd.DataFrame(model_result.get("oc_parameters", []))), use_container_width=True, hide_index=True)
        with cols[1]:
            st.caption("Global pulse fit summary")
            st.dataframe(clean_dataframe(pd.DataFrame([summary])), use_container_width=True, hide_index=True)
            st.caption("Pulse maxima")
            st.dataframe(clean_dataframe(pd.DataFrame(model_result.get("maxima", []))), use_container_width=True, hide_index=True)
            st.caption("Pulse arrival times")
            st.dataframe(clean_dataframe(pd.DataFrame(model_result.get("arrivals", []))), use_container_width=True, hide_index=True)
        dl_extra_cols = st.columns(2)
        with dl_extra_cols[0]:
            dataframe_download(
                "Download epoch-folding search",
                pd.DataFrame({
                    "trial_period": model_result.get("trial_period", []),
                    "epoch_folding_statistic": model_result.get("epoch_power", []),
                }),
                "pulse_epoch_folding_search.txt",
                "app_download_pulse_epoch_search",
            )
        with dl_extra_cols[1]:
            dataframe_download(
                "Download pulse arrivals",
                pd.DataFrame(model_result.get("arrivals", [])),
                "pulse_arrival_times.txt",
                "app_download_pulse_arrivals",
            )
        prefix_name = "pulse"
    else:
        return
    render_model_lab_evidence(model_result)
    dl_cols = st.columns(3)
    with dl_cols[0]:
        dataframe_download(
            "Download model folded data",
            parallel_series_dataframe({
                "phase": model_result.get("phase", []),
                "flux": model_result.get("flux", []),
                "error": model_result.get("error", []),
                "counts": model_result.get("counts", []),
            }),
            f"{prefix_name}_folded_profile.txt",
            f"app_download_{prefix_name}_folded",
        )
    with dl_cols[1]:
        dataframe_download(
            "Download phase model",
            parallel_series_dataframe({
                "phase": model_result.get("model_phase", []),
                "model_flux": model_result.get("model_flux", []),
            }),
            f"{prefix_name}_phase_model.txt",
            f"app_download_{prefix_name}_phase_model",
        )
    with dl_cols[2]:
        dataframe_download(
            "Download time-domain model",
            parallel_series_dataframe({
                "time": model_result.get("model_time", []),
                "model_flux": model_result.get("model_time_flux", []),
            }),
            f"{prefix_name}_time_model.txt",
            f"app_download_{prefix_name}_time_model",
        )


def layout_one() -> None:
    prefix = "l1"
    st.title("Periodicity Workbench")
    st.caption(f"Version {APP_VERSION}")
    with st.sidebar:
        with st.expander("Input", expanded=True):
            input_controls(prefix, st)
        with st.expander("Frequency Search", expanded=True):
            search_controls(prefix, st)
            st.divider()
            st.markdown("**Uncertainties**")
            uncertainty_controls(prefix, st)
            st.divider()
            st.markdown("**Manual Exclusions**")
            manual_exclusion_controls(prefix, st)
        with st.expander("Folded Profile", expanded=False):
            folded_controls(prefix, st)
        with st.expander("Iterative Prewhitening", expanded=False):
            prewhitening_controls(prefix, st)
        with st.expander("Period Tomography", expanded=False):
            advanced_controls(prefix, st)
        with st.expander("Model Laboratory", expanded=False):
            model_lab_controls(prefix, st)
        with st.expander("Generate report", expanded=False):
            report_controls(prefix, st)
        if st.button("Help", use_container_width=True, key=f"{prefix}_show_help"):
            st.session_state["app_show_help"] = True
            st.session_state["app_show_bibliography"] = False
            st.rerun()
    result = st.session_state.get("app_result")
    if st.session_state.get("app_show_help", False):
        render_help_document()
        return
    if st.session_state.get("app_show_bibliography", False):
        render_bibliography_search(prefix)
        return
    render_file_preview(prefix)
    render_search_outputs(result)
    render_secondary_outputs(result)
    render_model_lab_outputs(result)

st.sidebar.header("Analysis controls")

if "app_prewhitening_periods" not in st.session_state:
    st.session_state["app_prewhitening_periods"] = []

layout_one()
