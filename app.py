#!/usr/bin/env python3
from __future__ import annotations

import base64
import gzip
import io
import json
import os
import re
from dataclasses import dataclass, asdict
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))

import matplotlib

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.timeseries import LombScargle
from scipy.optimize import least_squares, minimize_scalar
from scipy.signal import find_peaks


@dataclass
class PeakSummary:
    label: str
    frequency: float
    period: float
    power: float
    fap: float
    kind: str
    window_frequency: float | None = None
    window_period: float | None = None
    window_power: float | None = None
    frequency_error: float | None = None
    frequency_p16: float | None = None
    frequency_p84: float | None = None
    period_error: float | None = None
    period_p16: float | None = None
    period_p84: float | None = None


ALLOWED_UPLOAD_EXTENSIONS = {"", ".txt", ".dat", ".fits", ".fit", ".fts", ".ftz", ".lc", ".gz"}
FITS_UPLOAD_EXTENSIONS = (".fits", ".fit", ".fts", ".ftz", ".fits.gz")
ALLOWED_ASCII_CONTROLS = {9, 10, 12, 13}
SUSPICIOUS_SHELL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"`",
        r"\$\(",
        r"\b(?:bash|sh|zsh|python|python3|perl|ruby|osascript|sudo|curl|wget)\b",
        r"\b(?:rm|mv|cp|chmod|chown|open|exec|eval)\s+",
        r"(?:^|\s)(?:&&|\|\||;|>|<)(?:\s|$)",
    ]
]
NUMERIC_VALUE_TOKEN = re.compile(r"^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+\-]?\d+)?$")
MISSING_VALUE_TOKENS = {"nan", "null", "none", "na", "n/a", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
COMMENT_PREFIXES = ("#", "%")
G_SI = 6.67430e-11
M_SUN_SI = 1.98847e30
R_SUN_SI = 6.957e8
DAY_SI = 86400.0


def is_numeric_table_row(line: str) -> bool:
    tokens = line.replace(",", " ").split()
    if not tokens:
        return False
    for token in tokens:
        lower_token = token.lower()
        if lower_token in MISSING_VALUE_TOKENS:
            continue
        if NUMERIC_VALUE_TOKEN.match(token):
            continue
        return False
    return True


def is_fits_upload(filename: str, raw: bytes | None = None) -> bool:
    lower_name = (filename or "").lower()
    if raw:
        if raw.startswith(b"SIMPLE  "):
            return True
        if raw.startswith(b"\x1f\x8b"):
            try:
                return gzip.decompress(raw[:4096]).startswith(b"SIMPLE  ")
            except (OSError, EOFError):
                try:
                    return fits_payload(raw).startswith(b"SIMPLE  ")
                except ValueError:
                    return False
    return lower_name.endswith(FITS_UPLOAD_EXTENSIONS)


def fits_payload(raw: bytes) -> bytes:
    if raw.startswith(b"\x1f\x8b"):
        try:
            return gzip.decompress(raw)
        except OSError as exc:
            raise ValueError("Compressed FITS upload could not be decompressed") from exc
    return raw


def validate_upload(filename: str, raw: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if is_fits_upload(filename, raw):
        try:
            with fits.open(io.BytesIO(fits_payload(raw)), memmap=False):
                pass
        except Exception as exc:
            raise ValueError("Uploaded FITS file could not be opened") from exc
        return ""
    if suffix == ".gz" and not (filename or "").lower().endswith(".fits.gz"):
        raise ValueError("Only .fits.gz compressed FITS files are supported")
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ".txt, .dat, .lc, FITS files by content, or no extension"
        raise ValueError(f"Unsupported file extension '{suffix or '(none)'}'; expected {allowed}")
    if not raw:
        raise ValueError("Uploaded file is empty")

    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Uploaded file must be plain ASCII text") from exc

    for byte in raw:
        if byte < 32 and byte not in ALLOWED_ASCII_CONTROLS:
            raise ValueError("Uploaded file contains non-text control characters")

    for pattern in SUSPICIOUS_SHELL_PATTERNS:
        if pattern.search(text):
            raise ValueError("Uploaded file contains command-like text and was rejected")

    numeric_rows = 0
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith(COMMENT_PREFIXES):
            continue
        if not is_numeric_table_row(stripped):
            raise ValueError(
                f"Line {line_number} is not a numeric table row or a comment. "
                "Use '#' or '%' for comments and numeric columns for data; NULL/NaN values are allowed."
            )
        numeric_rows += 1
    if numeric_rows < 10:
        raise ValueError("Uploaded file must contain at least 10 numeric data rows")
    return text


def fits_table_metadata(raw: bytes) -> list[dict]:
    rows: list[dict] = []
    try:
        hdul = fits.open(io.BytesIO(fits_payload(raw)), memmap=False)
    except Exception as exc:
        raise ValueError("Uploaded FITS file could not be opened") from exc
    with hdul:
        for idx, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            columns = getattr(hdu, "columns", None)
            if data is None or columns is None or not hasattr(columns, "names"):
                continue
            numeric_columns = []
            for column in columns:
                name = column.name
                try:
                    values = np.asarray(data[name])
                except Exception:
                    continue
                if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
                    continue
                numeric_columns.append(
                    {
                        "name": str(name),
                        "format": str(getattr(column, "format", "")),
                        "unit": "" if getattr(column, "unit", None) is None else str(column.unit),
                    }
                )
            if numeric_columns:
                extname = str(hdu.header.get("EXTNAME", "")).strip()
                label = f"{idx}: {extname}" if extname else f"{idx}: {hdu.__class__.__name__}"
                rows.append(
                    {
                        "index": idx,
                        "name": extname or hdu.__class__.__name__,
                        "label": label,
                        "n_rows": int(len(data)),
                        "columns": numeric_columns,
                    }
                )
    if not rows:
        raise ValueError("FITS file does not contain a table extension with scalar numeric columns")
    return rows


def read_fits_columns(
    raw: bytes,
    extension: int,
    time_column: str,
    flux_column: str,
    error_column: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        hdul = fits.open(io.BytesIO(fits_payload(raw)), memmap=False)
    except Exception as exc:
        raise ValueError("Uploaded FITS file could not be opened") from exc
    with hdul:
        if extension < 0 or extension >= len(hdul):
            raise ValueError(f"FITS extension {extension} is outside the available range")
        data = getattr(hdul[extension], "data", None)
        if data is None:
            raise ValueError(f"FITS extension {extension} does not contain table data")
        names = set(data.names or [])
        required = [time_column, flux_column]
        if error_column:
            required.append(error_column)
        missing = [name for name in required if name not in names]
        if missing:
            raise ValueError("FITS column not found: " + ", ".join(missing))
        t = np.asarray(data[time_column], dtype=float)
        y = np.asarray(data[flux_column], dtype=float)
        if t.ndim != 1 or y.ndim != 1:
            raise ValueError("FITS time and flux columns must be scalar one-dimensional columns")
        if error_column:
            dy = np.asarray(data[error_column], dtype=float)
            if dy.ndim != 1:
                raise ValueError("FITS error column must be a scalar one-dimensional column")
            good = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
        else:
            dy = np.ones_like(y, dtype=float)
            good = np.isfinite(t) & np.isfinite(y)
        if good.sum() < 10:
            raise ValueError("Need at least 10 valid FITS rows after filtering finite values")
        t, y, dy = t[good], y[good], dy[good]
        order = np.argsort(t)
        return t[order], y[order], dy[order]


def read_columns(
    raw: bytes,
    filename: str,
    time_col: int,
    flux_col: int,
    error_col: int | None,
    fits_extension: int | None = None,
    fits_time_col: str | None = None,
    fits_flux_col: str | None = None,
    fits_error_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if is_fits_upload(filename, raw):
        if fits_extension is None or not fits_time_col or not fits_flux_col:
            raise ValueError("Choose a FITS table extension and time/flux columns before running the analysis")
        return read_fits_columns(raw, int(fits_extension), fits_time_col, fits_flux_col, fits_error_col)
    text = validate_upload(filename, raw)
    numeric_text = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith(COMMENT_PREFIXES)
    )
    data = np.genfromtxt(io.StringIO(numeric_text.replace(",", " ")), invalid_raise=False)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError("Could not parse the uploaded text file as a numeric table")
    max_col = data.shape[1] - 1
    required_cols = [time_col, flux_col]
    if error_col is not None:
        required_cols.append(error_col)
    for col in required_cols:
        if col < 0 or col > max_col:
            raise ValueError(f"Column index {col + 1} is outside the available range 1-{data.shape[1]}")
    t, y = data[:, time_col], data[:, flux_col]
    if error_col is None:
        dy = np.ones_like(y, dtype=float)
        good = np.isfinite(t) & np.isfinite(y)
    else:
        dy = data[:, error_col]
        good = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
    if good.sum() < 10:
        if error_col is None:
            raise ValueError("Need at least 10 valid rows with finite time and flux values")
        raise ValueError("Need at least 10 valid rows with finite values and positive errors")
    t, y, dy = t[good], y[good], dy[good]
    order = np.argsort(t)
    return t[order], y[order], dy[order]


def optional_float(value: str | None) -> float | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    return float(cleaned)


def normalize_time_unit(time_unit: str) -> str:
    normalized = (time_unit or "days").strip().lower()
    if normalized in {"s", "sec", "secs", "second", "seconds"}:
        return "seconds"
    if normalized in {"d", "day", "days"}:
        return "days"
    raise ValueError("Time units must be days or seconds")


def unit_labels(time_unit: str) -> dict[str, str]:
    normalized = normalize_time_unit(time_unit)
    if normalized == "seconds":
        return {
            "time_unit": "seconds",
            "time_label": "Time [s]",
            "period_label": "Period [s]",
            "frequency_label": "Frequency [Hz]",
            "baseline_unit": "s",
            "period_unit": "s",
            "frequency_unit": "Hz",
        }
    return {
        "time_unit": "days",
        "time_label": "Time [d]",
        "period_label": "Period [d]",
        "frequency_label": "Frequency [cycles/day]",
        "baseline_unit": "d",
        "period_unit": "d",
        "frequency_unit": "cycles/day",
    }


def apply_data_limits(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    fields: dict[str, str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin = optional_float(fields.get("xmin"))
    xmax = optional_float(fields.get("xmax"))
    ymin = optional_float(fields.get("ymin"))
    ymax = optional_float(fields.get("ymax"))
    good = np.ones_like(t, dtype=bool)
    if xmin is not None:
        good &= t >= xmin
    if xmax is not None:
        good &= t <= xmax
    if ymin is not None:
        good &= y >= ymin
    if ymax is not None:
        good &= y <= ymax
    if good.sum() < 10:
        raise ValueError("Need at least 10 valid rows after applying the selected x/y limits")
    return t[good], y[good], dy[good]


def frequency_grid(t: np.ndarray, fmin: float, fmax: float, samples_per_peak: float) -> np.ndarray:
    baseline = float(t.max() - t.min())
    if baseline <= 0:
        raise ValueError("Time baseline must be positive")
    n_freq = int(np.ceil((fmax - fmin) * baseline * samples_per_peak))
    return np.linspace(fmin, fmax, max(n_freq, 1000))


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * sorted_weights.sum()
    return float(sorted_values[np.searchsorted(np.cumsum(sorted_weights), cutoff)])


def spectral_window(t: np.ndarray, frequency: np.ndarray, chunk_size: int = 500) -> np.ndarray:
    shifted = t - t.min()
    out = np.empty_like(frequency)
    n = len(shifted)
    for start in range(0, len(frequency), chunk_size):
        freq = frequency[start:start + chunk_size]
        z = np.exp(2j * np.pi * freq[:, None] * shifted[None, :]).sum(axis=1)
        out[start:start + chunk_size] = (np.abs(z) / n) ** 2
    return out


def window_peak_indices(
    window_power: np.ndarray,
    threshold: float,
) -> np.ndarray:
    indices, _ = find_peaks(window_power)
    endpoint_indices = []
    if len(window_power) >= 2:
        if window_power[0] > window_power[1]:
            endpoint_indices.append(0)
        if window_power[-1] > window_power[-2]:
            endpoint_indices.append(len(window_power) - 1)
    if endpoint_indices:
        indices = np.unique(np.concatenate([indices, np.asarray(endpoint_indices, dtype=int)]))
    if len(indices) == 0:
        return indices
    return indices[window_power[indices] > threshold]


def sampling_window_peaks(
    frequency: np.ndarray,
    window_power: np.ndarray,
    threshold: float,
    min_period: float,
) -> list[dict[str, float]]:
    indices = window_peak_indices(window_power, threshold)
    rows: list[dict[str, float]] = []
    for idx in indices[np.argsort(window_power[indices])[::-1]]:
        period = float(1.0 / frequency[idx])
        if period < min_period:
            continue
        rows.append(
            {
                "period": period,
                "frequency": float(frequency[idx]),
                "power": float(window_power[idx]),
            }
        )
    return rows


def classify_peaks(
    peaks: list[PeakSummary],
    frequency: np.ndarray,
    window_power: np.ndarray,
    baseline: float,
    min_window_period: float = 0.0,
    window_power_threshold: float = 0.01,
    relative_tolerance: float = 0.01,
) -> None:
    ranked = window_peak_indices(window_power, window_power_threshold)
    if len(ranked) == 0:
        return
    resolution = 1.0 / baseline
    grid_tolerance = max(2.0 * np.median(np.diff(frequency)), resolution)
    for peak in peaks:
        eligible = ranked[
            (window_power[ranked] > window_power_threshold)
            & ((1.0 / frequency[ranked]) >= min_window_period)
        ]
        if len(eligible) == 0:
            continue
        distances = np.abs(frequency[eligible] - peak.frequency)
        nearest = int(eligible[int(np.argmin(distances))])
        freq_delta = abs(frequency[nearest] - peak.frequency)
        period_delta = abs((1.0 / frequency[nearest]) - peak.period)
        freq_tolerance = max(grid_tolerance, relative_tolerance * peak.frequency)
        period_grid_tolerance = grid_tolerance / max(peak.frequency**2, 1e-12)
        period_tolerance = max(2.0 * period_grid_tolerance, relative_tolerance * peak.period)
        if freq_delta <= freq_tolerance and period_delta <= period_tolerance:
            peak.kind = "sampling-window artefact"
            peak.window_frequency = float(frequency[nearest])
            peak.window_period = float(1.0 / frequency[nearest])
            peak.window_power = float(window_power[nearest])


def find_lomb_scargle_peaks(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    frequency: np.ndarray,
    max_peaks: int,
    min_considered_period: float = 2.0,
    min_period_separation: float = 0.03,
) -> tuple[np.ndarray, LombScargle, list[PeakSummary]]:
    ls = LombScargle(t, y, dy, center_data=True, fit_mean=True)
    power = ls.power(frequency)
    peak_indices, props = find_peaks(power, prominence=np.nanstd(power) * 0.25)
    if len(peak_indices) == 0:
        peak_indices = np.array([int(np.argmax(power))])
    ranked_all = peak_indices[np.argsort(power[peak_indices])[::-1]]
    selected: list[int] = []
    for idx in ranked_all:
        period = float(1.0 / frequency[idx])
        if period < min_considered_period:
            continue
        if any(abs(period - 1.0 / frequency[old]) < min_period_separation * max(period, 1.0 / frequency[old]) for old in selected):
            continue
        selected.append(int(idx))
        if len(selected) >= max_peaks:
            break
    if not selected:
        selected = [int(ranked_all[0])]
    peaks: list[PeakSummary] = []
    for n, idx in enumerate(selected, start=1):
        f = float(frequency[idx])
        pwr = float(power[idx])
        fap = float(ls.false_alarm_probability(pwr))
        peaks.append(PeakSummary(f"peak {n}", f, float(1.0 / f), pwr, fap, "candidate"))
    return power, ls, peaks


def local_lomb_peak(
    frequency: np.ndarray,
    power: np.ndarray,
    ls: LombScargle,
    target_frequency: float,
    fractional_width: float = 0.03,
) -> PeakSummary | None:
    low = target_frequency * (1.0 - fractional_width)
    high = target_frequency * (1.0 + fractional_width)
    mask = (frequency >= low) & (frequency <= high)
    if not np.any(mask):
        return None
    indices = np.flatnonzero(mask)
    idx = int(indices[int(np.argmax(power[mask]))])
    if idx <= 0 or idx >= len(power) - 1:
        return None
    f = float(frequency[idx])
    pwr = float(power[idx])
    return PeakSummary("local peak", f, float(1.0 / f), pwr, float(ls.false_alarm_probability(pwr)), "candidate")


def append_unique_peak(peaks: list[PeakSummary], peak: PeakSummary | None, label: str, kind: str, tolerance: float = 0.02) -> None:
    if peak is None:
        return
    for old in peaks:
        if abs(old.period - peak.period) < tolerance * max(old.period, peak.period):
            if not label.startswith("peak"):
                old.label = label
            if kind != old.kind and ("artefact" in kind or "harmonic" in kind):
                old.kind = kind
            if peak.window_frequency is not None:
                old.window_frequency = peak.window_frequency
                old.window_period = peak.window_period
                old.window_power = peak.window_power
            return
    peak.label = label
    peak.kind = kind
    peaks.append(peak)


def add_harmonic_and_window_peaks(
    peaks: list[PeakSummary],
    frequency: np.ndarray,
    power: np.ndarray,
    ls: LombScargle,
    window_power: np.ndarray,
    primary: PeakSummary | None,
    min_considered_period: float,
    window_power_threshold: float = 0.01,
    max_harmonic_order: int = 8,
) -> None:
    if primary is not None:
        for order in range(2, max_harmonic_order + 1):
            target_frequency = order * primary.frequency
            if target_frequency > frequency.max():
                break
            harmonic = local_lomb_peak(frequency, power, ls, target_frequency, fractional_width=0.04)
            if harmonic is not None and harmonic.period >= min_considered_period and harmonic.fap <= 0.2:
                append_unique_peak(peaks, harmonic, f"P/{order} harmonic", "harmonic candidate")


def add_bootstrap_errors(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    ls: LombScargle,
    peaks: list[PeakSummary],
    n_bootstrap: int,
    local_width: float,
    n_local_frequency: int,
    seed: int,
) -> None:
    if n_bootstrap <= 1:
        return
    rng = np.random.default_rng(seed)
    n = len(t)
    for peak in peaks:
        if "excluded" in peak.kind:
            continue
        low = max(peak.frequency * (1.0 - local_width), 1e-12)
        high = peak.frequency * (1.0 + local_width)
        local_frequency = np.linspace(low, high, n_local_frequency)
        model = ls.model(t, peak.frequency)
        residuals = y - model
        boot_freq = np.empty(n_bootstrap)
        for i in range(n_bootstrap):
            boot_y = model + residuals[rng.integers(0, n, n)]
            boot_ls = LombScargle(t, boot_y, dy, center_data=True, fit_mean=True)
            boot_power = boot_ls.power(local_frequency)
            boot_freq[i] = local_frequency[int(np.argmax(boot_power))]
        boot_period = 1.0 / boot_freq
        peak.frequency_error = float(np.std(boot_freq, ddof=1))
        peak.frequency_p16 = float(np.percentile(boot_freq, 16))
        peak.frequency_p84 = float(np.percentile(boot_freq, 84))
        peak.period_error = float(np.std(boot_period, ddof=1))
        peak.period_p16 = float(np.percentile(boot_period, 16))
        peak.period_p84 = float(np.percentile(boot_period, 84))


def normalize_fit_method(value: str | None) -> str:
    cleaned = (value or "standard").strip().lower().replace(" ", "_").replace("-", "_")
    if cleaned in {"standard", "weighted"}:
        return "standard"
    if cleaned in {"robust", "soft_l1", "huber"}:
        return "robust"
    if cleaned in {"display", "display_optimized", "display_optimised", "unweighted"}:
        return "display_optimized"
    raise ValueError("Model fitting must be standard, robust, or display-optimized")


def sinusoid_design(x: np.ndarray, frequencies: list[float]) -> np.ndarray:
    cols = [np.ones_like(x)]
    for freq in frequencies:
        phase = 2.0 * np.pi * freq * x
        cols.extend([np.cos(phase), np.sin(phase)])
    return np.column_stack(cols)


def safe_weights(dy: np.ndarray, method: str) -> np.ndarray:
    if method == "display_optimized":
        return np.ones_like(dy, dtype=float)
    safe_dy = np.where(np.isfinite(dy) & (dy > 0.0), dy, np.nan)
    return np.where(np.isfinite(safe_dy), 1.0 / safe_dy**2, 1.0)


def display_range_percentiles(n_points: int) -> tuple[float, float]:
    if n_points >= 50:
        return 1.0, 99.0
    if n_points >= 20:
        return 2.5, 97.5
    return 0.0, 100.0


def fit_sinusoids(
    x: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    frequencies: list[float],
    method: str = "standard",
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    method = normalize_fit_method(method)
    design = sinusoid_design(x, frequencies)
    weights = safe_weights(dy, method)
    sqrt_weights = np.sqrt(weights)
    coeff, *_ = np.linalg.lstsq(design * sqrt_weights[:, None], y * sqrt_weights, rcond=None)
    display_low_percentile = None
    display_high_percentile = None
    if method == "robust":
        scale = float(np.median(np.abs(y - design @ coeff)))
        f_scale = max(scale, float(np.median(dy[np.isfinite(dy) & (dy > 0.0)])) if np.any(np.isfinite(dy) & (dy > 0.0)) else 1.0, 1e-12)
        result = least_squares(
            lambda params: (design @ params - y) * sqrt_weights,
            coeff,
            loss="soft_l1",
            f_scale=f_scale,
            max_nfev=2000,
        )
        coeff = result.x
    elif method == "display_optimized" and len(y) >= 4:
        model_initial = design @ coeff
        display_low_percentile, display_high_percentile = display_range_percentiles(len(y))
        data_low, data_high = np.nanpercentile(y, [display_low_percentile, display_high_percentile])
        model_low, model_high = np.nanpercentile(model_initial, [display_low_percentile, display_high_percentile])
        data_span = float(data_high - data_low)
        model_span = float(model_high - model_low)
        if np.isfinite(data_span) and np.isfinite(model_span) and data_span > 0.0 and model_span > 0.0:
            amplitude_scale = data_span / model_span
            data_midrange = 0.5 * float(data_low + data_high)
            model_midrange = 0.5 * float(model_low + model_high)
            vertical_shift = data_midrange - amplitude_scale * model_midrange
            coeff = coeff * amplitude_scale
            coeff[0] += vertical_shift
    model = design @ coeff
    residuals = y - model
    dof = max(1, len(y) - len(coeff))
    summary: dict[str, float | str] = {
        "method": method,
        "offset": float(coeff[0]),
        "rms": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
        "weighted_rms": float(np.sqrt(np.average(residuals**2, weights=weights))) if len(residuals) else None,
        "chi2_red": float(np.sum(weights * residuals**2) / dof) if len(residuals) else None,
        "n_points": int(len(y)),
        "display_low_percentile": display_low_percentile,
        "display_high_percentile": display_high_percentile,
    }
    return model, coeff, summary


def fit_sinusoids_with_terms(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    terms: list[dict[str, float | str]],
    method: str = "standard",
) -> tuple[np.ndarray, list[dict[str, float | str]]]:
    frequencies = [float(term["frequency"]) for term in terms]
    shifted = t - t.min()
    model, coeff, summary = fit_sinusoids(shifted, y, dy, frequencies, method=method)
    design = sinusoid_design(shifted, frequencies)
    weights = safe_weights(dy, normalize_fit_method(method))
    normal = design.T @ (design * weights[:, None])
    covariance = np.linalg.pinv(normal)
    rows: list[dict[str, float | str]] = []
    for n, term in enumerate(terms):
        cos_idx = 1 + 2 * n
        sin_idx = cos_idx + 1
        cos_coeff = float(coeff[cos_idx])
        sin_coeff = float(coeff[sin_idx])
        amplitude = float(np.hypot(cos_coeff, sin_coeff))
        if amplitude > 0:
            grad = np.asarray([cos_coeff / amplitude, sin_coeff / amplitude])
            cov2 = covariance[np.ix_([cos_idx, sin_idx], [cos_idx, sin_idx])]
            amplitude_error = float(np.sqrt(max(0.0, grad @ cov2 @ grad)))
        else:
            amplitude_error = None
        rows.append(
            {
                "label": str(term["label"]),
                "fit_method": summary["method"],
                "offset": summary["offset"],
                "main_period": term.get("main_period"),
                "main_period_error": term.get("main_period_error"),
                "period": float(1.0 / float(term["frequency"])),
                "period_error": term.get("period_error"),
                "frequency": float(term["frequency"]),
                "frequency_error": term.get("frequency_error"),
                "cos_coeff": cos_coeff,
                "sin_coeff": sin_coeff,
                "amplitude": amplitude,
                "amplitude_error": amplitude_error,
                "phase_of_max": float((np.arctan2(sin_coeff, cos_coeff) / (2.0 * np.pi)) % 1.0) if amplitude > 0 else None,
                "rms": summary["rms"],
                "weighted_rms": summary["weighted_rms"],
                "chi2_red": summary["chi2_red"],
            }
        )
    return model, rows


def detected_harmonics_for_period(
    period: float,
    peaks: list[PeakSummary],
    tolerance: float = 0.04,
    max_harmonic_order: int = 8,
) -> list[tuple[int, PeakSummary]]:
    harmonics: list[tuple[int, PeakSummary]] = []
    used_periods: list[float] = []
    for order in range(2, max_harmonic_order + 1):
        target = period / order
        candidates = [
            peak for peak in usable_candidate_peaks(peaks)
            if abs(peak.period - target) <= tolerance * max(peak.period, target)
        ]
        if not candidates:
            continue
        harmonic = max(candidates, key=lambda peak: peak.power)
        if any(abs(harmonic.period - old) <= 1e-5 * max(harmonic.period, old) for old in used_periods):
            continue
        used_periods.append(harmonic.period)
        harmonics.append((order, harmonic))
    return harmonics


def matching_peak_for_period(period: float, peaks: list[PeakSummary], tolerance: float = 0.04) -> PeakSummary | None:
    candidates = [
        peak for peak in peaks
        if abs(peak.period - period) <= tolerance * max(peak.period, period)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda peak: peak.power)


def prewhitening_terms_from_periods(
    periods: list[float],
    peaks: list[PeakSummary],
) -> list[dict[str, float | str]]:
    terms: list[dict[str, float | str]] = []
    for idx, period in enumerate(periods, start=1):
        main_peak = matching_peak_for_period(period, peaks)
        terms.append(
            {
                "label": f"step {idx}",
                "frequency": 1.0 / period,
                "main_period": period,
                "main_period_error": None if main_peak is None else main_peak.period_error,
                "period_error": None if main_peak is None else main_peak.period_error,
                "frequency_error": None if main_peak is None else main_peak.frequency_error,
            }
        )
    return terms


def folded_profile(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    period: float,
    t0: float,
    n_bins: int,
    phase_frequency_ratios: list[float],
    fit_method: str = "standard",
) -> dict:
    phase = ((t - t0) / period) % 1.0
    weights = 1.0 / dy**2
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, means, errors, counts = [], [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (phase >= low) & (phase < high)
        if not np.any(mask):
            continue
        w = weights[mask]
        centers.append(0.5 * (low + high))
        means.append(np.average(y[mask], weights=w))
        errors.append(1.0 / np.sqrt(w.sum()))
        counts.append(int(mask.sum()))
    centers = np.asarray(centers)
    means = np.asarray(means)
    errors = np.asarray(errors)
    counts = np.asarray(counts)

    _, coeff, fit_summary = fit_sinusoids(centers, means, errors, phase_frequency_ratios, method=fit_method)

    def model(ph: np.ndarray) -> np.ndarray:
        return sinusoid_design(ph, phase_frequency_ratios) @ coeff

    grid = np.linspace(0.0, 1.0, 50001, endpoint=False)
    values = model(grid)
    peak_idx, _ = find_peaks(values)
    maxima = []
    for idx in peak_idx:
        guess = grid[idx]
        result = minimize_scalar(lambda x: -float(model(np.asarray([x]))[0]), bounds=(max(0, guess - 0.03), min(1, guess + 0.03)), method="bounded")
        ph = float(result.x)
        val = float(model(np.asarray([ph]))[0])
        if all(abs(ph - old[0]) > 1e-4 for old in maxima):
            maxima.append((ph, val))
    maxima.sort(key=lambda x: x[0])
    terms = []
    for idx, ratio in enumerate(phase_frequency_ratios):
        cos_idx = 1 + 2 * idx
        sin_idx = cos_idx + 1
        cos_coeff = float(coeff[cos_idx])
        sin_coeff = float(coeff[sin_idx])
        amplitude = float(np.hypot(cos_coeff, sin_coeff))
        phase_of_max = float((np.arctan2(sin_coeff, cos_coeff) / (2.0 * np.pi * ratio)) % (1.0 / ratio)) if amplitude > 0 and ratio != 0 else None
        terms.append(
            {
                "component": idx + 1,
                "fit_method": fit_summary["method"],
                "offset": fit_summary["offset"],
                "frequency_ratio": float(ratio),
                "cos_coeff": cos_coeff,
                "sin_coeff": sin_coeff,
                "amplitude": amplitude,
                "phase_of_max": phase_of_max,
                "rms": fit_summary["rms"],
                "weighted_rms": fit_summary["weighted_rms"],
                "chi2_red": fit_summary["chi2_red"],
            }
        )
    return {
        "phase": centers,
        "flux": means,
        "error": errors,
        "counts": counts,
        "model_phase": np.linspace(0.0, 2.0, 1000),
        "model_flux": model(np.linspace(0.0, 2.0, 1000) % 1.0),
        "maxima": maxima,
        "phase_frequency_ratios": phase_frequency_ratios,
        "fit_summary": fit_summary,
        "fit_terms": terms,
    }


def phase_binned_profile(
    phase: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = 1.0 / np.where(np.isfinite(dy) & (dy > 0.0), dy, 1.0) ** 2
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centers, means, errors, counts = [], [], [], []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (phase >= low) & (phase < high)
        if not np.any(mask):
            continue
        w = weights[mask]
        centers.append(0.5 * (low + high))
        means.append(np.average(y[mask], weights=w))
        errors.append(1.0 / np.sqrt(w.sum()))
        counts.append(int(mask.sum()))
    return np.asarray(centers), np.asarray(means), np.asarray(errors), np.asarray(counts)


def information_criteria(residuals: np.ndarray, n_parameters: int) -> tuple[float, float]:
    n = max(1, int(len(residuals)))
    variance = max(float(np.mean(residuals**2)), 1e-300)
    aic = float(n * np.log(variance) + 2 * n_parameters)
    bic = float(n * np.log(variance) + n_parameters * np.log(n))
    return aic, bic


def periodic_model_maxima(frequencies: list[float], coeff: np.ndarray) -> list[tuple[float, float]]:
    def model(ph: np.ndarray) -> np.ndarray:
        return sinusoid_design(ph, frequencies) @ coeff

    grid = np.linspace(0.0, 1.0, 50001, endpoint=False)
    values = model(grid)
    peak_idx, _ = find_peaks(values)
    maxima = []
    for idx in peak_idx:
        guess = grid[idx]
        result = minimize_scalar(
            lambda x: -float(model(np.asarray([x]))[0]),
            bounds=(max(0.0, guess - 0.03), min(1.0, guess + 0.03)),
            method="bounded",
        )
        ph = float(result.x)
        val = float(model(np.asarray([ph]))[0])
        if all(abs(ph - old[0]) > 1e-4 for old in maxima):
            maxima.append((ph, val))
    maxima.sort(key=lambda item: item[0])
    return maxima


def sampled_curve_extrema(
    phase_grid: np.ndarray,
    values: np.ndarray,
    kind: str = "model maximum",
) -> list[dict[str, float | str]]:
    peak_idx, _ = find_peaks(values)
    extrema = [
        {"phase": float(phase_grid[idx]), "flux": float(values[idx]), "kind": kind}
        for idx in peak_idx
        if 0.0 <= phase_grid[idx] < 1.0
    ]
    return extrema


def sampled_curve_minima(
    phase_grid: np.ndarray,
    values: np.ndarray,
    kind: str = "model minimum",
) -> list[dict[str, float | str]]:
    peak_idx, _ = find_peaks(-values)
    extrema = [
        {"phase": float(phase_grid[idx]), "flux": float(values[idx]), "kind": kind}
        for idx in peak_idx
        if 0.0 <= phase_grid[idx] < 1.0
    ]
    extrema.sort(key=lambda row: row["flux"])
    return extrema


def fourier_model_lab(result: dict, fields: dict[str, str]) -> dict:
    series = result.get("series", {})
    t = np.asarray(series.get("time", []), dtype=float)
    y = np.asarray(series.get("flux", []), dtype=float)
    dy = np.asarray(series.get("error", []), dtype=float)
    if len(t) < 4 or len(y) != len(t):
        raise ValueError("Fourier model lab requires a completed analysis with at least 4 data points")
    if len(dy) != len(t):
        dy = np.ones_like(y)

    period_raw = str(fields.get("model_lab_period", "")).strip()
    default_period = result.get("folded_period") or result.get("primary_period")
    if period_raw:
        period = float(period_raw)
    elif default_period is not None:
        period = float(default_period)
    else:
        raise ValueError("Choose a Fourier model period or run an analysis with a primary period")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("Fourier model period must be positive")

    t0_raw = str(fields.get("model_lab_t0", "")).strip()
    t0 = float(t0_raw) if t0_raw else float(result.get("t0", t[0]))
    n_bins = max(4, int(fields.get("model_lab_bins", fields.get("fold_bins", "20"))))
    fit_method = normalize_fit_method(fields.get("model_lab_fit_method", fields.get("model_fit_method", "standard")))
    selection = fields.get("model_lab_fourier_selection", "manual").strip().lower()
    manual_harmonics = max(1, int(fields.get("model_lab_fourier_harmonics", "3")))
    max_harmonics = max(1, int(fields.get("model_lab_fourier_max_harmonics", str(manual_harmonics))))
    max_harmonics = min(max_harmonics, max(1, (len(y) - 1) // 2), 20)

    phase = ((t - t0) / period) % 1.0
    trials = []
    selected_trial = None
    for n_harmonics in range(1, max_harmonics + 1):
        ratios = [float(order) for order in range(1, n_harmonics + 1)]
        model_at_data, coeff, summary = fit_sinusoids(phase, y, dy, ratios, method=fit_method)
        residuals = y - model_at_data
        n_parameters = len(coeff)
        aic, bic = information_criteria(residuals, n_parameters)
        trial = {
            "harmonics": n_harmonics,
            "n_parameters": n_parameters,
            "rms": summary["rms"],
            "weighted_rms": summary["weighted_rms"],
            "chi2_red": summary["chi2_red"],
            "AIC": aic,
            "BIC": bic,
            "_coeff": coeff,
            "_model": model_at_data,
            "_summary": summary,
        }
        trials.append(trial)
    if selection == "aic":
        selected_trial = min(trials, key=lambda item: float(item["AIC"]))
    elif selection == "bic":
        selected_trial = min(trials, key=lambda item: float(item["BIC"]))
    else:
        target = min(manual_harmonics, max_harmonics)
        selected_trial = trials[target - 1]
        selection = "manual"

    n_selected = int(selected_trial["harmonics"])
    ratios = [float(order) for order in range(1, n_selected + 1)]
    coeff = np.asarray(selected_trial["_coeff"], dtype=float)
    model_at_data = np.asarray(selected_trial["_model"], dtype=float)
    summary = dict(selected_trial["_summary"])
    centers, means, errors, counts = phase_binned_profile(phase, y, dy, n_bins)
    model_phase = np.linspace(0.0, 2.0, 1600)
    model_flux = sinusoid_design(model_phase % 1.0, ratios) @ coeff
    model_time = np.linspace(float(np.nanmin(t)), float(np.nanmax(t)), 2500)
    model_time_phase = ((model_time - t0) / period) % 1.0
    model_time_flux = sinusoid_design(model_time_phase, ratios) @ coeff
    maxima = periodic_model_maxima(ratios, coeff)

    terms = []
    for idx, ratio in enumerate(ratios):
        cos_idx = 1 + 2 * idx
        sin_idx = cos_idx + 1
        cos_coeff = float(coeff[cos_idx])
        sin_coeff = float(coeff[sin_idx])
        amplitude = float(np.hypot(cos_coeff, sin_coeff))
        terms.append(
            {
                "component": idx + 1,
                "fit_method": summary["method"],
                "offset": float(coeff[0]),
                "frequency_ratio": ratio,
                "period": period / ratio,
                "cos_coeff": cos_coeff,
                "sin_coeff": sin_coeff,
                "amplitude": amplitude,
                "phase_of_max": float((np.arctan2(sin_coeff, cos_coeff) / (2.0 * np.pi * ratio)) % (1.0 / ratio)) if amplitude > 0 else None,
                "rms": summary["rms"],
                "weighted_rms": summary["weighted_rms"],
                "chi2_red": summary["chi2_red"],
            }
        )

    clean_trials = [
        {key: value for key, value in trial.items() if not key.startswith("_")}
        for trial in trials
    ]
    return {
        "family": "fourier",
        "period": period,
        "t0": t0,
        "bins": n_bins,
        "selection": selection,
        "selected_harmonics": n_selected,
        "period_label": result.get("period_label", "Period"),
        "phase": centers.tolist(),
        "flux": means.tolist(),
        "error": errors.tolist(),
        "counts": counts.tolist(),
        "data_phase": phase.tolist(),
        "data_time": t.tolist(),
        "data_flux": y.tolist(),
        "data_error": dy.tolist(),
        "model_at_data": model_at_data.tolist(),
        "model_phase": model_phase.tolist(),
        "model_flux": model_flux.tolist(),
        "model_time": model_time.tolist(),
        "model_time_flux": model_time_flux.tolist(),
        "maxima": [{"phase": ph, "flux": val} for ph, val in maxima],
        "terms": terms,
        "summary": {
            **summary,
            "selected_harmonics": n_selected,
            "selection": selection,
            "AIC": selected_trial["AIC"],
            "BIC": selected_trial["BIC"],
            "n_parameters": selected_trial["n_parameters"],
        },
        "trials": clean_trials,
    }


def solve_kepler(mean_anomaly: np.ndarray, eccentricity: float, max_iter: int = 60) -> np.ndarray:
    mean_anomaly = np.asarray(mean_anomaly, dtype=float)
    eccentricity = float(np.clip(eccentricity, 0.0, 0.95))
    eccentric_anomaly = mean_anomaly.copy()
    if eccentricity > 0.75:
        eccentric_anomaly = np.where(np.sin(mean_anomaly) >= 0.0, np.pi, -np.pi)
    for _ in range(max_iter):
        step = (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly) / (
            1.0 - eccentricity * np.cos(eccentric_anomaly)
        )
        eccentric_anomaly -= step
        if np.nanmax(np.abs(step)) < 1e-12:
            break
    return eccentric_anomaly


def true_anomaly_from_phase(phase: np.ndarray, eccentricity: float) -> np.ndarray:
    mean_anomaly = 2.0 * np.pi * phase
    eccentric_anomaly = solve_kepler(mean_anomaly, eccentricity)
    return np.arctan2(
        np.sqrt(1.0 - eccentricity**2) * np.sin(eccentric_anomaly),
        np.cos(eccentric_anomaly) - eccentricity,
    )


def eccentric_design(phase: np.ndarray, eccentricity: float, n_harmonics: int) -> np.ndarray:
    true_anomaly = true_anomaly_from_phase(phase, eccentricity)
    cols = [np.ones_like(phase)]
    for order in range(1, n_harmonics + 1):
        angle = order * true_anomaly
        cols.extend([np.cos(angle), np.sin(angle)])
    return np.column_stack(cols)


def fit_weighted_design(design: np.ndarray, y: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    weights = safe_weights(dy, "standard")
    sqrt_weights = np.sqrt(weights)
    coeff, *_ = np.linalg.lstsq(design * sqrt_weights[:, None], y * sqrt_weights, rcond=None)
    model = design @ coeff
    residuals = y - model
    dof = max(1, len(y) - len(coeff))
    summary: dict[str, float | str] = {
        "method": "weighted",
        "offset": float(coeff[0]),
        "rms": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
        "weighted_rms": float(np.sqrt(np.average(residuals**2, weights=weights))) if len(residuals) else None,
        "chi2_red": float(np.sum(weights * residuals**2) / dof) if len(residuals) else None,
        "n_points": int(len(y)),
    }
    return model, coeff, summary


def fit_eccentric_harmonic_model(
    phase: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    n_harmonics: int,
    fit_eccentricity: bool,
    eccentricity_guess: float,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, float | str]]:
    n_harmonics = max(1, int(n_harmonics))
    eccentricity_guess = float(np.clip(eccentricity_guess, 0.0, 0.9))

    def score(eccentricity: float) -> float:
        design = eccentric_design(phase, eccentricity, n_harmonics)
        model, _, _ = fit_weighted_design(design, y, dy)
        weights = safe_weights(dy, "standard")
        return float(np.sum(weights * (y - model) ** 2))

    if fit_eccentricity:
        optimized = minimize_scalar(score, bounds=(0.0, 0.9), method="bounded", options={"xatol": 1e-4})
        eccentricity = float(optimized.x)
    else:
        eccentricity = eccentricity_guess
    design = eccentric_design(phase, eccentricity, n_harmonics)
    model, coeff, summary = fit_weighted_design(design, y, dy)
    summary["method"] = "eccentric_harmonic"
    summary["eccentricity"] = eccentricity
    return model, coeff, eccentricity, summary


def circular_distance(phase: np.ndarray, center: float) -> np.ndarray:
    return ((phase - center + 0.5) % 1.0) - 0.5


def eclipse_gaussian(phase: np.ndarray, center: float, width: float) -> np.ndarray:
    width = max(float(width), 1e-4)
    return np.exp(-0.5 * (circular_distance(phase, center) / width) ** 2)


def empirical_eclipse_model(phase: np.ndarray, params: np.ndarray, include_secondary: bool) -> np.ndarray:
    offset, ellip_cos, ellip_sin, depth1, center1, width1 = params[:6]
    model = (
        offset
        + ellip_cos * np.cos(4.0 * np.pi * phase)
        + ellip_sin * np.sin(4.0 * np.pi * phase)
        + depth1 * eclipse_gaussian(phase, center1, width1)
    )
    if include_secondary:
        depth2, center2, width2 = params[6:9]
        model += depth2 * eclipse_gaussian(phase, center2, width2)
    return model


def fit_empirical_eclipse_model(
    phase: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    include_secondary: bool,
    primary_center_guess: float | None,
    secondary_center_guess: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    weights = safe_weights(dy, "standard")
    sqrt_weights = np.sqrt(weights)
    median = float(np.nanmedian(y))
    span = max(float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5)), 1e-6)
    if primary_center_guess is None:
        primary_center_guess = float(phase[int(np.nanargmax(np.abs(y - median)))])
    if secondary_center_guess is None:
        secondary_center_guess = float((primary_center_guess + 0.5) % 1.0)
    depth1_guess = float(y[np.argmin(np.abs(circular_distance(phase, primary_center_guess)))] - median)
    depth2_guess = float(y[np.argmin(np.abs(circular_distance(phase, secondary_center_guess)))] - median)
    params0 = [
        median,
        0.0,
        0.0,
        depth1_guess,
        primary_center_guess % 1.0,
        0.05,
    ]
    lower = [median - 3.0 * span, -3.0 * span, -3.0 * span, -3.0 * span, 0.0, 0.005]
    upper = [median + 3.0 * span, 3.0 * span, 3.0 * span, 3.0 * span, 1.0, 0.3]
    if include_secondary:
        params0.extend([depth2_guess, secondary_center_guess % 1.0, 0.05])
        lower.extend([-3.0 * span, 0.0, 0.005])
        upper.extend([3.0 * span, 1.0, 0.3])

    result = least_squares(
        lambda params: (empirical_eclipse_model(phase, params, include_secondary) - y) * sqrt_weights,
        np.asarray(params0, dtype=float),
        bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
        loss="soft_l1",
        f_scale=max(float(np.nanmedian(dy[np.isfinite(dy) & (dy > 0.0)])) if np.any(np.isfinite(dy) & (dy > 0.0)) else span * 0.05, 1e-8),
        max_nfev=5000,
    )
    params = result.x
    model = empirical_eclipse_model(phase, params, include_secondary)
    residuals = y - model
    dof = max(1, len(y) - len(params))
    summary: dict[str, float | str] = {
        "method": "empirical_eclipses",
        "offset": float(params[0]),
        "rms": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
        "weighted_rms": float(np.sqrt(np.average(residuals**2, weights=weights))) if len(residuals) else None,
        "chi2_red": float(np.sum(weights * residuals**2) / dof) if len(residuals) else None,
        "n_points": int(len(y)),
    }
    return model, params, summary


def disk_overlap_area(distance: np.ndarray, radius1: float, radius2: float) -> np.ndarray:
    distance = np.asarray(distance, dtype=float)
    radius1 = max(float(radius1), 1e-6)
    radius2 = max(float(radius2), 1e-6)
    area = np.zeros_like(distance, dtype=float)

    separated = distance >= radius1 + radius2
    contained = distance <= abs(radius1 - radius2)
    area[contained] = np.pi * min(radius1, radius2) ** 2

    partial = ~(separated | contained)
    if np.any(partial):
        d = np.maximum(distance[partial], 1e-10)
        arg1 = np.clip((d**2 + radius1**2 - radius2**2) / (2.0 * d * radius1), -1.0, 1.0)
        arg2 = np.clip((d**2 + radius2**2 - radius1**2) / (2.0 * d * radius2), -1.0, 1.0)
        term = np.maximum(
            (-d + radius1 + radius2)
            * (d + radius1 - radius2)
            * (d - radius1 + radius2)
            * (d + radius1 + radius2),
            0.0,
        )
        area[partial] = radius1**2 * np.arccos(arg1) + radius2**2 * np.arccos(arg2) - 0.5 * np.sqrt(term)
    return area


def physical_eclipse_proxy(
    phase: np.ndarray,
    eccentricity: float,
    omega_deg: float,
    inclination_deg: float,
    radius1_over_a: float,
    radius2_over_a: float,
    temperature1: float,
    temperature2: float,
    limb_u1: float,
    limb_u2: float,
    third_light_fraction: float,
    ellipsoidal_amp: float,
    reflection_amp: float,
    beaming_amp: float,
    include_ellipsoidal: bool,
    include_reflection: bool,
    include_beaming: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phase = np.asarray(phase, dtype=float) % 1.0
    eccentricity = float(np.clip(eccentricity, 0.0, 0.9))
    omega = np.deg2rad(float(omega_deg))
    inclination = np.deg2rad(float(np.clip(inclination_deg, 0.0, 90.0)))
    radius1_over_a = float(np.clip(radius1_over_a, 1e-4, 0.9))
    radius2_over_a = float(np.clip(radius2_over_a, 1e-4, 0.9))
    temperature1 = max(float(temperature1), 1.0)
    temperature2 = max(float(temperature2), 1.0)
    limb_u1 = float(np.clip(limb_u1, 0.0, 1.0))
    limb_u2 = float(np.clip(limb_u2, 0.0, 1.0))
    third_light_fraction = max(float(third_light_fraction), 0.0)

    true_anomaly = true_anomaly_from_phase(phase, eccentricity)
    separation = (1.0 - eccentricity**2) / np.maximum(1.0 + eccentricity * np.cos(true_anomaly), 1e-6)
    theta = true_anomaly + omega
    sin_i = np.sin(inclination)
    cos_i = np.cos(inclination)
    projected_separation = separation * np.sqrt(np.cos(theta) ** 2 + (np.sin(theta) * cos_i) ** 2)
    line_of_sight = separation * sin_i * np.sin(theta)

    surface1 = max(1.0 - limb_u1 / 3.0, 1e-4)
    surface2 = (temperature2 / temperature1) ** 4 * max(1.0 - limb_u2 / 3.0, 1e-4)
    flux1 = surface1 * np.pi * radius1_over_a**2
    flux2 = surface2 * np.pi * radius2_over_a**2
    third_light = third_light_fraction * (flux1 + flux2)
    overlap = disk_overlap_area(projected_separation, radius1_over_a, radius2_over_a)

    blocked_surface = np.where(line_of_sight >= 0.0, surface1, surface2)
    flux = flux1 + flux2 + third_light - overlap * blocked_surface
    median_flux = float(np.nanmedian(flux)) if np.any(np.isfinite(flux)) else 1.0
    proxy = flux / max(abs(median_flux), 1e-10) - 1.0

    if include_ellipsoidal:
        proxy += float(ellipsoidal_amp) * np.cos(4.0 * np.pi * phase)
    if include_reflection:
        proxy += float(reflection_amp) * np.cos(2.0 * np.pi * phase)
    if include_beaming:
        proxy += float(beaming_amp) * np.sin(2.0 * np.pi * phase)
    return proxy, projected_separation, line_of_sight, separation


def binary_semi_major_axis_rsun(period: float, result: dict, mass1_msun: float, mass2_msun: float) -> float:
    period_unit = str(result.get("period_unit", result.get("baseline_unit", "d"))).lower()
    period_seconds = period * (DAY_SI if period_unit.startswith("d") else 1.0)
    total_mass = max(float(mass1_msun) + float(mass2_msun), 1e-6) * M_SUN_SI
    semi_major_axis_m = (G_SI * total_mass * (period_seconds / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
    return float(semi_major_axis_m / R_SUN_SI)


def fit_physical_eclipse_toy_model(
    phase: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    physical_params: dict[str, float | bool],
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str], np.ndarray, np.ndarray, np.ndarray]:
    weights = safe_weights(dy, "standard")
    sqrt_weights = np.sqrt(weights)
    proxy, projected_separation, line_of_sight, separation = physical_eclipse_proxy(
        phase,
        float(physical_params["eccentricity"]),
        float(physical_params["omega_deg"]),
        float(physical_params["inclination_deg"]),
        float(physical_params["radius1_over_a"]),
        float(physical_params["radius2_over_a"]),
        float(physical_params["temperature1"]),
        float(physical_params["temperature2"]),
        float(physical_params["limb_u1"]),
        float(physical_params["limb_u2"]),
        float(physical_params["third_light_fraction"]),
        float(physical_params["ellipsoidal_amp"]),
        float(physical_params["reflection_amp"]),
        float(physical_params["beaming_amp"]),
        bool(physical_params["include_ellipsoidal"]),
        bool(physical_params["include_reflection"]),
        bool(physical_params["include_beaming"]),
    )
    design = np.column_stack([np.ones_like(proxy), proxy])
    coeff, *_ = np.linalg.lstsq(design * sqrt_weights[:, None], y * sqrt_weights, rcond=None)
    model = design @ coeff
    residuals = y - model
    dof = max(1, len(y) - len(coeff))
    summary: dict[str, float | str] = {
        "method": "physical_eclipse_toy",
        "offset": float(coeff[0]),
        "scale": float(coeff[1]),
        "rms": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
        "weighted_rms": float(np.sqrt(np.average(residuals**2, weights=weights))) if len(residuals) else None,
        "chi2_red": float(np.sum(weights * residuals**2) / dof) if len(residuals) else None,
        "n_points": int(len(y)),
    }
    return model, coeff, summary, projected_separation, line_of_sight, separation


def binary_model_lab(result: dict, fields: dict[str, str]) -> dict:
    series = result.get("series", {})
    t = np.asarray(series.get("time", []), dtype=float)
    y = np.asarray(series.get("flux", []), dtype=float)
    dy = np.asarray(series.get("error", []), dtype=float)
    if len(t) < 8 or len(y) != len(t):
        raise ValueError("Binary model lab requires a completed analysis with at least 8 data points")
    if len(dy) != len(t):
        dy = np.ones_like(y)

    period_raw = str(fields.get("model_lab_binary_period", fields.get("model_lab_period", ""))).strip()
    default_period = result.get("folded_period") or result.get("primary_period")
    if period_raw:
        period = float(period_raw)
    elif default_period is not None:
        period = float(default_period)
    else:
        raise ValueError("Choose a binary model period or run an analysis with a primary period")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("Binary model period must be positive")
    t0_raw = str(fields.get("model_lab_binary_t0", fields.get("model_lab_t0", ""))).strip()
    t0 = float(t0_raw) if t0_raw else float(result.get("t0", t[0]))
    n_bins = max(4, int(fields.get("model_lab_binary_bins", fields.get("fold_bins", "24"))))
    phase = ((t - t0) / period) % 1.0
    centers, means, errors, counts = phase_binned_profile(phase, y, dy, n_bins)
    model_kind = fields.get("model_lab_binary_kind", "eccentric_harmonic").strip().lower()
    model_time = np.linspace(float(np.nanmin(t)), float(np.nanmax(t)), 2500)
    model_time_phase = ((model_time - t0) / period) % 1.0
    model_phase = np.linspace(0.0, 2.0, 1600)

    if model_kind == "physical_eclipse_toy":
        eccentricity = float(np.clip(float(fields.get("model_lab_binary_physical_eccentricity", fields.get("model_lab_binary_eccentricity", "0.0"))), 0.0, 0.9))
        omega_deg = float(fields.get("model_lab_binary_omega", "90.0"))
        inclination_deg = float(np.clip(float(fields.get("model_lab_binary_inclination", "85.0")), 0.0, 90.0))
        mass1_msun = max(float(fields.get("model_lab_binary_mass1", "10.0")), 1e-4)
        mass2_msun = max(float(fields.get("model_lab_binary_mass2", "1.4")), 1e-4)
        radius1_rsun = max(float(fields.get("model_lab_binary_radius1", "6.0")), 1e-4)
        radius2_rsun = max(float(fields.get("model_lab_binary_radius2", "1.0")), 1e-4)
        temperature1 = max(float(fields.get("model_lab_binary_temperature1", "20000.0")), 1.0)
        temperature2 = max(float(fields.get("model_lab_binary_temperature2", "8000.0")), 1.0)
        limb_u1 = float(np.clip(float(fields.get("model_lab_binary_limb_u1", "0.4")), 0.0, 1.0))
        limb_u2 = float(np.clip(float(fields.get("model_lab_binary_limb_u2", "0.4")), 0.0, 1.0))
        third_light_fraction = max(float(fields.get("model_lab_binary_third_light", "0.0")), 0.0)
        include_ellipsoidal = str(fields.get("model_lab_binary_include_ellipsoidal", "true")).strip().lower() in {"1", "true", "yes", "on"}
        include_reflection = str(fields.get("model_lab_binary_include_reflection", "true")).strip().lower() in {"1", "true", "yes", "on"}
        include_beaming = str(fields.get("model_lab_binary_include_beaming", "false")).strip().lower() in {"1", "true", "yes", "on"}
        ellipsoidal_amp = float(fields.get("model_lab_binary_ellipsoidal_amp", "0.02"))
        reflection_amp = float(fields.get("model_lab_binary_reflection_amp", "0.01"))
        beaming_amp = float(fields.get("model_lab_binary_beaming_amp", "0.0"))

        semi_major_axis_rsun = binary_semi_major_axis_rsun(period, result, mass1_msun, mass2_msun)
        radius1_over_a = radius1_rsun / max(semi_major_axis_rsun, 1e-8)
        radius2_over_a = radius2_rsun / max(semi_major_axis_rsun, 1e-8)
        physical_params: dict[str, float | bool] = {
            "eccentricity": eccentricity,
            "omega_deg": omega_deg,
            "inclination_deg": inclination_deg,
            "radius1_over_a": radius1_over_a,
            "radius2_over_a": radius2_over_a,
            "temperature1": temperature1,
            "temperature2": temperature2,
            "limb_u1": limb_u1,
            "limb_u2": limb_u2,
            "third_light_fraction": third_light_fraction,
            "ellipsoidal_amp": ellipsoidal_amp,
            "reflection_amp": reflection_amp,
            "beaming_amp": beaming_amp,
            "include_ellipsoidal": include_ellipsoidal,
            "include_reflection": include_reflection,
            "include_beaming": include_beaming,
        }
        model_at_data, coeff, summary, projected_separation, line_of_sight, separation = fit_physical_eclipse_toy_model(phase, y, dy, physical_params)
        proxy_phase, projected_phase, line_phase, separation_phase = physical_eclipse_proxy(model_phase % 1.0, **physical_params)
        model_flux = coeff[0] + coeff[1] * proxy_phase
        proxy_time, _, _, _ = physical_eclipse_proxy(model_time_phase, **physical_params)
        model_time_flux = coeff[0] + coeff[1] * proxy_time
        residuals = y - model_at_data
        n_parameters = 2
        aic, bic = information_criteria(residuals, n_parameters)
        q = mass2_msun / mass1_msun
        brightness_ratio = (temperature2 / temperature1) ** 4 * (radius2_rsun / radius1_rsun) ** 2
        eclipse_possible = np.cos(np.deg2rad(inclination_deg)) < radius1_over_a + radius2_over_a
        extrema = sampled_curve_minima(model_phase, model_flux, "primary eclipse" if coeff[1] >= 0.0 else "model minimum")
        if coeff[1] < 0.0:
            extrema = sampled_curve_extrema(model_phase, model_flux, "primary eclipse")
            extrema.sort(key=lambda row: row["flux"], reverse=True)
        extrema = extrema[:2]
        for idx, row in enumerate(extrema):
            row["kind"] = "primary eclipse" if idx == 0 else "secondary eclipse"
        params_table = [
            {"parameter": "offset", "value": float(coeff[0])},
            {"parameter": "scale", "value": float(coeff[1])},
            {"parameter": "eccentricity", "value": eccentricity},
            {"parameter": "omega_deg", "value": omega_deg},
            {"parameter": "inclination_deg", "value": inclination_deg},
            {"parameter": "mass1_msun", "value": mass1_msun},
            {"parameter": "mass2_msun", "value": mass2_msun},
            {"parameter": "mass_ratio_q_m2_over_m1", "value": q},
            {"parameter": "semi_major_axis_rsun", "value": semi_major_axis_rsun},
            {"parameter": "radius1_rsun", "value": radius1_rsun},
            {"parameter": "radius2_rsun", "value": radius2_rsun},
            {"parameter": "radius1_over_a", "value": radius1_over_a},
            {"parameter": "radius2_over_a", "value": radius2_over_a},
            {"parameter": "temperature1_K", "value": temperature1},
            {"parameter": "temperature2_K", "value": temperature2},
            {"parameter": "brightness_ratio_f2_over_f1", "value": brightness_ratio},
            {"parameter": "limb_darkening_u1", "value": limb_u1},
            {"parameter": "limb_darkening_u2", "value": limb_u2},
            {"parameter": "third_light_fraction", "value": third_light_fraction},
            {"parameter": "ellipsoidal_amp_proxy", "value": ellipsoidal_amp if include_ellipsoidal else 0.0},
            {"parameter": "reflection_amp_proxy", "value": reflection_amp if include_reflection else 0.0},
            {"parameter": "beaming_amp_proxy", "value": beaming_amp if include_beaming else 0.0},
        ]
        summary = {
            **summary,
            "eccentricity": eccentricity,
            "omega_deg": omega_deg,
            "inclination_deg": inclination_deg,
            "mass_ratio_q_m2_over_m1": q,
            "semi_major_axis_rsun": semi_major_axis_rsun,
            "radius1_over_a": radius1_over_a,
            "radius2_over_a": radius2_over_a,
            "temperature_ratio_t2_over_t1": temperature2 / temperature1,
            "brightness_ratio_f2_over_f1": brightness_ratio,
            "third_light_fraction": third_light_fraction,
            "eclipse_possible": "yes" if eclipse_possible else "unlikely",
            "minimum_projected_separation_over_a": float(np.nanmin(projected_phase)) if len(projected_phase) else None,
            "mean_separation_over_a": float(np.nanmean(separation_phase)) if len(separation_phase) else None,
        }
        formula = (
            "y(phi) = C + A Q(phi; e, omega, i, R1/a, R2/a, T2/T1, limb darkening, L3) "
            "+ optional ellipsoidal/reflection/beaming proxy terms"
        )
    elif model_kind == "empirical_eclipses":
        include_secondary = str(fields.get("model_lab_binary_secondary", "true")).strip().lower() in {"1", "true", "yes", "on"}
        primary_guess = optional_float(fields.get("model_lab_binary_primary_phase"))
        secondary_guess = optional_float(fields.get("model_lab_binary_secondary_phase"))
        model_at_data, params, summary = fit_empirical_eclipse_model(
            phase,
            y,
            dy,
            include_secondary,
            primary_guess,
            secondary_guess,
        )
        model_flux = empirical_eclipse_model(model_phase % 1.0, params, include_secondary)
        model_time_flux = empirical_eclipse_model(model_time_phase, params, include_secondary)
        residuals = y - model_at_data
        n_parameters = len(params)
        aic, bic = information_criteria(residuals, n_parameters)
        params_table = [
            {"parameter": "offset", "value": float(params[0])},
            {"parameter": "ellipsoidal_cos", "value": float(params[1])},
            {"parameter": "ellipsoidal_sin", "value": float(params[2])},
            {"parameter": "primary_depth", "value": float(params[3])},
            {"parameter": "primary_phase", "value": float(params[4])},
            {"parameter": "primary_width_phase", "value": float(params[5])},
        ]
        if include_secondary:
            params_table.extend([
                {"parameter": "secondary_depth", "value": float(params[6])},
                {"parameter": "secondary_phase", "value": float(params[7])},
                {"parameter": "secondary_width_phase", "value": float(params[8])},
            ])
        formula = "y(phi) = C + E_c cos(4 pi phi) + E_s sin(4 pi phi) + D1 exp(-0.5 d(phi,phi1)^2/w1^2)"
        if include_secondary:
            formula += " + D2 exp(-0.5 d(phi,phi2)^2/w2^2)"
        extrema = [
            {"phase": float(params[4]), "flux": float(empirical_eclipse_model(np.asarray([params[4]]), params, include_secondary)[0]), "kind": "primary eclipse"},
        ]
        if include_secondary:
            extrema.append({"phase": float(params[7]), "flux": float(empirical_eclipse_model(np.asarray([params[7]]), params, include_secondary)[0]), "kind": "secondary eclipse"})
    else:
        n_harmonics = max(1, int(fields.get("model_lab_binary_harmonics", "2")))
        fit_eccentricity = str(fields.get("model_lab_binary_fit_eccentricity", "true")).strip().lower() in {"1", "true", "yes", "on"}
        eccentricity_guess = float(fields.get("model_lab_binary_eccentricity", "0.2"))
        model_at_data, coeff, eccentricity, summary = fit_eccentric_harmonic_model(
            phase,
            y,
            dy,
            n_harmonics,
            fit_eccentricity,
            eccentricity_guess,
        )
        model_flux = eccentric_design(model_phase % 1.0, eccentricity, n_harmonics) @ coeff
        model_time_flux = eccentric_design(model_time_phase, eccentricity, n_harmonics) @ coeff
        residuals = y - model_at_data
        n_parameters = len(coeff) + (1 if fit_eccentricity else 0)
        aic, bic = information_criteria(residuals, n_parameters)
        params_table = [
            {"parameter": "eccentricity", "value": eccentricity},
            {"parameter": "offset", "value": float(coeff[0])},
        ]
        for idx in range(n_harmonics):
            cos_idx = 1 + 2 * idx
            sin_idx = cos_idx + 1
            params_table.extend([
                {"parameter": f"cos_true_anomaly_h{idx + 1}", "value": float(coeff[cos_idx])},
                {"parameter": f"sin_true_anomaly_h{idx + 1}", "value": float(coeff[sin_idx])},
                {"parameter": f"amplitude_h{idx + 1}", "value": float(np.hypot(coeff[cos_idx], coeff[sin_idx]))},
            ])
        formula = "y(t) = C + sum_k [a_k cos(k nu(t,e)) + b_k sin(k nu(t,e))], with T0 as periastron epoch"
        extrema = sampled_curve_extrema(model_phase, model_flux, "model maximum")

    summary = {
        **summary,
        "period": period,
        "T0": t0,
        "AIC": aic,
        "BIC": bic,
        "n_parameters": n_parameters,
    }
    return {
        "family": "binary",
        "model_kind": model_kind,
        "period": period,
        "t0": t0,
        "bins": n_bins,
        "phase": centers.tolist(),
        "flux": means.tolist(),
        "error": errors.tolist(),
        "counts": counts.tolist(),
        "data_phase": phase.tolist(),
        "data_time": t.tolist(),
        "data_flux": y.tolist(),
        "data_error": dy.tolist(),
        "model_at_data": model_at_data.tolist(),
        "model_phase": model_phase.tolist(),
        "model_flux": model_flux.tolist(),
        "model_time": model_time.tolist(),
        "model_time_flux": model_time_flux.tolist(),
        "extrema": extrema,
        "parameters": params_table,
        "summary": summary,
        "formula": formula,
    }


def bondi_hoyle_proxy(
    phase: np.ndarray,
    eccentricity: float,
    wind_speed_ratio: float,
    wind_beta: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eccentricity = float(np.clip(eccentricity, 0.0, 0.9))
    wind_speed_ratio = max(float(wind_speed_ratio), 0.05)
    wind_beta = max(float(wind_beta), 0.0)
    true_anomaly = true_anomaly_from_phase(phase, eccentricity)
    separation = (1.0 - eccentricity**2) / (1.0 + eccentricity * np.cos(true_anomaly))
    separation = np.maximum(separation, 1e-4)
    orbital_speed2 = np.maximum(2.0 / separation - 1.0, 1e-4)
    wind_speed = wind_speed_ratio * np.maximum(1.0 - 0.15 / separation, 0.05) ** wind_beta
    relative_speed2 = wind_speed**2 + orbital_speed2
    density = 1.0 / (separation**2 * np.maximum(wind_speed, 1e-4))
    proxy = density / np.maximum(relative_speed2, 1e-8) ** 1.5
    proxy = proxy / np.nanmedian(proxy) - 1.0
    return proxy, separation, true_anomaly


def fit_bondi_hoyle_model(
    phase: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    fit_eccentricity: bool,
    eccentricity_guess: float,
    fit_wind_speed: bool,
    wind_speed_guess: float,
    wind_beta: float,
    include_phase_lag: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    weights = safe_weights(dy, "standard")
    sqrt_weights = np.sqrt(weights)
    eccentricity_guess = float(np.clip(eccentricity_guess, 0.0, 0.9))
    wind_speed_guess = max(float(wind_speed_guess), 0.05)
    phase_lag_guess = 0.0

    variable0 = []
    lower, upper = [], []
    if fit_eccentricity:
        variable0.append(eccentricity_guess)
        lower.append(0.0)
        upper.append(0.9)
    if fit_wind_speed:
        variable0.append(wind_speed_guess)
        lower.append(0.05)
        upper.append(20.0)
    if include_phase_lag:
        variable0.append(phase_lag_guess)
        lower.append(-0.25)
        upper.append(0.25)

    def unpack(variable_params: np.ndarray) -> tuple[float, float, float]:
        idx = 0
        eccentricity = eccentricity_guess
        wind_speed_ratio = wind_speed_guess
        phase_lag = 0.0
        if fit_eccentricity:
            eccentricity = float(variable_params[idx])
            idx += 1
        if fit_wind_speed:
            wind_speed_ratio = float(variable_params[idx])
            idx += 1
        if include_phase_lag:
            phase_lag = float(variable_params[idx])
        return eccentricity, wind_speed_ratio, phase_lag

    def linear_model_for(variable_params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
        eccentricity, wind_speed_ratio, phase_lag = unpack(variable_params)
        shifted_phase = (phase - phase_lag) % 1.0
        proxy, _, _ = bondi_hoyle_proxy(shifted_phase, eccentricity, wind_speed_ratio, wind_beta)
        design = np.column_stack([np.ones_like(proxy), proxy])
        coeff, *_ = np.linalg.lstsq(design * sqrt_weights[:, None], y * sqrt_weights, rcond=None)
        return design @ coeff, coeff, eccentricity, wind_speed_ratio, phase_lag

    if variable0:
        optimized = least_squares(
            lambda variable_params: (linear_model_for(variable_params)[0] - y) * sqrt_weights,
            np.asarray(variable0, dtype=float),
            bounds=(np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)),
            loss="soft_l1",
            max_nfev=3000,
        )
        variables = optimized.x
    else:
        variables = np.asarray([], dtype=float)
    model, coeff, eccentricity, wind_speed_ratio, phase_lag = linear_model_for(variables)
    residuals = y - model
    dof = max(1, len(y) - (2 + len(variables)))
    summary: dict[str, float | str] = {
        "method": "bondi_hoyle_toy",
        "offset": float(coeff[0]),
        "scale": float(coeff[1]),
        "eccentricity": eccentricity,
        "wind_speed_ratio": wind_speed_ratio,
        "wind_beta": wind_beta,
        "phase_lag": phase_lag,
        "rms": float(np.sqrt(np.mean(residuals**2))) if len(residuals) else None,
        "weighted_rms": float(np.sqrt(np.average(residuals**2, weights=weights))) if len(residuals) else None,
        "chi2_red": float(np.sum(weights * residuals**2) / dof) if len(residuals) else None,
        "n_points": int(len(y)),
    }
    return model, coeff, summary


def physical_wind_to_ratio(
    period: float,
    period_unit: str,
    eccentricity: float,
    v_inf_km_s: float,
    donor_mass_msun: float,
    compact_mass_msun: float,
    donor_radius_rsun: float,
) -> dict[str, float]:
    if period <= 0.0:
        raise ValueError("Bondi-Hoyle period must be positive")
    period_seconds = period * (DAY_SI if str(period_unit).lower().startswith("d") else 1.0)
    total_mass = (donor_mass_msun + compact_mass_msun) * M_SUN_SI
    if total_mass <= 0.0:
        raise ValueError("Total mass must be positive")
    semi_major_axis_m = (G_SI * total_mass * (period_seconds / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
    orbital_speed_m_s = 2.0 * np.pi * semi_major_axis_m / period_seconds
    v_inf_m_s = max(v_inf_km_s, 1e-6) * 1000.0
    wind_speed_ratio = v_inf_m_s / max(orbital_speed_m_s, 1e-12)
    periastron_m = semi_major_axis_m * (1.0 - float(np.clip(eccentricity, 0.0, 0.95)))
    donor_radius_m = donor_radius_rsun * R_SUN_SI
    return {
        "wind_speed_ratio": float(wind_speed_ratio),
        "v_inf_km_s": float(v_inf_km_s),
        "donor_mass_msun": float(donor_mass_msun),
        "compact_mass_msun": float(compact_mass_msun),
        "total_mass_msun": float(donor_mass_msun + compact_mass_msun),
        "donor_radius_rsun": float(donor_radius_rsun),
        "semi_major_axis_rsun": float(semi_major_axis_m / R_SUN_SI),
        "orbital_speed_km_s": float(orbital_speed_m_s / 1000.0),
        "periastron_distance_rsun": float(periastron_m / R_SUN_SI),
        "radius_periastron_fraction": float(donor_radius_m / periastron_m) if periastron_m > 0.0 else float("nan"),
    }


def bondi_hoyle_model_lab(result: dict, fields: dict[str, str]) -> dict:
    series = result.get("series", {})
    t = np.asarray(series.get("time", []), dtype=float)
    y = np.asarray(series.get("flux", []), dtype=float)
    dy = np.asarray(series.get("error", []), dtype=float)
    if len(t) < 8 or len(y) != len(t):
        raise ValueError("Bondi-Hoyle model lab requires a completed analysis with at least 8 data points")
    if len(dy) != len(t):
        dy = np.ones_like(y)

    period_raw = str(fields.get("model_lab_bh_period", fields.get("model_lab_period", ""))).strip()
    default_period = result.get("folded_period") or result.get("primary_period")
    if period_raw:
        period = float(period_raw)
    elif default_period is not None:
        period = float(default_period)
    else:
        raise ValueError("Choose a Bondi-Hoyle orbital period or run an analysis with a primary period")
    if not np.isfinite(period) or period <= 0.0:
        raise ValueError("Bondi-Hoyle period must be positive")
    t0_raw = str(fields.get("model_lab_bh_t0", fields.get("model_lab_t0", ""))).strip()
    t0 = float(t0_raw) if t0_raw else float(result.get("t0", t[0]))
    n_bins = max(4, int(fields.get("model_lab_bh_bins", fields.get("fold_bins", "24"))))
    eccentricity_guess = float(fields.get("model_lab_bh_eccentricity", "0.3"))
    wind_input_mode = fields.get("model_lab_bh_wind_input_mode", "ratio").strip().lower()
    physical_wind: dict[str, float] = {}
    if wind_input_mode in {"v_inf", "vinf", "physical"}:
        physical_wind = physical_wind_to_ratio(
            period,
            result.get("baseline_unit", result.get("period_unit", "d")),
            eccentricity_guess,
            float(fields.get("model_lab_bh_vinf", "1000.0")),
            float(fields.get("model_lab_bh_donor_mass", "18.0")),
            float(fields.get("model_lab_bh_compact_mass", "1.4")),
            float(fields.get("model_lab_bh_donor_radius", "8.0")),
        )
        wind_speed_guess = physical_wind["wind_speed_ratio"]
    else:
        wind_input_mode = "ratio"
        wind_speed_guess = float(fields.get("model_lab_bh_wind_speed_ratio", "3.0"))
    wind_beta = float(fields.get("model_lab_bh_wind_beta", "0.8"))
    fit_eccentricity = str(fields.get("model_lab_bh_fit_eccentricity", "true")).strip().lower() in {"1", "true", "yes", "on"}
    fit_wind_speed = str(fields.get("model_lab_bh_fit_wind_speed", "true")).strip().lower() in {"1", "true", "yes", "on"} and wind_input_mode == "ratio"
    include_phase_lag = str(fields.get("model_lab_bh_phase_lag", "true")).strip().lower() in {"1", "true", "yes", "on"}

    phase = ((t - t0) / period) % 1.0
    centers, means, errors, counts = phase_binned_profile(phase, y, dy, n_bins)
    model_at_data, coeff, summary = fit_bondi_hoyle_model(
        phase,
        y,
        dy,
        fit_eccentricity,
        eccentricity_guess,
        fit_wind_speed,
        wind_speed_guess,
        wind_beta,
        include_phase_lag,
    )
    residuals = y - model_at_data
    n_parameters = 2 + int(fit_eccentricity) + int(fit_wind_speed) + int(include_phase_lag)
    aic, bic = information_criteria(residuals, n_parameters)
    model_phase = np.linspace(0.0, 2.0, 1600)
    phase_lag = float(summary["phase_lag"])
    proxy_phase, separation_phase, true_anomaly_phase = bondi_hoyle_proxy(
        (model_phase % 1.0 - phase_lag) % 1.0,
        float(summary["eccentricity"]),
        float(summary["wind_speed_ratio"]),
        wind_beta,
    )
    model_flux = float(coeff[0]) + float(coeff[1]) * proxy_phase
    model_time = np.linspace(float(np.nanmin(t)), float(np.nanmax(t)), 2500)
    model_time_phase = ((model_time - t0) / period) % 1.0
    proxy_time, _, _ = bondi_hoyle_proxy(
        (model_time_phase - phase_lag) % 1.0,
        float(summary["eccentricity"]),
        float(summary["wind_speed_ratio"]),
        wind_beta,
    )
    model_time_flux = float(coeff[0]) + float(coeff[1]) * proxy_time
    extrema = sampled_curve_extrema(model_phase, model_flux, "BHL accretion maximum")
    if len(extrema) == 0 and len(model_flux):
        idx = int(np.nanargmax(model_flux))
        extrema = [{"phase": float(model_phase[idx] % 1.0), "flux": float(model_flux[idx]), "kind": "BHL accretion maximum"}]

    summary = {
        **summary,
        "period": period,
        "T0": t0,
        "wind_input_mode": wind_input_mode,
        "AIC": aic,
        "BIC": bic,
        "n_parameters": n_parameters,
        **physical_wind,
    }
    parameters = [
        {"parameter": "offset", "value": float(coeff[0])},
        {"parameter": "bhl_scale", "value": float(coeff[1])},
        {"parameter": "eccentricity", "value": float(summary["eccentricity"])},
        {"parameter": "wind_speed_ratio_vwind_vorb", "value": float(summary["wind_speed_ratio"])},
        {"parameter": "wind_beta", "value": wind_beta},
        {"parameter": "phase_lag", "value": phase_lag},
    ]
    if physical_wind:
        parameters.extend([
            {"parameter": "v_inf_km_s", "value": physical_wind["v_inf_km_s"]},
            {"parameter": "donor_mass_msun", "value": physical_wind["donor_mass_msun"]},
            {"parameter": "compact_mass_msun", "value": physical_wind["compact_mass_msun"]},
            {"parameter": "donor_radius_rsun", "value": physical_wind["donor_radius_rsun"]},
            {"parameter": "semi_major_axis_rsun", "value": physical_wind["semi_major_axis_rsun"]},
            {"parameter": "orbital_speed_km_s", "value": physical_wind["orbital_speed_km_s"]},
            {"parameter": "periastron_distance_rsun", "value": physical_wind["periastron_distance_rsun"]},
            {"parameter": "radius_periastron_fraction", "value": physical_wind["radius_periastron_fraction"]},
        ])
    return {
        "family": "bondi_hoyle",
        "period": period,
        "t0": t0,
        "bins": n_bins,
        "phase": centers.tolist(),
        "flux": means.tolist(),
        "error": errors.tolist(),
        "counts": counts.tolist(),
        "data_phase": phase.tolist(),
        "data_time": t.tolist(),
        "data_flux": y.tolist(),
        "data_error": dy.tolist(),
        "model_at_data": model_at_data.tolist(),
        "model_phase": model_phase.tolist(),
        "model_flux": model_flux.tolist(),
        "model_time": model_time.tolist(),
        "model_time_flux": model_time_flux.tolist(),
        "proxy_phase": proxy_phase.tolist(),
        "separation_phase": separation_phase.tolist(),
        "true_anomaly_phase": true_anomaly_phase.tolist(),
        "extrema": extrema,
        "parameters": parameters,
        "summary": summary,
        "formula": "y(t) = C + A * [rho(r) / v_rel^3], rho ~ 1/(r^2 v_w), v_rel^2 ~ v_w^2 + v_orb^2; toy wind-fed accretion model",
    }


def epoch_folding_statistic(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    periods: np.ndarray,
    n_bins: int,
    t0: float,
) -> np.ndarray:
    weights = safe_weights(dy, "standard")
    global_mean = float(np.average(y, weights=weights))
    stats = np.full_like(periods, np.nan, dtype=float)
    for idx, period in enumerate(periods):
        phase = ((t - t0) / period) % 1.0
        centers, means, errors, counts = phase_binned_profile(phase, y, dy, n_bins)
        good = np.isfinite(means) & np.isfinite(errors) & (errors > 0.0) & (counts > 0)
        if good.sum() < 3:
            continue
        stats[idx] = float(np.sum(((means[good] - global_mean) / errors[good]) ** 2))
    return stats


def interp_periodic_template(phase: np.ndarray, template_phase: np.ndarray, template_flux: np.ndarray) -> np.ndarray:
    template_phase = np.asarray(template_phase, dtype=float) % 1.0
    template_flux = np.asarray(template_flux, dtype=float)
    order = np.argsort(template_phase)
    ph = template_phase[order]
    fl = template_flux[order]
    ph_ext = np.concatenate([ph - 1.0, ph, ph + 1.0])
    fl_ext = np.concatenate([fl, fl, fl])
    return np.interp(phase % 1.0, ph_ext, fl_ext)


def estimate_pulse_arrivals(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    period: float,
    t0: float,
    coeff: np.ndarray,
    harmonic_ratios: list[float],
    n_segments: int,
    min_points: int,
) -> tuple[list[dict[str, float | int]], np.ndarray, np.ndarray]:
    n_segments = max(2, int(n_segments))
    min_points = max(4, int(min_points))
    edges = np.linspace(float(np.nanmin(t)), float(np.nanmax(t)), n_segments + 1)
    template_phase = np.linspace(0.0, 1.0, 1500, endpoint=False)
    template_flux = sinusoid_design(template_phase, harmonic_ratios) @ coeff
    template_flux = template_flux - float(np.nanmean(template_flux))
    rows: list[dict[str, float | int]] = []
    raw_shifts: list[float] = []
    centers_for_unwrap: list[float] = []

    for seg_idx, (low, high) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        if seg_idx == n_segments:
            mask = (t >= low) & (t <= high)
        else:
            mask = (t >= low) & (t < high)
        if int(mask.sum()) < min_points:
            continue
        seg_t = t[mask]
        seg_y = y[mask]
        seg_dy = dy[mask]
        phase = ((seg_t - t0) / period) % 1.0
        weights = safe_weights(seg_dy, "standard")
        sqrt_weights = np.sqrt(weights)

        def shifted_score(shift: float) -> float:
            template = interp_periodic_template((phase - shift) % 1.0, template_phase, template_flux)
            design = np.column_stack([np.ones_like(template), template])
            coeff_seg, *_ = np.linalg.lstsq(design * sqrt_weights[:, None], seg_y * sqrt_weights, rcond=None)
            residual = seg_y - design @ coeff_seg
            return float(np.sum(weights * residual**2))

        coarse = np.linspace(-0.5, 0.5, 361)
        scores = np.asarray([shifted_score(value) for value in coarse], dtype=float)
        best_idx = int(np.nanargmin(scores))
        step = float(coarse[1] - coarse[0])
        left = max(-0.5, float(coarse[best_idx] - 2.5 * step))
        right = min(0.5, float(coarse[best_idx] + 2.5 * step))
        refined = minimize_scalar(shifted_score, bounds=(left, right), method="bounded")
        shift = float(refined.x)
        chi2 = float(refined.fun)
        dense = np.linspace(max(-0.5, shift - 0.08), min(0.5, shift + 0.08), 321)
        dense_scores = np.asarray([shifted_score(value) for value in dense], dtype=float)
        inside = dense[dense_scores <= chi2 + 1.0]
        if len(inside) >= 2:
            shift_error = 0.5 * float(inside.max() - inside.min())
        else:
            shift_error = float("nan")
        center = 0.5 * float(low + high)
        raw_shifts.append(shift)
        centers_for_unwrap.append(center)
        rows.append(
            {
                "segment": seg_idx,
                "time": center,
                "time_start": float(low),
                "time_end": float(high),
                "n_points": int(mask.sum()),
                "phase_shift": shift,
                "phase_shift_error": shift_error if np.isfinite(shift_error) else None,
                "arrival_time": center + shift * period,
                "arrival_time_error": shift_error * period if np.isfinite(shift_error) else None,
                "chi2": chi2,
            }
        )

    if not rows:
        return rows, template_phase, template_flux

    unwrapped = np.unwrap(2.0 * np.pi * np.asarray(raw_shifts, dtype=float)) / (2.0 * np.pi)
    reference_shift = float(np.nanmedian(unwrapped))
    for row, unwrapped_shift in zip(rows, unwrapped):
        row["phase_shift_unwrapped"] = float(unwrapped_shift)
        row["oc_time"] = float((unwrapped_shift - reference_shift) * period)
    return rows, template_phase, template_flux


def fit_oc_sinusoid(
    arrivals: list[dict[str, float | int]],
    orbital_period: float | None,
    t_ref: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | str]], dict[str, float | str], str] | None:
    if orbital_period is None or not np.isfinite(orbital_period) or orbital_period <= 0.0 or len(arrivals) < 4:
        return None
    time = np.asarray([float(row["time"]) for row in arrivals], dtype=float)
    oc = np.asarray([float(row["oc_time"]) for row in arrivals], dtype=float)
    err = np.asarray([
        float(row["arrival_time_error"]) if row.get("arrival_time_error") is not None else np.nan
        for row in arrivals
    ], dtype=float)
    if not np.any(np.isfinite(err) & (err > 0.0)):
        err = np.ones_like(oc)
    else:
        fallback = float(np.nanmedian(err[np.isfinite(err) & (err > 0.0)]))
        err = np.where(np.isfinite(err) & (err > 0.0), err, fallback)
    frequency = 1.0 / orbital_period
    model_at_arrivals, coeff, summary = fit_sinusoids(time - t_ref, oc, err, [frequency], method="standard")
    model_time = np.linspace(float(np.nanmin(time)), float(np.nanmax(time)), 1000)
    model_oc = sinusoid_design(model_time - t_ref, [frequency]) @ coeff
    amplitude = float(np.hypot(coeff[1], coeff[2]))
    phase_of_max = float((np.arctan2(coeff[2], coeff[1]) / (2.0 * np.pi)) % 1.0)
    parameters = [
        {"parameter": "oc_offset", "value": float(coeff[0])},
        {"parameter": "oc_cos_coeff", "value": float(coeff[1])},
        {"parameter": "oc_sin_coeff", "value": float(coeff[2])},
        {"parameter": "oc_amplitude_time", "value": amplitude},
        {"parameter": "projected_a_sini_proxy", "value": amplitude},
        {"parameter": "orbital_phase_of_max_delay", "value": phase_of_max},
        {"parameter": "orbital_period", "value": orbital_period},
    ]
    residuals = oc - model_at_arrivals
    aic, bic = information_criteria(residuals, len(coeff))
    summary = {
        **summary,
        "orbital_period": orbital_period,
        "oc_amplitude_time": amplitude,
        "phase_of_max_delay": phase_of_max,
        "AIC": aic,
        "BIC": bic,
    }
    formula = "O-C(t) = C + A_c cos(2 pi (t-Tref)/P_orb) + A_s sin(2 pi (t-Tref)/P_orb)"
    return model_time, model_oc, parameters, summary, formula


def pulse_period_model_lab(result: dict, fields: dict[str, str]) -> dict:
    series = result.get("series", {})
    t = np.asarray(series.get("time", []), dtype=float)
    y = np.asarray(series.get("flux", []), dtype=float)
    dy = np.asarray(series.get("error", []), dtype=float)
    if len(t) < 12 or len(y) != len(t):
        raise ValueError("Pulse period analysis requires a completed analysis with at least 12 data points")
    if len(dy) != len(t):
        dy = np.ones_like(y)

    period_raw = str(fields.get("model_lab_pulse_period", fields.get("model_lab_period", ""))).strip()
    default_period = result.get("folded_period") or result.get("primary_period")
    if period_raw:
        period_guess = float(period_raw)
    elif default_period is not None:
        period_guess = float(default_period)
    else:
        raise ValueError("Choose a pulse period or run an analysis with a primary period")
    if not np.isfinite(period_guess) or period_guess <= 0.0:
        raise ValueError("Pulse period must be positive")
    t0_raw = str(fields.get("model_lab_pulse_t0", fields.get("model_lab_t0", ""))).strip()
    t0 = float(t0_raw) if t0_raw else float(result.get("t0", t[0]))
    n_bins = max(4, int(fields.get("model_lab_pulse_bins", fields.get("fold_bins", "24"))))
    epoch_bins = max(4, int(fields.get("model_lab_pulse_epoch_bins", str(n_bins))))
    n_trials = max(30, int(fields.get("model_lab_pulse_trials", "300")))
    search_width = max(0.0, float(fields.get("model_lab_pulse_search_width", "2.0"))) / 100.0
    n_harmonics = max(1, min(20, int(fields.get("model_lab_pulse_harmonics", "3"))))
    n_segments = max(2, int(fields.get("model_lab_pulse_segments", "8")))
    min_points = max(4, int(fields.get("model_lab_pulse_min_points", "20")))
    fit_method = normalize_fit_method(fields.get("model_lab_pulse_fit_method", fields.get("model_fit_method", "standard")))
    orbital_period = optional_float(fields.get("model_lab_pulse_orbital_period"))

    if search_width > 0.0:
        low = max(period_guess * (1.0 - search_width), np.finfo(float).eps)
        high = period_guess * (1.0 + search_width)
        trial_periods = np.linspace(low, high, n_trials)
    else:
        trial_periods = np.asarray([period_guess], dtype=float)
    epoch_power = epoch_folding_statistic(t, y, dy, trial_periods, epoch_bins, t0)
    best_idx = int(np.nanargmax(epoch_power)) if np.any(np.isfinite(epoch_power)) else 0
    best_period = float(trial_periods[best_idx])
    best_epoch_power = float(epoch_power[best_idx]) if np.isfinite(epoch_power[best_idx]) else None

    phase = ((t - t0) / best_period) % 1.0
    centers, means, errors, counts = phase_binned_profile(phase, y, dy, n_bins)
    harmonic_ratios = [float(order) for order in range(1, n_harmonics + 1)]
    model_at_data, coeff, summary = fit_sinusoids(phase, y, dy, harmonic_ratios, method=fit_method)
    residuals = y - model_at_data
    aic, bic = information_criteria(residuals, len(coeff))
    model_phase = np.linspace(0.0, 2.0, 1600)
    model_flux = sinusoid_design(model_phase % 1.0, harmonic_ratios) @ coeff
    model_time = np.linspace(float(np.nanmin(t)), float(np.nanmax(t)), 2500)
    model_time_phase = ((model_time - t0) / best_period) % 1.0
    model_time_flux = sinusoid_design(model_time_phase, harmonic_ratios) @ coeff
    maxima = periodic_model_maxima(harmonic_ratios, coeff)
    profile_min = float(np.nanmin(model_flux))
    profile_max = float(np.nanmax(model_flux))
    profile_mean = float(coeff[0])
    peak_to_peak_pf = float((profile_max - profile_min) / (profile_max + profile_min)) if abs(profile_max + profile_min) > 1e-12 else None
    rms_pf = float(
        np.sqrt(np.sum([coeff[1 + 2 * idx] ** 2 + coeff[2 + 2 * idx] ** 2 for idx in range(n_harmonics)]) / 2.0) / abs(profile_mean)
    ) if abs(profile_mean) > 1e-12 else None
    arrivals, template_phase, template_flux = estimate_pulse_arrivals(
        t,
        y,
        dy,
        best_period,
        t0,
        coeff,
        harmonic_ratios,
        n_segments,
        min_points,
    )
    oc_fit = fit_oc_sinusoid(arrivals, orbital_period, t0)
    if oc_fit is None:
        oc_model_time = np.asarray([], dtype=float)
        oc_model = np.asarray([], dtype=float)
        oc_parameters: list[dict[str, float | str]] = []
        oc_summary: dict[str, float | str] = {}
        oc_formula = ""
    else:
        oc_model_time, oc_model, oc_parameters, oc_summary, oc_formula = oc_fit

    terms = []
    for idx, ratio in enumerate(harmonic_ratios):
        cos_idx = 1 + 2 * idx
        sin_idx = cos_idx + 1
        cos_coeff = float(coeff[cos_idx])
        sin_coeff = float(coeff[sin_idx])
        terms.append(
            {
                "component": idx + 1,
                "frequency_ratio": ratio,
                "period": best_period / ratio,
                "cos_coeff": cos_coeff,
                "sin_coeff": float(coeff[sin_idx]),
                "amplitude": float(np.hypot(cos_coeff, sin_coeff)),
                "phase_of_max": float((np.arctan2(coeff[sin_idx], coeff[cos_idx]) / (2.0 * np.pi * ratio)) % (1.0 / ratio)),
            }
        )
    summary = {
        **summary,
        "period_guess": period_guess,
        "best_period": best_period,
        "T0": t0,
        "epoch_folding_statistic": best_epoch_power,
        "peak_to_peak_pulsed_fraction": peak_to_peak_pf,
        "rms_pulsed_fraction": rms_pf,
        "AIC": aic,
        "BIC": bic,
        "n_parameters": len(coeff),
        "n_arrivals": len(arrivals),
    }
    return {
        "family": "pulse",
        "period": best_period,
        "period_guess": period_guess,
        "t0": t0,
        "bins": n_bins,
        "phase": centers.tolist(),
        "flux": means.tolist(),
        "error": errors.tolist(),
        "counts": counts.tolist(),
        "data_phase": phase.tolist(),
        "data_time": t.tolist(),
        "data_flux": y.tolist(),
        "data_error": dy.tolist(),
        "model_at_data": model_at_data.tolist(),
        "model_phase": model_phase.tolist(),
        "model_flux": model_flux.tolist(),
        "model_time": model_time.tolist(),
        "model_time_flux": model_time_flux.tolist(),
        "maxima": [{"phase": ph, "flux": val} for ph, val in maxima],
        "terms": terms,
        "summary": summary,
        "trial_period": trial_periods.tolist(),
        "epoch_power": epoch_power.tolist(),
        "template_phase": template_phase.tolist(),
        "template_flux": template_flux.tolist(),
        "arrivals": arrivals,
        "oc_model_time": oc_model_time.tolist(),
        "oc_model": oc_model.tolist(),
        "oc_parameters": oc_parameters,
        "oc_summary": oc_summary,
        "formula": "y(phi) = C + sum_k [a_k cos(2 pi k phi) + b_k sin(2 pi k phi)]",
        "oc_formula": oc_formula,
    }


def parse_period_list(text: str) -> list[float]:
    periods: list[float] = []
    for item in re.split(r"[,;\\s]+", text.strip()):
        if not item:
            continue
        period = float(item)
        if period <= 0:
            raise ValueError("Periods must be positive")
        periods.append(period)
    return periods


def period_is_excluded(period: float, excluded_periods: list[float], tolerance: float) -> bool:
    return any(abs(period - old) <= tolerance * max(period, old) for old in excluded_periods)


def apply_manual_exclusions(
    peaks: list[PeakSummary],
    excluded_periods: list[float],
    tolerance: float,
) -> None:
    if not excluded_periods:
        return
    for peak in peaks:
        if period_is_excluded(peak.period, excluded_periods, tolerance):
            peak.kind = "manually excluded"


def usable_candidate_peaks(peaks: list[PeakSummary]) -> list[PeakSummary]:
    return [
        peak for peak in peaks
        if "artefact" not in peak.kind and "excluded" not in peak.kind
    ]


def folded_configuration(primary_period: float | None, fields: dict[str, str]) -> tuple[float, list[float], list[float]]:
    mode = fields.get("fold_fit_mode", "harmonics")
    if mode == "selected":
        periods = parse_period_list(fields.get("fold_fit_periods", ""))
        if not periods:
            if primary_period is None:
                raise ValueError("Selected folded periods are required when no primary period was found")
            periods = [primary_period]
        folded_period = periods[0]
        ratios = [folded_period / period for period in periods]
        return folded_period, ratios, periods

    if primary_period is None:
        raise ValueError("No primary period is available for harmonic folded fitting")
    n_harmonics = int(fields.get("fold_fit_harmonics", "2"))
    if n_harmonics < 1:
        raise ValueError("Number of folded-fit harmonics must be at least 1")
    folded_period = primary_period
    ratios = [float(k) for k in range(1, n_harmonics + 1)]
    periods = [primary_period / k for k in range(1, n_harmonics + 1)]
    return folded_period, ratios, periods


def fig_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def make_periodogram_plot(frequency, power, peaks, title) -> str:
    period = 1.0 / frequency
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(period, power, color="0.1", lw=0.8)
    for peak in peaks:
        if "excluded" in peak.kind:
            color = "0.45"
        elif "artefact" in peak.kind:
            color = "tab:red"
        else:
            color = "tab:blue"
        ax.axvline(peak.period, color=color, ls="--", lw=1.0)
        ax.scatter([peak.period], [peak.power], color=color, s=12)
        ax.text(peak.period, peak.power, f" {peak.period:.2f} d", color=color, fontsize=9, va="bottom")
    ax.set_xlabel("Period [d]")
    ax.set_ylabel("Lomb-Scargle power")
    ax.set_title(title)
    ax.set_xlim(max(1.0 / frequency.max(), 0.0), 1.0 / frequency.min())
    return fig_to_data_uri(fig)


def make_window_plot(frequency, window_power, window_peaks) -> str:
    period = 1.0 / frequency
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(period, window_power, color="0.15", lw=0.8)
    for peak in window_peaks:
        ax.axvline(peak["period"], color="tab:red", ls="--", lw=0.8, alpha=0.35)
    ax.set_xlabel("Period [d]")
    ax.set_ylabel("Sampling-window power")
    ax.set_title("Sampling window")
    ax.set_xlim(max(1.0 / frequency.max(), 0.0), 1.0 / frequency.min())
    return fig_to_data_uri(fig)


def make_folded_plot(folded) -> str:
    phase = folded["phase"]
    flux = folded["flux"]
    error = folded["error"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.errorbar(np.r_[phase, phase + 1], np.r_[flux, flux], yerr=np.r_[error, error], fmt="o", color="0.15", ecolor="0.55", capsize=2)
    ax.plot(folded["model_phase"], folded["model_flux"], color="tab:blue", ls=":", lw=2.0)
    for ph, val in folded["maxima"]:
        for cyc in [0, 1]:
            ax.axvline(ph + cyc, color="tab:blue", ls="--", lw=1)
            ax.scatter([ph + cyc], [val], color="tab:blue", s=18)
    ax.set_xlim(0, 2)
    ax.set_xlabel("Orbital phase")
    ax.set_ylabel("Weighted mean flux")
    ax.set_title("Folded light curve")
    return fig_to_data_uri(fig)


def empty_folded_payload() -> dict:
    return {
        "phase": np.asarray([], dtype=float),
        "flux": np.asarray([], dtype=float),
        "error": np.asarray([], dtype=float),
        "model_phase": np.asarray([], dtype=float),
        "model_flux": np.asarray([], dtype=float),
        "maxima": [],
        "fit_summary": {},
        "fit_terms": [],
    }


def empty_analysis_result(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    freq: np.ndarray,
    power: np.ndarray,
    win: np.ndarray,
    window_peaks: list[dict[str, float]],
    message: str,
    fields: dict[str, str] | None = None,
    t0: float | None = None,
    has_error_column: bool = True,
    labels: dict[str, str] | None = None,
) -> dict:
    labels = labels or unit_labels("days")
    flux_is_magnitude = str((fields or {}).get("flux_is_magnitude", "")).strip().lower() in {"1", "true", "yes", "on"}
    period = 1.0 / freq
    residual_power = np.zeros_like(power)
    folded = empty_folded_payload()
    folded_period = None
    fit_periods: list[float] = []
    fit_ratios: list[float] = []
    if fields is not None:
        try:
            fold_bins = int(fields.get("fold_bins", "10"))
            fit_method = normalize_fit_method(fields.get("model_fit_method", "standard"))
            folded_period, fit_ratios, fit_periods = folded_configuration(None, fields)
            folded = folded_profile(t, y, dy, folded_period, float(t0 if t0 is not None else t[0]), fold_bins, fit_ratios, fit_method)
        except ValueError:
            folded = empty_folded_payload()
    return {
        "analysis_message": message,
        "n_points": len(t),
        "baseline": float(t.max() - t.min()),
        "primary_period": None,
        "folded_period": folded_period,
        "t0": float(t0 if t0 is not None else t[0]),
        "has_error_column": has_error_column,
        "flux_is_magnitude": flux_is_magnitude,
        **labels,
        "excluded_periods": [],
        "exclusion_tolerance": None,
        "has_prewhitening": False,
        "prewhiten_periods": [],
        "prewhitening_terms": [],
        "window_peaks": window_peaks,
        "peaks": [],
        "residual_peaks": [],
        "folded_maxima": [{"phase": ph, "flux": val} for ph, val in folded["maxima"]],
        "fold_fit_periods": fit_periods,
        "fold_fit_ratios": fit_ratios,
        "fold_fit_terms": folded.get("fit_terms", []),
        "fold_fit_summary": folded.get("fit_summary", {}),
        "series": {
            "period": period.tolist(),
            "frequency": freq.tolist(),
            "power": power.tolist(),
            "window_power": win.tolist(),
            "residual_power": residual_power.tolist(),
            "time": t.tolist(),
            "flux": y.tolist(),
            "error": dy.tolist(),
            "prewhitening_model_flux": y.tolist(),
            "fold_phase": folded["phase"].tolist(),
            "fold_flux": folded["flux"].tolist(),
            "fold_error": folded["error"].tolist(),
            "fold_model_phase": folded["model_phase"].tolist(),
            "fold_model_flux": folded["model_flux"].tolist(),
        },
        "plots": {
            "periodogram": make_periodogram_plot(freq, power, [], "Lomb-Scargle periodogram"),
            "window": make_window_plot(freq, win, window_peaks),
            "prewhitened": make_periodogram_plot(freq, residual_power, [], "After prewhitening"),
            "folded": make_folded_plot(folded),
        },
    }


def run_analysis(fields: dict[str, str], file_bytes: bytes, filename: str = "uploaded.dat") -> dict:
    time_col = int(fields.get("time_col", "1")) - 1
    flux_col = int(fields.get("flux_col", "2")) - 1
    error_col_raw = fields.get("error_col", "3").strip().lower()
    error_col = None if error_col_raw in {"", "none", "0", "no"} else int(error_col_raw) - 1
    fits_extension_raw = fields.get("fits_extension", "").strip()
    fits_extension = int(fits_extension_raw) if fits_extension_raw else None
    fits_time_col = fields.get("fits_time_col", "").strip() or None
    fits_flux_col = fields.get("fits_flux_col", "").strip() or None
    fits_error_col_raw = fields.get("fits_error_col", "").strip()
    fits_error_col = None if fits_error_col_raw.lower() in {"", "none", "0", "no"} else fits_error_col_raw
    fmin = float(fields.get("fmin", "0.01"))
    fmax = float(fields.get("fmax", "1.0"))
    samples_per_peak = float(fields.get("samples_per_peak", "10"))
    max_peaks = int(fields.get("max_peaks", "6"))
    min_considered_period = float(fields.get("min_considered_period", fields.get("min_marked_period", "2.0")))
    window_artifact_power = float(fields.get("window_artifact_power", "0.01"))
    window_artifact_tolerance = float(fields.get("window_artifact_tolerance", "0.01"))
    excluded_periods = parse_period_list(fields.get("excluded_periods", ""))
    exclusion_tolerance = float(fields.get("exclusion_tolerance", "0.01"))
    prewhiten_periods = parse_period_list(fields.get("prewhiten_periods", ""))
    n_bootstrap = int(fields.get("n_bootstrap", "1000"))
    bootstrap_width = float(fields.get("bootstrap_width", "0.03"))
    fold_bins = int(fields.get("fold_bins", "10"))
    labels = unit_labels(fields.get("time_unit", "days"))
    fit_method = normalize_fit_method(fields.get("model_fit_method", "standard"))
    flux_is_magnitude = str(fields.get("flux_is_magnitude", "")).strip().lower() in {"1", "true", "yes", "on"}
    t, y, dy = read_columns(
        file_bytes,
        filename,
        time_col,
        flux_col,
        error_col,
        fits_extension=fits_extension,
        fits_time_col=fits_time_col,
        fits_flux_col=fits_flux_col,
        fits_error_col=fits_error_col,
    )
    t, y, dy = apply_data_limits(t, y, dy, fields)
    has_error_column = fits_error_col is not None if is_fits_upload(filename, file_bytes) else error_col is not None
    y_offset = weighted_median(y, 1.0 / dy**2)
    y_analysis = y - y_offset
    t0_raw = fields.get("t0", "").strip()
    t0 = float(t0_raw) if t0_raw else float(t[0])

    freq = frequency_grid(t, fmin, fmax, samples_per_peak)
    power, ls, peaks = find_lomb_scargle_peaks(t, y_analysis, dy, freq, max_peaks, min_considered_period=min_considered_period)
    win = spectral_window(t, freq)
    window_peaks = sampling_window_peaks(freq, win, window_artifact_power, 0.0)
    classify_peaks(
        peaks,
        freq,
        win,
        float(t.max() - t.min()),
        min_window_period=0.0,
        window_power_threshold=window_artifact_power,
        relative_tolerance=window_artifact_tolerance,
    )
    apply_manual_exclusions(peaks, excluded_periods, exclusion_tolerance)
    candidate_peaks = usable_candidate_peaks(peaks)
    primary = candidate_peaks[0] if candidate_peaks else None
    add_harmonic_and_window_peaks(
        peaks,
        freq,
        power,
        ls,
        win,
        primary,
        min_considered_period,
        window_power_threshold=window_artifact_power,
    )
    classify_peaks(
        peaks,
        freq,
        win,
        float(t.max() - t.min()),
        min_window_period=0.0,
        window_power_threshold=window_artifact_power,
        relative_tolerance=window_artifact_tolerance,
    )
    apply_manual_exclusions(peaks, excluded_periods, exclusion_tolerance)
    peaks.sort(key=lambda peak: peak.power, reverse=True)
    add_bootstrap_errors(t, y_analysis, dy, ls, peaks, n_bootstrap, bootstrap_width, 1200, seed=12345)

    candidate_peaks = usable_candidate_peaks(peaks)
    if not candidate_peaks:
        return empty_analysis_result(
            t,
            y,
            dy,
            freq,
            power,
            win,
            window_peaks,
            "No usable candidate peak remains after sampling-window/manual filtering. "
            "Try lowering the frequency range, increasing the sampling-window artefact threshold, "
            "or removing some manual exclusions.",
            fields=fields,
            t0=t0,
            has_error_column=has_error_column,
            labels=labels,
        )
    primary = candidate_peaks[0]
    prewhiten_base_periods = prewhiten_periods
    prewhiten_terms = prewhitening_terms_from_periods(prewhiten_base_periods, peaks) if prewhiten_base_periods else []
    if prewhiten_terms:
        prewhitening_fit_method = "standard" if fit_method == "display_optimized" else fit_method
        prewhiten_model, prewhitening_table = fit_sinusoids_with_terms(
            t,
            y_analysis,
            dy,
            prewhiten_terms,
            method=prewhitening_fit_method,
        )
        for row in prewhitening_table:
            row["offset"] = float(row.get("offset", 0.0)) + y_offset
        if fit_method == "display_optimized":
            prewhiten_display_model, prewhitening_display_table = fit_sinusoids_with_terms(
                t,
                y_analysis,
                dy,
                prewhiten_terms,
                method=fit_method,
            )
            for row in prewhitening_display_table:
                row["offset"] = float(row.get("offset", 0.0)) + y_offset
        else:
            prewhiten_display_model = prewhiten_model
            prewhitening_display_table = []
        residuals = y_analysis - prewhiten_model
        residual_power, residual_ls, residual_peaks = find_lomb_scargle_peaks(
            t, residuals, dy, freq, max_peaks, min_considered_period=min_considered_period
        )
        classify_peaks(
            residual_peaks,
            freq,
            win,
            float(t.max() - t.min()),
            min_window_period=0.0,
            window_power_threshold=window_artifact_power,
            relative_tolerance=window_artifact_tolerance,
        )
        apply_manual_exclusions(residual_peaks, excluded_periods, exclusion_tolerance)
        residual_candidate_peaks = usable_candidate_peaks(residual_peaks)
        residual_primary = residual_candidate_peaks[0] if residual_candidate_peaks else None
        add_harmonic_and_window_peaks(
            residual_peaks,
            freq,
            residual_power,
            residual_ls,
            win,
            residual_primary,
            min_considered_period,
            window_power_threshold=window_artifact_power,
        )
        classify_peaks(
            residual_peaks,
            freq,
            win,
            float(t.max() - t.min()),
            min_window_period=0.0,
            window_power_threshold=window_artifact_power,
            relative_tolerance=window_artifact_tolerance,
        )
        apply_manual_exclusions(residual_peaks, excluded_periods, exclusion_tolerance)
        residual_peaks.sort(key=lambda peak: peak.power, reverse=True)
    else:
        prewhiten_model = np.zeros_like(y_analysis)
        prewhiten_display_model = prewhiten_model
        prewhitening_table = []
        prewhitening_display_table = []
        residual_power = np.zeros_like(power)
        residual_peaks = []

    folded_period, fit_ratios, fit_periods = folded_configuration(primary.period, fields)
    folded = folded_profile(t, y, dy, folded_period, t0, fold_bins, fit_ratios, fit_method)
    period = 1.0 / freq
    return {
        "n_points": len(t),
        "baseline": float(t.max() - t.min()),
        "primary_period": primary.period,
        "folded_period": folded_period,
        "t0": t0,
        "has_error_column": has_error_column,
        "flux_is_magnitude": flux_is_magnitude,
        **labels,
        "excluded_periods": excluded_periods,
        "exclusion_tolerance": exclusion_tolerance,
        "has_prewhitening": bool(prewhitening_table),
        "prewhiten_periods": prewhiten_base_periods,
        "prewhitening_terms": prewhitening_table,
        "prewhitening_display_terms": prewhitening_display_table,
        "window_peaks": window_peaks,
        "peaks": [asdict(p) for p in peaks],
        "residual_peaks": [asdict(p) for p in residual_peaks],
        "folded_maxima": [{"phase": ph, "flux": val} for ph, val in folded["maxima"]],
        "fold_fit_periods": fit_periods,
        "fold_fit_ratios": fit_ratios,
        "fold_fit_terms": folded.get("fit_terms", []),
        "fold_fit_summary": folded.get("fit_summary", {}),
        "series": {
            "period": period.tolist(),
            "frequency": freq.tolist(),
            "power": power.tolist(),
            "window_power": win.tolist(),
            "residual_power": residual_power.tolist(),
            "time": t.tolist(),
            "flux": y.tolist(),
            "error": dy.tolist(),
            "prewhitening_model_flux": (prewhiten_model + y_offset).tolist(),
            "prewhitening_display_model_flux": (prewhiten_display_model + y_offset).tolist(),
            "fold_phase": folded["phase"].tolist(),
            "fold_flux": folded["flux"].tolist(),
            "fold_error": folded["error"].tolist(),
            "fold_model_phase": folded["model_phase"].tolist(),
            "fold_model_flux": folded["model_flux"].tolist(),
        },
        "plots": {
            "periodogram": make_periodogram_plot(freq, power, peaks, "Lomb-Scargle periodogram"),
            "window": make_window_plot(freq, win, window_peaks),
            "prewhitened": make_periodogram_plot(freq, residual_power, residual_peaks, "After prewhitening"),
            "folded": make_folded_plot(folded),
        },
    }


def update_folded_profile(result: dict, fields: dict[str, str]) -> dict:
    primary_period = None if result.get("primary_period") is None else float(result["primary_period"])
    fold_bins = int(fields.get("fold_bins", "10"))
    series = result["series"]
    t = np.asarray(series["time"], dtype=float)
    y = np.asarray(series["flux"], dtype=float)
    dy = np.asarray(series["error"], dtype=float)
    t0_raw = fields.get("t0", "").strip()
    t0 = float(t0_raw) if t0_raw else float(result.get("t0", t[0]))
    fit_method = normalize_fit_method(fields.get("model_fit_method", "standard"))

    folded_period, fit_ratios, fit_periods = folded_configuration(primary_period, fields)
    folded = folded_profile(t, y, dy, folded_period, t0, fold_bins, fit_ratios, fit_method)

    updated = dict(result)
    updated_series = dict(series)
    updated_series.update(
        {
            "fold_phase": folded["phase"].tolist(),
            "fold_flux": folded["flux"].tolist(),
            "fold_error": folded["error"].tolist(),
            "fold_model_phase": folded["model_phase"].tolist(),
            "fold_model_flux": folded["model_flux"].tolist(),
        }
    )
    updated["series"] = updated_series
    updated["t0"] = t0
    updated["folded_period"] = folded_period
    updated["folded_maxima"] = [{"phase": ph, "flux": val} for ph, val in folded["maxima"]]
    updated["fold_fit_periods"] = fit_periods
    updated["fold_fit_ratios"] = fit_ratios
    updated["fold_fit_terms"] = folded.get("fit_terms", [])
    updated["fold_fit_summary"] = folded.get("fit_summary", {})
    if "plots" in updated:
        updated_plots = dict(updated["plots"])
        updated_plots["folded"] = make_folded_plot(folded)
        updated["plots"] = updated_plots
    return updated


def advanced_grid(result: dict, fields: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, int]:
    t = np.asarray(result["series"]["time"], dtype=float)
    fmin = float(fields.get("advanced_fmin", fields.get("fmin", "0.01")))
    fmax = float(fields.get("advanced_fmax", fields.get("fmax", "1.0")))
    if fmin <= 0 or fmax <= fmin:
        raise ValueError("Advanced frequency range must satisfy 0 < min frequency < max frequency")
    period_min = 1.0 / fmax
    period_max = 1.0 / fmin
    n_periods = int(fields.get("advanced_period_bins", "200"))
    if n_periods < 20:
        raise ValueError("Advanced period bins must be at least 20")
    period_axis = np.linspace(period_min, period_max, n_periods)
    frequency = 1.0 / period_axis

    baseline = float(t.max() - t.min())
    window_width = float(fields.get("advanced_window_width", max(period_max * 3.0, baseline / 8.0)))
    window_step = float(fields.get("advanced_window_step", window_width / 4.0))
    min_points = int(fields.get("advanced_min_points", "30"))
    if window_width <= 0 or window_step <= 0:
        raise ValueError("Advanced window width and step must be positive")
    if min_points < 3:
        raise ValueError("Advanced minimum points per window must be at least 3")

    start = float(t.min() + 0.5 * window_width)
    stop = float(t.max() - 0.5 * window_width)
    if stop < start:
        centers = np.asarray([0.5 * (t.min() + t.max())])
    else:
        centers = np.arange(start, stop + 0.5 * window_step, window_step)
    return period_axis, frequency, centers, window_width, window_step, min_points


def empty_advanced_result(
    method: str,
    method_label: str,
    metric: str,
    period_axis: np.ndarray,
    frequency: np.ndarray,
    window_width: float,
    window_step: float,
    min_points: int,
    message: str,
) -> dict:
    return {
        "method": method,
        "method_label": method_label,
        "metric": metric,
        "window_width": window_width,
        "window_step": window_step,
        "min_points": min_points,
        "period": period_axis.tolist(),
        "frequency": frequency.tolist(),
        "time": [],
        "values": np.empty((0, len(period_axis))).tolist(),
        "counts": [],
        "best_period": [],
        "best_value": [],
        "message": message,
    }


def sliding_lomb_scargle(result: dict, fields: dict[str, str]) -> dict:
    series = result["series"]
    t = np.asarray(series["time"], dtype=float)
    y = np.asarray(series["flux"], dtype=float)
    dy = np.asarray(series["error"], dtype=float)
    period_axis, frequency, centers, window_width, window_step, min_points = advanced_grid(result, fields)
    metric = fields.get("advanced_metric", "power")
    if metric not in {"power", "amplitude"}:
        raise ValueError("Advanced metric must be 'power' or 'amplitude'")
    track_period = optional_float(fields.get("advanced_track_period"))
    track_width_fraction = optional_float(fields.get("advanced_track_width_fraction"))
    track_mask = None
    track_min_period = None
    track_max_period = None
    if track_period is not None and track_width_fraction is not None and track_width_fraction > 0:
        if track_period <= 0:
            raise ValueError("Advanced track period must be positive")
        track_half_width = track_period * track_width_fraction
        track_min_period = track_period - track_half_width
        track_max_period = track_period + track_half_width
        track_mask = (period_axis >= track_min_period) & (period_axis <= track_max_period)
        if not np.any(track_mask):
            closest_idx = int(np.nanargmin(np.abs(period_axis - track_period)))
            track_mask = np.zeros_like(period_axis, dtype=bool)
            track_mask[closest_idx] = True

    rows = []
    valid_centers = []
    counts = []
    best_periods = []
    best_values = []
    for center in centers:
        mask = (t >= center - 0.5 * window_width) & (t <= center + 0.5 * window_width)
        n_local = int(mask.sum())
        if n_local < min_points:
            continue
        tw, yw, dyw = t[mask], y[mask], dy[mask]
        yw = yw - weighted_median(yw, 1.0 / dyw**2)
        ls = LombScargle(tw, yw, dyw, center_data=True, fit_mean=True)
        if metric == "amplitude":
            values = np.empty_like(frequency)
            for idx, freq in enumerate(frequency):
                params = ls.model_parameters(freq)
                values[idx] = float(np.hypot(params[-2], params[-1]))
        else:
            values = ls.power(frequency)
        rows.append(values)
        valid_centers.append(float(center))
        counts.append(n_local)
        if track_mask is not None:
            masked_values = np.where(track_mask, values, np.nan)
            best_idx = int(np.nanargmax(masked_values))
        else:
            best_idx = int(np.nanargmax(values))
        best_periods.append(float(period_axis[best_idx]))
        best_values.append(float(values[best_idx]))

    if rows:
        matrix = np.vstack(rows)
        message = ""
    else:
        return empty_advanced_result(
            "sliding_lomb_scargle",
            "Sliding Lomb-Scargle tomographic map",
            metric,
            period_axis,
            frequency,
            window_width,
            window_step,
            min_points,
            "No sliding windows had enough points for the selected settings.",
        )

    return {
        "method": "sliding_lomb_scargle",
        "method_label": "Sliding Lomb-Scargle tomographic map",
        "metric": metric,
        "window_width": window_width,
        "window_step": window_step,
        "min_points": min_points,
        "period": period_axis.tolist(),
        "frequency": frequency.tolist(),
        "time": valid_centers,
        "values": matrix.tolist(),
        "counts": counts,
        "best_period": best_periods,
        "best_value": best_values,
        "track_period": track_period,
        "track_min_period": track_min_period,
        "track_max_period": track_max_period,
        "message": message,
    }


def wwz_map(result: dict, fields: dict[str, str]) -> dict:
    series = result["series"]
    t = np.asarray(series["time"], dtype=float)
    y = np.asarray(series["flux"], dtype=float)
    dy = np.asarray(series["error"], dtype=float)
    period_axis, frequency, centers, window_width, window_step, min_points = advanced_grid(result, fields)
    decay = float(fields.get("advanced_wwz_decay", "0.0125"))
    if decay <= 0:
        raise ValueError("WWZ decay must be positive")

    base_weights = 1.0 / dy**2
    rows = []
    valid_centers = []
    counts = []
    best_periods = []
    best_values = []
    for center in centers:
        values = np.full_like(frequency, np.nan, dtype=float)
        effective_counts = np.zeros_like(frequency, dtype=float)
        for idx, freq in enumerate(frequency):
            omega = 2.0 * np.pi * freq
            wavelet = np.exp(-decay * (omega * (t - center)) ** 2)
            weights = base_weights * wavelet
            weight_sum = float(np.sum(weights))
            if weight_sum <= 0:
                continue
            norm_weights = weights / weight_sum
            neff = 1.0 / float(np.sum(norm_weights**2))
            effective_counts[idx] = neff
            if neff < min_points:
                continue
            phase = omega * (t - center)
            design = np.column_stack([np.ones_like(t), np.cos(phase), np.sin(phase)])
            sqrt_weights = np.sqrt(weights)
            weighted_design = design * sqrt_weights[:, None]
            weighted_y = y * sqrt_weights
            try:
                coeffs, *_ = np.linalg.lstsq(weighted_design, weighted_y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            model = design @ coeffs
            y_mean = float(np.sum(norm_weights * y))
            total_var = float(np.sum(norm_weights * (y - y_mean) ** 2))
            resid_var = float(np.sum(norm_weights * (y - model) ** 2))
            model_var = max(total_var - resid_var, 0.0)
            if resid_var <= 0:
                values[idx] = np.nan
            else:
                values[idx] = max(neff - 3.0, 0.0) * model_var / (2.0 * resid_var)
        if np.all(np.isnan(values)):
            continue
        rows.append(values)
        valid_centers.append(float(center))
        counts.append(float(np.nanmax(effective_counts)))
        best_idx = int(np.nanargmax(values))
        best_periods.append(float(period_axis[best_idx]))
        best_values.append(float(values[best_idx]))

    if not rows:
        return empty_advanced_result(
            "wwz",
            "WWZ tomographic map",
            "WWZ",
            period_axis,
            frequency,
            window_width,
            window_step,
            min_points,
            "No WWZ windows had enough effective points for the selected settings.",
        )

    return {
        "method": "wwz",
        "method_label": "WWZ tomographic map",
        "metric": "WWZ",
        "window_width": window_width,
        "window_step": window_step,
        "min_points": min_points,
        "wwz_decay": decay,
        "period": period_axis.tolist(),
        "frequency": frequency.tolist(),
        "time": valid_centers,
        "values": np.vstack(rows).tolist(),
        "counts": counts,
        "best_period": best_periods,
        "best_value": best_values,
        "message": "",
    }


def advanced_time_frequency_map(result: dict, fields: dict[str, str]) -> dict:
    method = fields.get("advanced_method", "v1")
    if method == "v2":
        return wwz_map(result, fields)
    return sliding_lomb_scargle(result, fields)


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            path = "/index.html"
        file_path = STATIC / path.lstrip("/")
        if not file_path.exists() or not file_path.is_file():
            self._send(404, "text/plain", b"Not found")
            return
        ctype = "text/html" if file_path.suffix == ".html" else "text/plain"
        if file_path.suffix == ".css":
            ctype = "text/css"
        if file_path.suffix == ".js":
            ctype = "application/javascript"
        self._send(200, ctype, file_path.read_bytes())

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/analyze":
            self._send(404, "text/plain", b"Not found")
            return
        length = int(self.headers.get("Content-Length", "0"))
        content_type = self.headers.get("Content-Type", "")
        body = self.rfile.read(length)
        message = BytesParser(policy=email_policy).parsebytes(
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode("utf-8") + body
        )
        fields: dict[str, str] = {}
        file_bytes: bytes | None = None
        upload_filename = "uploaded.dat"
        for part in message.iter_parts():
            name = part.get_param("name", header="content-disposition")
            if not name:
                continue
            payload = part.get_payload(decode=True) or b""
            part_filename = part.get_filename()
            if name == "file" and part_filename:
                file_bytes = payload
                upload_filename = part_filename
            else:
                fields[name] = payload.decode("utf-8", errors="replace")
        if file_bytes is None:
            self._send(400, "application/json", json.dumps({"error": "No file uploaded"}).encode())
            return
        try:
            result = run_analysis(fields, file_bytes, upload_filename)
            self._send(200, "application/json", json.dumps(result).encode())
        except Exception as exc:
            self._send(400, "application/json", json.dumps({"error": str(exc)}).encode())


def main() -> None:
    port = int(os.environ.get("PORT", "8765"))
    host = os.environ.get("HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Periodicity Workbench running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
