#!/usr/bin/env python3
from __future__ import annotations

import base64
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


ALLOWED_UPLOAD_EXTENSIONS = {"", ".txt", ".dat"}
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


def validate_upload(filename: str, raw: bytes) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ".txt, .dat, or no extension"
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


def read_columns(
    raw: bytes,
    filename: str,
    time_col: int,
    flux_col: int,
    error_col: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        data_low, data_high = np.nanpercentile(y, [5, 95])
        model_low, model_high = np.nanpercentile(model_initial, [5, 95])
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
        for order, harmonic in detected_harmonics_for_period(period, peaks):
            terms.append(
                {
                    "label": f"step {idx} detected P/{order} harmonic",
                    "frequency": harmonic.frequency,
                    "main_period": period,
                    "main_period_error": None if main_peak is None else main_peak.period_error,
                    "period_error": harmonic.period_error,
                    "frequency_error": harmonic.frequency_error,
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
    t, y, dy = read_columns(file_bytes, filename, time_col, flux_col, error_col)
    t, y, dy = apply_data_limits(t, y, dy, fields)
    has_error_column = error_col is not None
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
        prewhiten_model, prewhitening_table = fit_sinusoids_with_terms(t, y_analysis, dy, prewhiten_terms, method=fit_method)
        for row in prewhitening_table:
            row["offset"] = float(row.get("offset", 0.0)) + y_offset
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
        prewhitening_table = []
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
