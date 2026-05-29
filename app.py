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
from scipy.optimize import minimize_scalar
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
NUMERIC_TABLE_LINE = re.compile(r"^[\s,+\-0-9.eEdD]+$")


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
        if not stripped or stripped.startswith("#"):
            continue
        if not NUMERIC_TABLE_LINE.match(stripped.replace(",", " ")):
            raise ValueError(
                f"Line {line_number} is not a numeric table row or a comment. "
                "Use '#' for comments and numeric columns for data."
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
    error_col: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    text = validate_upload(filename, raw)
    data = np.genfromtxt(io.StringIO(text.replace(",", " ")), comments="#", invalid_raise=False)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError("Could not parse the uploaded text file as a numeric table")
    max_col = data.shape[1] - 1
    for col in [time_col, flux_col, error_col]:
        if col < 0 or col > max_col:
            raise ValueError(f"Column index {col + 1} is outside the available range 1-{data.shape[1]}")
    t, y, dy = data[:, time_col], data[:, flux_col], data[:, error_col]
    good = np.isfinite(t) & np.isfinite(y) & np.isfinite(dy) & (dy > 0)
    if good.sum() < 10:
        raise ValueError("Need at least 10 valid rows with finite values and positive errors")
    t, y, dy = t[good], y[good], dy[good]
    order = np.argsort(t)
    return t[order], y[order], dy[order]


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


def classify_peaks(
    peaks: list[PeakSummary],
    frequency: np.ndarray,
    window_power: np.ndarray,
    baseline: float,
    max_window_peaks: int = 50,
) -> None:
    window_indices, _ = find_peaks(window_power)
    if len(window_indices) == 0:
        return
    ranked = window_indices[np.argsort(window_power[window_indices])[::-1]][:max_window_peaks]
    resolution = 1.0 / baseline
    tolerance = max(2.0 * np.median(np.diff(frequency)), resolution)
    for peak in peaks:
        distances = np.abs(frequency[ranked] - peak.frequency)
        nearest = int(ranked[int(np.argmin(distances))])
        if abs(frequency[nearest] - peak.frequency) <= tolerance and window_power[nearest] >= 0.01:
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
    min_marked_period: float = 2.0,
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
        if period < min_marked_period:
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
        peaks.append(PeakSummary(f"peak {n}", f, float(1.0 / f), pwr, fap, "data candidate"))
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
    return PeakSummary("local peak", f, float(1.0 / f), pwr, float(ls.false_alarm_probability(pwr)), "data candidate")


def append_unique_peak(peaks: list[PeakSummary], peak: PeakSummary | None, label: str, kind: str, tolerance: float = 0.02) -> None:
    if peak is None:
        return
    for old in peaks:
        if abs(old.period - peak.period) < tolerance * max(old.period, peak.period):
            if not label.startswith("peak"):
                old.label = label
            if kind != old.kind and "artefact" in kind:
                old.kind = kind
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
    min_marked_period: float,
    max_window_peaks: int = 4,
) -> None:
    if primary is not None:
        harmonic = local_lomb_peak(frequency, power, ls, 2.0 * primary.frequency, fractional_width=0.04)
        if harmonic is not None and harmonic.period >= min_marked_period and harmonic.fap <= 0.2:
            append_unique_peak(peaks, harmonic, "P/2 harmonic", "data candidate")

    window_indices, _ = find_peaks(window_power)
    if len(window_indices) == 0:
        return
    ranked = window_indices[np.argsort(window_power[window_indices])[::-1]]
    added = 0
    for idx in ranked:
        if window_power[idx] < 0.01:
            continue
        target_frequency = float(frequency[idx])
        target_period = float(1.0 / target_frequency)
        if target_period < min_marked_period:
            continue
        peak = local_lomb_peak(frequency, power, ls, target_frequency, fractional_width=0.025)
        if peak is None:
            continue
        peak.window_frequency = target_frequency
        peak.window_period = target_period
        peak.window_power = float(window_power[idx])
        before = len(peaks)
        append_unique_peak(peaks, peak, "sampling window", "sampling-window artefact", tolerance=0.025)
        if len(peaks) > before:
            added += 1
        if added >= max_window_peaks:
            break


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


def fit_sinusoids(t: np.ndarray, y: np.ndarray, dy: np.ndarray, frequencies: list[float]) -> tuple[np.ndarray, np.ndarray]:
    shifted = t - t.min()
    cols = [np.ones_like(t)]
    for freq in frequencies:
        phase = 2.0 * np.pi * freq * shifted
        cols.extend([np.cos(phase), np.sin(phase)])
    design = np.column_stack(cols)
    weights = 1.0 / dy**2
    coeff, *_ = np.linalg.lstsq(design * np.sqrt(weights)[:, None], y * np.sqrt(weights), rcond=None)
    return design @ coeff, coeff


def folded_profile(
    t: np.ndarray,
    y: np.ndarray,
    dy: np.ndarray,
    period: float,
    t0: float,
    n_bins: int,
    frequencies_in_phase: int,
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

    cols = [np.ones_like(centers)]
    for k in range(1, frequencies_in_phase + 1):
        cols.extend([np.cos(2.0 * np.pi * k * centers), np.sin(2.0 * np.pi * k * centers)])
    design = np.column_stack(cols)
    wbin = 1.0 / errors**2
    coeff, *_ = np.linalg.lstsq(design * np.sqrt(wbin)[:, None], means * np.sqrt(wbin), rcond=None)

    def model(ph: np.ndarray) -> np.ndarray:
        cols_model = [np.ones_like(ph)]
        for k in range(1, frequencies_in_phase + 1):
            cols_model.extend([np.cos(2.0 * np.pi * k * ph), np.sin(2.0 * np.pi * k * ph)])
        return np.column_stack(cols_model) @ coeff

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
    return {
        "phase": centers,
        "flux": means,
        "error": errors,
        "counts": counts,
        "model_phase": np.linspace(0.0, 2.0, 1000),
        "model_flux": model(np.linspace(0.0, 2.0, 1000) % 1.0),
        "maxima": maxima,
    }


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
        color = "tab:red" if "artefact" in peak.kind else "tab:blue"
        ax.axvline(peak.period, color=color, ls="--", lw=1.0)
        ax.scatter([peak.period], [peak.power], color=color, s=12)
        ax.text(peak.period, peak.power, f" {peak.period:.2f} d", color=color, fontsize=9, va="bottom")
    ax.set_xlabel("Period [d]")
    ax.set_ylabel("Lomb-Scargle power")
    ax.set_title(title)
    ax.set_xlim(max(1.0 / frequency.max(), 0.0), 1.0 / frequency.min())
    return fig_to_data_uri(fig)


def make_window_plot(frequency, window_power, peaks) -> str:
    period = 1.0 / frequency
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(period, window_power, color="0.15", lw=0.8)
    for peak in peaks:
        if peak.window_period is not None:
            ax.axvline(peak.window_period, color="tab:red", ls="--", lw=1.0)
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


def run_analysis(fields: dict[str, str], file_bytes: bytes, filename: str = "uploaded.dat") -> dict:
    time_col = int(fields.get("time_col", "1")) - 1
    flux_col = int(fields.get("flux_col", "2")) - 1
    error_col = int(fields.get("error_col", "3")) - 1
    fmin = float(fields.get("fmin", "0.01"))
    fmax = float(fields.get("fmax", "1.0"))
    samples_per_peak = float(fields.get("samples_per_peak", "10"))
    max_peaks = int(fields.get("max_peaks", "6"))
    min_marked_period = float(fields.get("min_marked_period", "2.0"))
    n_bootstrap = int(fields.get("n_bootstrap", "1000"))
    bootstrap_width = float(fields.get("bootstrap_width", "0.03"))
    include_harmonic = fields.get("include_harmonic", "on") == "on"
    fold_bins = int(fields.get("fold_bins", "10"))
    t, y, dy = read_columns(file_bytes, filename, time_col, flux_col, error_col)
    y_analysis = y - weighted_median(y, 1.0 / dy**2)
    t0_raw = fields.get("t0", "").strip()
    t0 = float(t0_raw) if t0_raw else float(t[0])

    freq = frequency_grid(t, fmin, fmax, samples_per_peak)
    power, ls, peaks = find_lomb_scargle_peaks(t, y_analysis, dy, freq, max_peaks, min_marked_period=min_marked_period)
    win = spectral_window(t, freq)
    classify_peaks(peaks, freq, win, float(t.max() - t.min()))
    candidate_peaks = [p for p in peaks if "artefact" not in p.kind]
    primary = candidate_peaks[0] if candidate_peaks else (peaks[0] if peaks else None)
    add_harmonic_and_window_peaks(peaks, freq, power, ls, win, primary, min_marked_period)
    peaks.sort(key=lambda peak: peak.power, reverse=True)
    add_bootstrap_errors(t, y_analysis, dy, ls, peaks, n_bootstrap, bootstrap_width, 1200, seed=12345)

    candidate_peaks = [p for p in peaks if "artefact" not in p.kind]
    primary = candidate_peaks[0] if candidate_peaks else peaks[0]
    remove_freqs = [primary.frequency]
    if include_harmonic:
        remove_freqs.append(2.0 * primary.frequency)
    prewhiten_model, _ = fit_sinusoids(t, y_analysis, dy, remove_freqs)
    residuals = y_analysis - prewhiten_model
    residual_power, residual_ls, residual_peaks = find_lomb_scargle_peaks(
        t, residuals, dy, freq, max_peaks, min_marked_period=min_marked_period
    )
    classify_peaks(residual_peaks, freq, win, float(t.max() - t.min()))
    residual_candidate_peaks = [p for p in residual_peaks if "artefact" not in p.kind]
    residual_primary = residual_candidate_peaks[0] if residual_candidate_peaks else (residual_peaks[0] if residual_peaks else None)
    add_harmonic_and_window_peaks(residual_peaks, freq, residual_power, residual_ls, win, residual_primary, min_marked_period)
    residual_peaks.sort(key=lambda peak: peak.power, reverse=True)

    folded = folded_profile(t, y, dy, primary.period, t0, fold_bins, 2 if include_harmonic else 1)
    return {
        "n_points": len(t),
        "baseline": float(t.max() - t.min()),
        "primary_period": primary.period,
        "t0": t0,
        "peaks": [asdict(p) for p in peaks],
        "residual_peaks": [asdict(p) for p in residual_peaks],
        "folded_maxima": [{"phase": ph, "flux": val} for ph, val in folded["maxima"]],
        "plots": {
            "periodogram": make_periodogram_plot(freq, power, peaks, "Lomb-Scargle periodogram"),
            "window": make_window_plot(freq, win, peaks),
            "prewhitened": make_periodogram_plot(freq, residual_power, residual_peaks, "After prewhitening"),
            "folded": make_folded_plot(folded),
        },
    }


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
