"""Scientific helpers for the wind-fed Bondi-Hoyle model laboratory."""

from __future__ import annotations

import numpy as np

G_SI = 6.67430e-11
M_SUN_SI = 1.98847e30
R_SUN_SI = 6.957e8
YEAR_SI = 365.25 * 86400.0
C_SI = 299792458.0
EDDINGTON_W_PER_MSUN = 1.26e31


def true_anomaly_from_phase(phase: np.ndarray, eccentricity: float) -> np.ndarray:
    """Convert mean orbital phase to true anomaly."""
    mean_anomaly = 2.0 * np.pi * np.asarray(phase, dtype=float)
    eccentric_anomaly = mean_anomaly.copy()
    for _ in range(40):
        denominator = np.maximum(1.0 - eccentricity * np.cos(eccentric_anomaly), 1e-12)
        step = (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly) / denominator
        eccentric_anomaly -= step
        if np.nanmax(np.abs(step)) < 1e-12:
            break
    return 2.0 * np.arctan2(
        np.sqrt(1.0 + eccentricity) * np.sin(eccentric_anomaly / 2.0),
        np.sqrt(1.0 - eccentricity) * np.cos(eccentric_anomaly / 2.0),
    )


def eggleton_roche_lobe_fraction(mass_ratio: float) -> float:
    """Return R_L / separation for the star with q=M_star/M_companion."""
    q = max(float(mass_ratio), 1e-12)
    q13 = q ** (1.0 / 3.0)
    q23 = q13 * q13
    return float(0.49 * q23 / (0.6 * q23 + np.log1p(q13)))


def bhl_orbital_state(
    phase: np.ndarray,
    eccentricity: float,
    wind_terminal_ratio: float,
    wind_beta: float,
    donor_radius_over_a: float,
) -> dict[str, np.ndarray]:
    """Return dimensionless orbital and wind quantities.

    Velocities are expressed in units of v_o=2*pi*a/P and separations in units
    of the semi-major axis. The wind is radial and the accretor velocity is
    resolved into radial and tangential components.
    """
    phase = np.asarray(phase, dtype=float)
    eccentricity = float(eccentricity)
    wind_terminal_ratio = float(wind_terminal_ratio)
    wind_beta = float(wind_beta)
    donor_radius_over_a = float(donor_radius_over_a)
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError("Eccentricity must satisfy 0 <= e < 1")
    if wind_terminal_ratio <= 0.0:
        raise ValueError("Terminal wind ratio must be positive")
    if wind_beta < 0.0:
        raise ValueError("Wind beta must be non-negative")
    if donor_radius_over_a <= 0.0:
        raise ValueError("Donor radius / a must be positive")
    periastron = 1.0 - eccentricity
    if donor_radius_over_a >= periastron:
        raise ValueError(
            "Invalid wind geometry: the donor radius reaches or exceeds the periastron separation"
        )

    true_anomaly = true_anomaly_from_phase(phase, eccentricity)
    separation = (1.0 - eccentricity**2) / (1.0 + eccentricity * np.cos(true_anomaly))
    launch_factor = 1.0 - donor_radius_over_a / separation
    if np.any(launch_factor <= 0.0):
        raise ValueError("Invalid beta wind law: sampled orbit enters the wind-launching radius")
    wind_speed = wind_terminal_ratio * launch_factor**wind_beta
    root = np.sqrt(1.0 - eccentricity**2)
    orbital_radial = eccentricity * np.sin(true_anomaly) / root
    orbital_tangential = (1.0 + eccentricity * np.cos(true_anomaly)) / root
    orbital_speed2 = orbital_radial**2 + orbital_tangential**2
    relative_speed2 = (wind_speed - orbital_radial) ** 2 + orbital_tangential**2
    density_shape = 1.0 / (separation**2 * wind_speed)
    column_shape = 1.0 / (separation * wind_speed)
    return {
        "phase": phase,
        "true_anomaly": true_anomaly,
        "separation": separation,
        "wind_speed": wind_speed,
        "orbital_radial": orbital_radial,
        "orbital_tangential": orbital_tangential,
        "orbital_speed": np.sqrt(orbital_speed2),
        "relative_speed": np.sqrt(relative_speed2),
        "density_shape": density_shape,
        "column_shape": column_shape,
    }


def bhl_accretion_state(
    phase: np.ndarray,
    eccentricity: float,
    wind_terminal_ratio: float,
    wind_beta: float,
    donor_radius_over_a: float,
    *,
    sound_speed_ratio: float = 0.0,
    compact_mass_fraction: float = 0.1,
    formulation: str = "classical",
) -> dict[str, np.ndarray | str]:
    """Calculate classical or revised-binary dimensionless accretion."""
    state: dict[str, np.ndarray | str] = dict(
        bhl_orbital_state(
            phase,
            eccentricity,
            wind_terminal_ratio,
            wind_beta,
            donor_radius_over_a,
        )
    )
    sound_speed_ratio = max(float(sound_speed_ratio), 0.0)
    compact_mass_fraction = float(compact_mass_fraction)
    if not 0.0 < compact_mass_fraction < 1.0:
        raise ValueError("Compact mass fraction must satisfy 0 < q < 1")
    formulation = str(formulation).strip().lower()
    if formulation not in {"classical", "revised"}:
        raise ValueError("BHL formulation must be 'classical' or 'revised'")

    separation = np.asarray(state["separation"], dtype=float)
    wind_speed = np.asarray(state["wind_speed"], dtype=float)
    orbital_radial = np.asarray(state["orbital_radial"], dtype=float)
    relative_speed = np.asarray(state["relative_speed"], dtype=float)
    effective_speed2 = relative_speed**2 + sound_speed_ratio**2
    accretion_radius_over_a = 2.0 * compact_mass_fraction / effective_speed2
    mach = relative_speed / max(sound_speed_ratio, 1e-12)
    classical_efficiency = (
        compact_mass_fraction**2
        / (separation**2 * wind_speed * effective_speed2**1.5)
    )
    revised_efficiency = (
        0.25
        * np.abs(1.0 - orbital_radial / wind_speed)
        * (accretion_radius_over_a / separation) ** 2
    )
    selected_efficiency = classical_efficiency if formulation == "classical" else revised_efficiency
    raw_proxy = selected_efficiency / compact_mass_fraction**2
    median = float(np.nanmedian(raw_proxy))
    if not np.isfinite(median) or median <= 0.0:
        raise ValueError("BHL proxy is not finite and positive")
    state.update(
        {
            "formulation": formulation,
            "sound_speed_ratio": np.full_like(relative_speed, sound_speed_ratio),
            "effective_speed": np.sqrt(effective_speed2),
            "mach": mach,
            "accretion_radius_over_a": accretion_radius_over_a,
            "classical_efficiency": classical_efficiency,
            "revised_efficiency": revised_efficiency,
            "selected_efficiency": selected_efficiency,
            "raw_proxy": raw_proxy,
            "proxy": raw_proxy / median - 1.0,
        }
    )
    return state


def periodic_response(
    phase: np.ndarray,
    signal: np.ndarray,
    *,
    mode: str = "none",
    delay_phase: float = 0.0,
    timescale_phase: float = 0.0,
) -> np.ndarray:
    """Apply a periodic delay or causal exponential response to a sampled curve.

    The inputs may be irregularly sampled. They are interpolated onto a dense
    periodic grid, transformed, and interpolated back to the requested phases.
    """
    phase = np.asarray(phase, dtype=float) % 1.0
    signal = np.asarray(signal, dtype=float)
    if len(phase) != len(signal):
        raise ValueError("Phase and response signal must have equal lengths")
    if len(phase) == 0:
        return signal.copy()
    mode = str(mode).strip().lower()
    if mode == "none":
        return signal.copy()
    grid_size = max(2048, int(2 ** np.ceil(np.log2(max(16, len(phase) * 4)))))
    grid = np.arange(grid_size, dtype=float) / grid_size
    order = np.argsort(phase)
    sorted_phase = phase[order]
    sorted_signal = signal[order]
    unique_phase, unique_indices = np.unique(sorted_phase, return_index=True)
    unique_signal = sorted_signal[unique_indices]
    extended_phase = np.concatenate([unique_phase - 1.0, unique_phase, unique_phase + 1.0])
    extended_signal = np.tile(unique_signal, 3)
    grid_signal = np.interp(grid, extended_phase, extended_signal)

    delay_bins = int(np.round(float(delay_phase) * grid_size))
    if mode == "delay":
        transformed = np.roll(grid_signal, delay_bins)
    elif mode == "exponential":
        tau_bins = max(float(timescale_phase) * grid_size, 1e-6)
        lag_bins = np.arange(grid_size, dtype=float)
        kernel = np.exp(-lag_bins / tau_bins)
        kernel /= np.sum(kernel)
        transformed = np.fft.irfft(np.fft.rfft(grid_signal) * np.fft.rfft(kernel), n=grid_size)
        if delay_bins:
            transformed = np.roll(transformed, delay_bins)
    else:
        raise ValueError("Response mode must be none, delay, or exponential")
    return np.interp(phase, np.concatenate([grid - 1.0, grid, grid + 1.0]), np.tile(transformed, 3))


def physical_bhl_outputs(
    state: dict[str, np.ndarray | str],
    *,
    period_seconds: float,
    semi_major_axis_m: float,
    orbital_speed_m_s: float,
    compact_mass_msun: float,
    donor_mass_loss_msun_yr: float,
    compact_radius_km: float,
    radiative_efficiency: float,
    luminosity_mode: str,
    eddington_cap: bool,
) -> dict[str, np.ndarray | float | str]:
    """Convert a dimensionless BHL state into mass rate and luminosity."""
    del period_seconds  # retained in the API to make the physical scale explicit
    selected_efficiency = np.asarray(state["selected_efficiency"], dtype=float)
    wind_ratio = np.asarray(state["wind_speed"], dtype=float)
    separation = np.asarray(state["separation"], dtype=float)
    donor_mdot_si = max(float(donor_mass_loss_msun_yr), 0.0) * M_SUN_SI / YEAR_SI
    compact_mass_si = float(compact_mass_msun) * M_SUN_SI
    wind_speed_si = wind_ratio * float(orbital_speed_m_s)
    radius_si = separation * float(semi_major_axis_m)
    density_si = donor_mdot_si / (4.0 * np.pi * radius_si**2 * wind_speed_si)
    mdot_si = selected_efficiency * donor_mdot_si
    luminosity_mode = str(luminosity_mode).strip().lower()
    if luminosity_mode == "radius":
        compact_radius_m = max(float(compact_radius_km), 1e-6) * 1000.0
        luminosity_si = G_SI * compact_mass_si * mdot_si / compact_radius_m
    elif luminosity_mode == "efficiency":
        luminosity_si = max(float(radiative_efficiency), 0.0) * mdot_si * C_SI**2
    else:
        raise ValueError("Luminosity mode must be radius or efficiency")
    eddington_luminosity = EDDINGTON_W_PER_MSUN * float(compact_mass_msun)
    uncapped_luminosity = luminosity_si.copy()
    if eddington_cap:
        luminosity_si = np.minimum(luminosity_si, eddington_luminosity)
    return {
        "density_kg_m3": density_si,
        "mdot_kg_s": mdot_si,
        "mdot_msun_yr": mdot_si * YEAR_SI / M_SUN_SI,
        "luminosity_w": luminosity_si,
        "luminosity_erg_s": luminosity_si * 1e7,
        "uncapped_luminosity_w": uncapped_luminosity,
        "eddington_luminosity_w": float(eddington_luminosity),
        "eddington_ratio": uncapped_luminosity / eddington_luminosity,
        "luminosity_mode": luminosity_mode,
    }


def bhl_validity_warnings(
    state: dict[str, np.ndarray | str],
    *,
    donor_radius_over_a: float,
    donor_mass_msun: float | None = None,
    compact_mass_msun: float | None = None,
    fitted_scale: float | None = None,
    attenuation_tau: float = 0.0,
) -> tuple[list[dict[str, str]], dict[str, float | str]]:
    """Return structured assumption warnings and scalar regime diagnostics."""
    separation = np.asarray(state["separation"], dtype=float)
    wind_speed = np.asarray(state["wind_speed"], dtype=float)
    mach = np.asarray(state["mach"], dtype=float)
    racc = np.asarray(state["accretion_radius_over_a"], dtype=float)
    efficiency = np.asarray(state["selected_efficiency"], dtype=float)
    warnings: list[dict[str, str]] = []

    def add(level: str, code: str, message: str) -> None:
        warnings.append({"level": level, "code": code, "message": message})

    if np.nanmin(mach) < 1.0:
        add("outside assumptions", "subsonic_flow", "The flow becomes subsonic; the high-Mach BHL interpretation is not valid.")
    elif np.nanmin(mach) < 3.0:
        add("caution", "transonic_flow", "The minimum Mach number is below 3, so pressure effects are important.")
    if np.nanmin(wind_speed) < 1.0:
        add("caution", "slow_wind", "The local wind is slower than the orbital velocity scale; classical BHL can over-capture.")
    if np.nanmax(racc / separation) > 0.1:
        add("caution", "large_accretion_radius", "The accretion radius is a substantial fraction of the separation; gradients and turbulence may matter.")
    if np.nanmax(efficiency) > 1.0:
        add("outside assumptions", "efficiency_above_unity", "The selected prescription predicts capture efficiency above unity.")
    if fitted_scale is not None and fitted_scale < 0.0:
        add("caution", "negative_scale", "Observed flux anticorrelates with intrinsic accretion; absorption, reprocessing, or model failure is likely.")
    if attenuation_tau > 0.0:
        add("caution", "attenuation_enabled", "The observed curve includes a phenomenological attenuation model, not full radiative transfer.")

    roche_fill = float("nan")
    if donor_mass_msun is not None and compact_mass_msun is not None and compact_mass_msun > 0.0:
        roche_fraction = eggleton_roche_lobe_fraction(donor_mass_msun / compact_mass_msun)
        roche_radius_periastron = roche_fraction * float(np.nanmin(separation))
        roche_fill = donor_radius_over_a / roche_radius_periastron
        if roche_fill >= 1.0:
            add("outside assumptions", "roche_overflow", "The donor fills or exceeds its periastron Roche lobe; a wind-only BHL model is inappropriate.")
        elif roche_fill >= 0.9:
            add("caution", "near_roche_overflow", "The donor nearly fills its periastron Roche lobe.")

    status = "valid"
    if any(item["level"] == "outside assumptions" for item in warnings):
        status = "outside assumptions"
    elif warnings:
        status = "caution"
    diagnostics: dict[str, float | str] = {
        "validity_status": status,
        "mach_min": float(np.nanmin(mach)),
        "mach_max": float(np.nanmax(mach)),
        "wind_ratio_min": float(np.nanmin(wind_speed)),
        "wind_ratio_max": float(np.nanmax(wind_speed)),
        "accretion_radius_over_a_max": float(np.nanmax(racc)),
        "accretion_radius_over_r_max": float(np.nanmax(racc / separation)),
        "capture_efficiency_max": float(np.nanmax(efficiency)),
        "minimum_separation_over_donor_radius": float(np.nanmin(separation) / donor_radius_over_a),
        "roche_lobe_filling_periastron": roche_fill,
    }
    return warnings, diagnostics
