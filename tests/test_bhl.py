import unittest

import numpy as np

from bhl import (
    bhl_accretion_state,
    bhl_orbital_state,
    bhl_validity_warnings,
    periodic_response,
    physical_bhl_outputs,
)


class BondiHoyleScientificTests(unittest.TestCase):
    def test_circular_relative_velocity_reduces_to_quadrature(self):
        state = bhl_orbital_state(
            np.linspace(0.0, 1.0, 32, endpoint=False),
            eccentricity=0.0,
            wind_terminal_ratio=3.0,
            wind_beta=0.8,
            donor_radius_over_a=0.15,
        )
        expected = np.sqrt(np.asarray(state["wind_speed"]) ** 2 + 1.0)
        np.testing.assert_allclose(state["relative_speed"], expected, rtol=1e-12)
        np.testing.assert_allclose(state["orbital_radial"], 0.0, atol=1e-12)

    def test_eccentric_relative_velocity_uses_radial_cross_term(self):
        state = bhl_orbital_state(
            np.array([0.125, 0.625]),
            eccentricity=0.4,
            wind_terminal_ratio=1.5,
            wind_beta=0.8,
            donor_radius_over_a=0.15,
        )
        wind = np.asarray(state["wind_speed"])
        radial = np.asarray(state["orbital_radial"])
        tangential = np.asarray(state["orbital_tangential"])
        expected = np.sqrt((wind - radial) ** 2 + tangential**2)
        np.testing.assert_allclose(state["relative_speed"], expected, rtol=1e-12)

    def test_revised_circular_efficiency_matches_closed_form(self):
        compact_fraction = 0.2
        state = bhl_accretion_state(
            np.array([0.0]),
            eccentricity=0.0,
            wind_terminal_ratio=2.0,
            wind_beta=0.0,
            donor_radius_over_a=0.1,
            compact_mass_fraction=compact_fraction,
            formulation="revised",
        )
        expected = (compact_fraction / (1.0 + 2.0**2)) ** 2
        self.assertAlmostEqual(float(np.asarray(state["selected_efficiency"])[0]), expected, places=14)

    def test_sound_speed_reduces_classical_rate(self):
        common = dict(
            phase=np.array([0.0]),
            eccentricity=0.0,
            wind_terminal_ratio=2.0,
            wind_beta=0.0,
            donor_radius_over_a=0.1,
            compact_mass_fraction=0.1,
            formulation="classical",
        )
        cold = bhl_accretion_state(**common, sound_speed_ratio=0.0)
        warm = bhl_accretion_state(**common, sound_speed_ratio=1.0)
        self.assertLess(
            float(np.asarray(warm["selected_efficiency"])[0]),
            float(np.asarray(cold["selected_efficiency"])[0]),
        )

    def test_classical_rate_has_compact_mass_squared_scaling(self):
        common = dict(
            phase=np.array([0.0]),
            eccentricity=0.0,
            wind_terminal_ratio=3.0,
            wind_beta=0.0,
            donor_radius_over_a=0.1,
            formulation="classical",
        )
        low = bhl_accretion_state(**common, compact_mass_fraction=0.1)
        high = bhl_accretion_state(**common, compact_mass_fraction=0.2)
        ratio = float(np.asarray(high["selected_efficiency"])[0] / np.asarray(low["selected_efficiency"])[0])
        self.assertAlmostEqual(ratio, 4.0, places=12)

    def test_invalid_orbit_intersecting_donor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "periastron"):
            bhl_orbital_state(
                np.array([0.0]),
                eccentricity=0.5,
                wind_terminal_ratio=2.0,
                wind_beta=0.8,
                donor_radius_over_a=0.5,
            )

    def test_physical_mass_loss_scaling(self):
        state = bhl_accretion_state(
            np.array([0.0, 0.5]),
            eccentricity=0.1,
            wind_terminal_ratio=3.0,
            wind_beta=0.8,
            donor_radius_over_a=0.1,
            compact_mass_fraction=0.1,
            formulation="classical",
        )
        kwargs = dict(
            state=state,
            period_seconds=86400.0,
            semi_major_axis_m=1e10,
            orbital_speed_m_s=2e5,
            compact_mass_msun=1.4,
            compact_radius_km=10.0,
            radiative_efficiency=0.1,
            luminosity_mode="radius",
            eddington_cap=False,
        )
        low = physical_bhl_outputs(**kwargs, donor_mass_loss_msun_yr=1e-7)
        high = physical_bhl_outputs(**kwargs, donor_mass_loss_msun_yr=2e-7)
        np.testing.assert_allclose(high["mdot_kg_s"], 2.0 * np.asarray(low["mdot_kg_s"]))

    def test_response_delay_preserves_periodic_shape(self):
        phase = np.linspace(0.0, 1.0, 512, endpoint=False)
        signal = np.cos(2.0 * np.pi * phase)
        delayed = periodic_response(phase, signal, mode="delay", delay_phase=0.25)
        np.testing.assert_allclose(delayed, np.sin(2.0 * np.pi * phase), atol=2e-2)

    def test_validity_warnings_include_roche_overflow_and_negative_scale(self):
        state = bhl_accretion_state(
            np.linspace(0.0, 1.0, 128, endpoint=False),
            eccentricity=0.2,
            wind_terminal_ratio=3.0,
            wind_beta=0.8,
            donor_radius_over_a=0.45,
            sound_speed_ratio=0.1,
            compact_mass_fraction=0.1,
            formulation="classical",
        )
        warnings, diagnostics = bhl_validity_warnings(
            state,
            donor_radius_over_a=0.45,
            donor_mass_msun=10.0,
            compact_mass_msun=1.4,
            fitted_scale=-1.0,
        )
        codes = {warning["code"] for warning in warnings}
        self.assertIn("negative_scale", codes)
        self.assertIn("roche_overflow", codes)
        self.assertEqual(diagnostics["validity_status"], "outside assumptions")


if __name__ == "__main__":
    unittest.main()
