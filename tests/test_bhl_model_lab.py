import unittest

import numpy as np

from app import bondi_hoyle_model_lab


def synthetic_result(sign: float = 1.0) -> dict:
    phase = np.linspace(0.0, 1.0, 96, endpoint=False)
    time = 5.0 * phase
    flux = 10.0 + sign * np.cos(2.0 * np.pi * phase)
    return {
        "series": {
            "time": time.tolist(),
            "flux": flux.tolist(),
            "error": np.full_like(time, 0.1).tolist(),
        },
        "folded_period": 5.0,
        "t0": 0.0,
        "baseline_unit": "d",
        "period_unit": "d",
    }


def base_fields() -> dict[str, str]:
    return {
        "model_lab_bh_period": "5",
        "model_lab_bh_t0": "0",
        "model_lab_bh_eccentricity": "0.2",
        "model_lab_bh_wind_input_mode": "v_inf",
        "model_lab_bh_vinf": "1200",
        "model_lab_bh_donor_mass": "18",
        "model_lab_bh_compact_mass": "1.4",
        "model_lab_bh_donor_radius": "8",
        "model_lab_bh_wind_beta": "0.8",
        "model_lab_bh_sound_speed_km_s": "10",
        "model_lab_bh_formulation": "classical",
        "model_lab_bh_normalization_mode": "normalized",
        "model_lab_bh_response_mode": "none",
        "model_lab_bh_fit_eccentricity": "false",
        "model_lab_bh_fit_wind_speed": "false",
        "model_lab_bh_phase_lag": "false",
    }


class BondiHoyleModelLabTests(unittest.TestCase):
    def test_result_contains_diagnostics_and_model_comparison(self):
        output = bondi_hoyle_model_lab(synthetic_result(), base_fields())
        self.assertEqual(output["family"], "bondi_hoyle")
        self.assertEqual(len(output["model_flux"]), 1600)
        self.assertEqual(len(output["relative_speed_phase"]), 1600)
        self.assertIn(output["summary"]["validity_status"], {"valid", "caution", "outside assumptions"})
        self.assertEqual(
            {row["model"] for row in output["model_comparison"]},
            {"Bondi-Hoyle", "Constant", "Sinusoid", "Fourier h=2"},
        )

    def test_response_delay_does_not_move_periastron_epoch(self):
        fields = base_fields()
        fields.update(
            {
                "model_lab_bh_response_mode": "delay",
                "model_lab_bh_phase_lag": "true",
            }
        )
        output = bondi_hoyle_model_lab(synthetic_result(), fields)
        self.assertEqual(output["t0"], 0.0)
        self.assertEqual(output["summary"]["T0"], 0.0)
        self.assertEqual(output["summary"]["response_mode"], "delay")

    def test_physical_mode_returns_rates_and_luminosity(self):
        fields = base_fields()
        fields.update(
            {
                "model_lab_bh_formulation": "revised",
                "model_lab_bh_normalization_mode": "physical",
                "model_lab_bh_mass_loss_msun_yr": "1e-6",
            }
        )
        output = bondi_hoyle_model_lab(synthetic_result(), fields)
        self.assertEqual(len(output["mdot_msun_yr_phase"]), 1600)
        self.assertGreater(output["summary"]["luminosity_max_erg_s"], 0.0)
        self.assertGreaterEqual(output["summary"]["eddington_ratio_max"], 0.0)


if __name__ == "__main__":
    unittest.main()
