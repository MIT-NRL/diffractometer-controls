import unittest

import numpy as np
from lmfit.models import GaussianModel, LorentzianModel, PseudoVoigtModel, VoigtModel

from diffraction_analysis.fitting import FitConfig, fit_curve


class DiffractionFittingTests(unittest.TestCase):
    def test_profile_types_with_constant_and_linear_background(self):
        rng = np.random.default_rng(17)
        x = np.linspace(-8, 8, 401)
        for name, model_class in [('Gaussian', GaussianModel), ('Lorentzian', LorentzianModel), ('Voigt', VoigtModel), ('Pseudo-Voigt', PseudoVoigtModel)]:
            for background in ('Constant', 'Linear', 'None'):
                with self.subTest(peak=name, background=background):
                    model = model_class()
                    params = model.make_params(center=0.7, amplitude=180, sigma=0.65)
                    if name == 'Voigt':
                        params['gamma'].set(value=0.35, expr='')
                    y = model.eval(params, x=x)
                    if background != 'None':
                        y += 8
                    if background == 'Linear':
                        y += 0.2 * x
                    y += rng.normal(0, 0.015, len(x))
                    result = fit_curve(x, y, FitConfig(peak=name, background=background, xmin=-6, xmax=6))
                    self.assertAlmostEqual(result['parameters']['p1_center']['value'], 0.7, delta=0.002)
                    self.assertAlmostEqual(result['parameters']['p1_amplitude']['value'], 180, delta=0.6)
                    self.assertTrue(result['parameters']['p1_fwhm']['value'] > 0)
                    self.assertEqual(len(result['residual']), result['ndata'])
                    self.assertLess(np.std(result['residual']), 0.03)

    def test_multiple_peaks_and_starting_centers(self):
        x = np.linspace(-8, 8, 501)
        y = 5 + 80 * np.exp(-0.5 * ((x + 2) / 0.5) ** 2) + 45 * np.exp(-0.5 * ((x - 2.5) / 0.8) ** 2)
        for centers in ((), (-2.2, 2.3)):
            result = fit_curve(x, y, FitConfig(peaks=2, centers=centers))
            self.assertAlmostEqual(result['parameters']['p1_center']['value'], -2, places=4)
            self.assertAlmostEqual(result['parameters']['p2_center']['value'], 2.5, places=4)

    def test_window_finite_filtering_and_descending_coordinates(self):
        x = np.linspace(-5, 5, 101)
        y = 3 + 40 * np.exp(-0.5 * ((x - 1) / 0.4) ** 2)
        y[3] = np.nan
        x[4] = np.inf
        y[90:] += 100
        result = fit_curve(x[::-1], y[::-1], FitConfig(xmin=-1, xmax=3))
        self.assertTrue(np.all(np.diff(result['x']) >= 0))
        self.assertAlmostEqual(result['parameters']['p1_center']['value'], 1, places=3)
        self.assertTrue(np.all((result['x'] >= -1) & (result['x'] <= 3)))

    def test_poisson_weighting_and_invalid_data(self):
        x = np.linspace(-4, 4, 101)
        y = 20 + 80 * np.exp(-0.5 * x ** 2)
        result = fit_curve(x, y, FitConfig(weighting='Poisson (counts)'))
        self.assertAlmostEqual(result['parameters']['p1_center']['value'], 0, places=4)
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            fit_curve(x, y - 100, FitConfig(weighting='Poisson (counts)'))
        with self.assertRaisesRegex(ValueError, 'constant'):
            fit_curve(x, np.ones_like(x))
        with self.assertRaisesRegex(ValueError, 'seven'):
            fit_curve(np.arange(3), np.arange(3))
        with self.assertRaisesRegex(ValueError, 'minimum'):
            fit_curve(x, y, FitConfig(xmin=3, xmax=-3))
        with self.assertRaisesRegex(ValueError, 'starting center'):
            fit_curve(x, y, FitConfig(peaks=2, centers=(0,)))
