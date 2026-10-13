"""Scientific regression tests; run with unittest discover -s tests -v."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from geoprior1d.lithology import _check_section_depth_constraints
from geoprior1d.resistivity import prior_res_reals
from geoprior1d.sampling import _generate_single_realization, get_prior_sample


def geology(frequency):
    return {
        'Classes': {'min_thick': np.array([1., 1., 1.]),
                    'max_thick': np.array([10., 2., 100.])},
        'Sections': {
            'N_sections': 3, 'types': [[1], [2], [3]],
            'probabilities': [[1.], [1.], [1.]],
            'min_layers': np.array([1, 1, 1]), 'max_layers': np.array([1, 1, 1]),
            'min_thick': np.array([1., 1., 1.]), 'max_thick': np.array([10., 2., 1.]),
            'frequency': np.array([1., frequency, 1.]), 'repeat': np.array([1, 1, 1]),
            'min_depth': np.array([0., 0., 0.]),
        },
        'Resistivity': {
            'res': np.array([100., 20., 200.]), 'res_unc': np.array([.1, .1, .1]),
            'unsat_res': np.array([1000., 200., 2000.]),
            'unsat_res_unc': np.array([.1, .1, .1]),
        },
    }


class MinimumDepthTests(unittest.TestCase):
    def test_bounded_batches_preserve_global_seeds_and_flags(self):
        info = geology(.5)
        z = np.arange(1., 21.)
        np.random.seed(123)
        offset = np.random.randint(0, 1e9)
        expected = [_generate_single_realization(i, info, z, offset) for i in range(2051)]
        for processes in (1, 2):
            np.random.seed(123)
            ms, ns, ws, flags = get_prior_sample(info, z, 2051, n_processes=processes)
            np.testing.assert_array_equal(ms, np.array([s[0] for s in expected], dtype=np.float32))
            np.testing.assert_array_equal(ns, np.array([s[1] for s in expected], dtype=np.float32))
            np.testing.assert_array_equal(ws, np.array([s[2] for s in expected], dtype=np.float32))
            np.testing.assert_allclose(flags, [max(s[3][0] for s in expected),
                                               max(s[3][1] for s in expected),
                                               sum(s[3][2] for s in expected)/2051])

    def test_only_present_upper_units_constrain_depth(self):
        depths = np.array([0., 9., 999.])
        for thicknesses, expected in [([3., 0., 0.], 0), ([3., 1., 0.], 1),
                                      ([9., 1., 0.], 0)]:
            with self.subTest(thicknesses=thicknesses):
                self.assertEqual(_check_section_depth_constraints(
                    np.array(thicknesses), 3, depths), expected)

    def test_absent_units_do_not_condition_the_ensemble(self):
        z = np.arange(.25, 20.25, .25)
        for frequency in (0., .5):
            base = geology(frequency)
            constrained = copy.deepcopy(base)
            constrained['Sections']['min_depth'][1] = 9.
            absent = present = 0
            upper_lengths = []
            for seed in range(100):
                before = _generate_single_realization(0, base, z, seed)
                after = _generate_single_realization(0, constrained, z, seed)
                if not np.any(before[0] == 2):
                    absent += 1
                    for left, right in zip(before, after):
                        np.testing.assert_array_equal(left, right)
                    upper_lengths.append(np.sum(after[0] == 1))
                else:
                    present += 1
                    self.assertTrue(np.all(z[after[0] == 2] > 9.))
                    self.assertEqual(after[3][:2], [0, 0])
            self.assertGreater(absent, 20)
            self.assertGreater(len(set(upper_lengths)), 10)
            if frequency:
                self.assertGreater(present, 20)

    def test_all_upper_units_optional_leaves_background(self):
        info = geology(.3)
        info['Sections']['frequency'] = np.array([.4, .3, 1.])
        z = np.arange(1., 21.)
        background_only = 0
        for seed in range(300):
            m, resistivity, _, flags = _generate_single_realization(0, info, z, seed)
            self.assertEqual(flags[:2], [0, 0])
            if not np.any(m != 3):
                background_only += 1
                self.assertEqual(len(np.unique(resistivity)), 1)  # one background layer
        self.assertGreater(background_only, 300 * .6 * .7 * .6)  # expected share is .42
        np.random.seed(123)
        results = [get_prior_sample(info, z, 200, n_processes=p) for p in (1, 2)]
        for _, _, _, flags in results:
            self.assertEqual(list(flags[:2]), [0, 0])


class GroundwaterTests(unittest.TestCase):
    def setUp(self):
        self.info = geology(0.)
        self.info['Resistivity']['res_unc'][:] = 0
        self.info['Resistivity']['unsat_res_unc'][:] = 0

    def sample(self, water, dz=1., count=4):
        z = np.arange(1, count + 1) * dz
        return prior_res_reals(self.info, np.ones(count), water, np.ones(count), z)

    def test_surface_first_cell_and_exact_boundaries(self):
        cases = [(0., [100, 100, 100, 100]), (.5, [550, 100, 100, 100]),
                 (1., [1000, 100, 100, 100]), (1.5, [1000, 550, 100, 100]),
                 (2., [1000, 1000, 100, 100]), (4., [1000]*4), (5., [1000]*4)]
        for water, expected in cases:
            with self.subTest(water=water):
                np.testing.assert_allclose(self.sample(water), expected, rtol=0, atol=1e-12)

    def test_fractional_cells_and_continuity(self):
        np.testing.assert_allclose(self.sample(.05, .1), [550, 100, 100, 100])
        np.testing.assert_allclose(self.sample(.3, .1), [1000, 1000, 1000, 100])
        for depth in (1., 2.):
            exact = self.sample(depth)
            for nearby in (np.nextafter(depth, 0), np.nextafter(depth, np.inf)):
                np.testing.assert_allclose(self.sample(nearby), exact, rtol=0, atol=1e-10)

    def test_per_layer_sampling_and_draw_order(self):
        info = geology(0.)
        m = np.array([1., 1., 2., 2.])
        layers = np.array([2, 2, 1, 1])
        z = np.arange(1., 5.)
        # Sorted layer IDs: class 2 sat/unsat, then class 1 sat/unsat.
        with patch('numpy.random.randn', side_effect=[.1, .2, .3, .4]) as draws:
            actual = prior_res_reals(info, m, 2.5, layers, z)
        sat2, unsat2 = 20 * 10**.01, 200 * 10**.02
        unsat1 = 1000 * 10**.04
        np.testing.assert_allclose(actual, [unsat1, unsat1, (sat2+unsat2)/2, sat2])
        self.assertEqual(draws.call_count, 4)

    def test_dry_draw_order_is_preserved(self):
        with patch('numpy.random.randn', side_effect=[.1, .2]) as draws:
            np.testing.assert_array_equal(self.sample(0.), [100.]*4)
        self.assertEqual(draws.call_count, 2)


if __name__ == '__main__':
    unittest.main()
