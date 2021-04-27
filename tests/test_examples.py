import unittest

import numpy as np
import droneposelib as dpl
from example.synthetic import generate_points_realistic, compare_to_gt


class ExampleTestCase(unittest.TestCase):
    def setUp(self):
        """This mimicks the example/synthetic script."""
        np.random.seed(0)
        self.tol = 14
        N = 4
        distortion_param = -1e-07
        R1, R2, f, F, x1, x2, R, t, x1u, x2u = generate_points_realistic(N, distortion_param)
        use_fast_solver = False
        out = dpl.get_valtonenornhag_arxiv_2021_frEfr(np.asfortranarray(x1[:2, :]), np.asfortranarray(x2[:2, :]),
                                                      np.asfortranarray(R1), np.asfortranarray(R2), use_fast_solver)
        self.f_err, self.F_err, self.r_err = compare_to_gt(out, f, F, distortion_param)

    def test_example_error_f(self):
        np.testing.assert_almost_equal(self.f_err, 8.852394837204933e-12, self.tol)

    def test_example_error_F(self):
        np.testing.assert_almost_equal(self.F_err, 3.2968551174587434e-13, self.tol)

    def test_example_error_r(self):
        np.testing.assert_almost_equal(self.r_err, 1.9412976064553898e-10, self.tol)
