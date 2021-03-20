import unittest
from approvaltests import verify
from tests.helpers import verify_numpy_array

import numpy as np
import droneposelib as dpl


class ValtonenOrnhagArxiv2021fEfTestCase(unittest.TestCase):
    def setUp(self):
        self.tol = 14
        self.p1 = np.array([
            [-34.998621048798569, -0.064090328787399, -8.738571263872560],
            [-26.916377657772212, 10.374453893285320, -8.028592688400703]
        ])

        self.p2 = np.array([
            [-164.2306957039563, -4.85637790217750, -44.47452317786740],
            [18.59197557487190, 33.7875137840784, 17.86327225402930]
        ])

        self.R1 = np.array([
            [-0.180619577025105, 0.225943086758464, -0.957249335304720],
            [-0.982474033611525, 0.004129208291573, 0.186353757456038],
            [0.046058025081098, 0.974131752477529, 0.221237399959154]
        ])

        self.R2 = np.array([
            [-0.460139184327122, 0.450642405815460, -0.764979315490049],
            [-0.848730576230669, -0.476191421985198, 0.229995953440211],
            [-0.260630658246353, 0.755091485654930, 0.601588321257572]
        ])

        self.sols = dpl.get_valtonenornhag_arxiv_2021_fEf(
            np.asfortranarray(self.p1),
            np.asfortranarray(self.p2),
            np.asfortranarray(self.R1),
            np.asfortranarray(self.R2)
        )

    def test_valtonenornhag_arxiv_2021_fEf_length(self):
        assert len(self.sols) == 4

    def test_valtonenornhag_arxiv_2021_fEf_sol0(self):
        np.testing.assert_almost_equal(self.sols[0]['f'], 5.998701610439096, self.tol)
        verify(verify_numpy_array(self.sols[0]['F']))

    def test_valtonenornhag_arxiv_2021_fEf_sol1(self):
        np.testing.assert_almost_equal(self.sols[1]['f'], -10.455953354057023, self.tol)
        verify(verify_numpy_array(self.sols[1]['F']))

    def test_valtonenornhag_arxiv_2021_fEf_sol2(self):
        np.testing.assert_almost_equal(self.sols[2]['f'], 51.61046271601693, self.tol)
        verify(verify_numpy_array(self.sols[2]['F']))

    def test_valtonenornhag_arxiv_2021_fEf_sol3(self):
        np.testing.assert_almost_equal(self.sols[3]['f'], -37.468067324800963, self.tol)
        verify(verify_numpy_array(self.sols[3]['F']))

    def test_valtonenornhag_arxiv_2021_fEf_dimensions01(self):
        """Check that an exception is raised when dimensions are incorrect."""
        p1 = np.random.randn(2, 2)
        p2 = np.random.randn(3, 2)
        R1 = np.random.randn(3, 3)
        R2 = np.random.randn(3, 3)
        with self.assertRaises(ValueError):
            sols = dpl.get_valtonenornhag_arxiv_2021_fEf( # noqa
                np.asfortranarray(p1),
                np.asfortranarray(p2),
                np.asfortranarray(R1),
                np.asfortranarray(R2)
            )

    def test_valtonenornhag_arxiv_2021_fEf_dimensions02(self):
        """Check that an exception is raised when dimensions are incorrect."""
        p1 = np.random.randn(2, 3)
        p2 = np.random.randn(2, 2)
        R1 = np.random.randn(3, 3)
        R2 = np.random.randn(3, 3)
        with self.assertRaises(ValueError):
            sols = dpl.get_valtonenornhag_arxiv_2021_fEf( # noqa
                np.asfortranarray(p1),
                np.asfortranarray(p2),
                np.asfortranarray(R1),
                np.asfortranarray(R2)
            )
