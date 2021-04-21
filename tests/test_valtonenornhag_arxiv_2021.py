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

    def test_valtonenornhag_arxiv_2021_fEf_sol0_t(self):
        verify(verify_numpy_array(self.sols[0]['t']))

    def test_valtonenornhag_arxiv_2021_fEf_sol1_t(self):
        verify(verify_numpy_array(self.sols[1]['t']))

    def test_valtonenornhag_arxiv_2021_fEf_sol2_t(self):
        verify(verify_numpy_array(self.sols[2]['t']))

    def test_valtonenornhag_arxiv_2021_fEf_sol3_t(self):
        verify(verify_numpy_array(self.sols[3]['t']))

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


class ValtonenOrnhagArxiv2021frEfrTestCase(unittest.TestCase):
    def setUp(self):
        self.tol = 6
        self.p1 = np.array([
            [99.0825859985542, 1136.84241396289, -1031.66650596755, -117.325418998838],
            [-301.923289351466, 1760.62028612233, -533.989983528509, 566.900954605729]
        ])

        self.p2 = np.array([
            [1829.78818884974, 15378.7866880232, 612.159309750213, 2756.44403323769],
            [474.180433404958, 4677.92468041337, 1092.76420021176, 1874.89973953780]
        ])

        self.R1 = np.array([
            [0.983417707845482, 0.013453875580959, 0.180855204867843],
            [-0.060089831339915, 0.965084992860598, 0.254951306575981],
            [-0.171110560940808, -0.261591188282616, 0.949890112669571]
        ])

        self.R2 = np.array([
            [0.556837962037774, 0.329755275145440, 0.762360113428931],
            [-0.787076103923440, 0.502747650438714, 0.357429722618378],
            [-0.265410419287405, -0.799065866178820, 0.539491474299248]
        ])

        self.sols = dpl.get_valtonenornhag_arxiv_2021_frEfr(
            np.asfortranarray(self.p1),
            np.asfortranarray(self.p2),
            np.asfortranarray(self.R1),
            np.asfortranarray(self.R2),
            False
        )

    def test_valtonenornhag_arxiv_2021_frEfr_length(self):
        assert len(self.sols) == 9

    def test_valtonenornhag_arxiv_2021_frEfr_sol0(self):
        np.testing.assert_almost_equal(self.sols[0]['f'], -125306.614892352, self.tol)
        np.testing.assert_almost_equal(self.sols[0]['r'], -203321.4180906687, self.tol)
        verify(verify_numpy_array(self.sols[0]['F']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol1(self):
        np.testing.assert_almost_equal(self.sols[1]['f'], -4414.62813326810, self.tol)
        np.testing.assert_almost_equal(self.sols[1]['r'], -86722.00865063918, self.tol)
        verify(verify_numpy_array(self.sols[1]['F']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol2(self):
        np.testing.assert_almost_equal(self.sols[2]['f'], 2445.470419511240, self.tol)
        np.testing.assert_almost_equal(self.sols[2]['r'], 1.987432298474289e-08, self.tol)
        verify(verify_numpy_array(self.sols[2]['F']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol3(self):
        np.testing.assert_almost_equal(self.sols[3]['f'], 2291.313422912460, self.tol)
        np.testing.assert_almost_equal(self.sols[3]['r'], 9774.556116633164, self.tol)
        verify(verify_numpy_array(self.sols[3]['F']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol4(self):
        np.testing.assert_almost_equal(self.sols[4]['f'], -75.4204581489593, self.tol)
        np.testing.assert_almost_equal(self.sols[4]['r'], 190060.1388957948, self.tol)
        verify(verify_numpy_array(self.sols[4]['F']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol5(self):
        np.testing.assert_almost_equal(self.sols[5]['f'], 1081.280695559530, self.tol)
        np.testing.assert_almost_equal(self.sols[5]['r'], -15606.02225174152, self.tol)
        verify(verify_numpy_array(self.sols[5]['F']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol6(self):
        np.testing.assert_almost_equal(self.sols[6]['f'], 7.142069186813480, self.tol)
        np.testing.assert_almost_equal(self.sols[6]['r'], -10387926.073495165, self.tol)
        verify(verify_numpy_array(self.sols[6]['F']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol7(self):
        np.testing.assert_almost_equal(self.sols[7]['f'], 96.42064345373980, self.tol)
        np.testing.assert_almost_equal(self.sols[7]['r'], -1845837.551419409, self.tol)
        verify(verify_numpy_array(self.sols[7]['F']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol8(self):
        np.testing.assert_almost_equal(self.sols[8]['f'], 944.9706198704850, self.tol)
        np.testing.assert_almost_equal(self.sols[8]['r'], 483098.2018965589, self.tol)
        verify(verify_numpy_array(self.sols[8]['F']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol0_t(self):
        verify(verify_numpy_array(self.sols[0]['t']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol1_t(self):
        verify(verify_numpy_array(self.sols[1]['t']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol2_t(self):
        verify(verify_numpy_array(self.sols[2]['t']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol3_t(self):
        verify(verify_numpy_array(self.sols[3]['t']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol4_t(self):
        verify(verify_numpy_array(self.sols[4]['t']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol5_t(self):
        verify(verify_numpy_array(self.sols[5]['t']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol6_t(self):
        verify(verify_numpy_array(self.sols[6]['t']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol7_t(self):
        verify(verify_numpy_array(self.sols[7]['t']))

    def test_valtonenornhag_arxiv_2021_frEfr_sol8_t(self):
        verify(verify_numpy_array(self.sols[8]['t']))

    def test_valtonenornhag_arxiv_2021_frEfr_dimensions01(self):
        """Check that an exception is raised when dimensions are incorrect."""
        p1 = np.random.randn(2, 2)
        p2 = np.random.randn(3, 2)
        R1 = np.random.randn(3, 3)
        R2 = np.random.randn(3, 3)
        with self.assertRaises(ValueError):
            sols = dpl.get_valtonenornhag_arxiv_2021_frEfr( # noqa
                np.asfortranarray(p1),
                np.asfortranarray(p2),
                np.asfortranarray(R1),
                np.asfortranarray(R2),
                False
            )

    def test_valtonenornhag_arxiv_2021_frEfr_dimensions02(self):
        """Check that an exception is raised when dimensions are incorrect."""
        p1 = np.random.randn(2, 3)
        p2 = np.random.randn(2, 2)
        R1 = np.random.randn(3, 3)
        R2 = np.random.randn(3, 3)
        with self.assertRaises(ValueError):
            sols = dpl.get_valtonenornhag_arxiv_2021_frEfr( # noqa
                np.asfortranarray(p1),
                np.asfortranarray(p2),
                np.asfortranarray(R1),
                np.asfortranarray(R2),
                False
            )
