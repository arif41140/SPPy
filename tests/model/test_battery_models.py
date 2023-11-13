import unittest

import numpy as np
import numpy.typing as npt

import SPPy
from SPPy.models import battery


class TestSPModel(unittest.TestCase):
    # Below are the parameters that are used for the test cases below
    k_p = 6.67e-11
    S_p = 1.1167
    max_conc_p = 51410.0
    c_e = 1000

    k_n = 1.76e-11
    S_n = 0.7824
    max_conc_n = 31833

    def test_m(self):
        soc_init_p = 0.6
        soc_init_n = 0.7

        testmodel = battery.SPM()
        self.assertEqual(-0.2893183331034342, testmodel.m(-1.656, self.k_p, self.S_p, self.max_conc_p,
                                                          soc_init_p, self.c_e))
        self.assertEqual(-2.7018597575301726, testmodel.m(-1.656, self.k_n, self.S_n, self.max_conc_n,
                                                          soc_init_n, self.c_e))

    def test_V(self):
        T = 298.15
        soc_init_p = 0.4956
        soc_init_n = 0.7568
        R_cell = 0.00148861
        I = -1.656

        testmodel = battery.SPM()
        OCP_p = 4.176505962016067
        OCP_n = 0.07464309895951012

        m_p = testmodel.m(-1.656, self.k_p, self.S_p, self.max_conc_p, soc_init_p, self.c_e)
        m_n = testmodel.m(-1.656, self.k_n, self.S_n, self.max_conc_n, soc_init_n, self.c_e)
        self.assertEqual(-0.28348389244322414, testmodel.m(-1.656, self.k_p, self.S_p, self.max_conc_p,
                                                           soc_init_p, self.c_e))
        self.assertEqual(-2.8860250955114384, testmodel.m(-1.656, self.k_n, self.S_n, self.max_conc_n,
                                                          soc_init_n, self.c_e))

        self.assertEqual(4.032392212009281,
                         testmodel.calc_cell_terminal_voltage(OCP_p, OCP_n, m_p, m_n, R_cell, T=T, I=I))


class TestESP(unittest.TestCase):
    def test_molar_flux(self):
        I = -1.656
        S = 0.7824
        self.assertEqual(2.1936265167099342e-05, battery.SPMe.molar_flux_electrode(I=I, S=S, electrode_type='n'))
        self.assertEqual(-2.1936265167099342e-05, battery.SPMe.molar_flux_electrode(I=I, S=S, electrode_type='p'))

    def test_i_0(self):
        k = 1.764e-11
        c_s_max = 31833
        c_e = 1000
        soc_surf = 0.5
        self.assertEqual(8.878634015491551e-06, battery.SPMe.i_0(k=k, c_s_max=c_s_max, c_e=c_e, soc_surf=soc_surf))

    def test_eta(self):
        j = 2.1936265167099342e-05
        i_0_ = 8.878634015491551e-06
        temp = 298.15
        self.assertAlmostEqual(0.05335777844201581, battery.SPMe.eta(temp=temp, j=j, i_0_=i_0_))

    def test_calc_terminal_voltage(self):
        ocp_p = 4.2
        ocp_n = 0.15
        eta_p = 0.05
        eta_n = 0.05
        l_p = 7.35E-05
        l_sep = 2.00E-05
        l_n = 7.00E-05
        battery_cross_area = 0.0596
        kappa_eff_avg = 0.2
        t_c = 0.38
        R_p = 0.0
        R_n = 0.0
        S_p = 1.1167
        S_n = 0.7824
        c_e_n = 1100
        c_e_p = 900
        i_app = -1.656

        print(battery.SPMe.calc_terminal_voltage(ocp_p=ocp_p, ocp_n=ocp_n, eta_p=eta_p, eta_n=eta_n, l_p=l_p,
                                                 l_sep=l_sep, l_n=l_n, battery_cross_area=battery_cross_area,
                                                 kappa_eff_avg=kappa_eff_avg, t_c=t_c, R_p=R_p, R_n=R_n,
                                                 S_p=S_p, S_n=S_n, c_e_n=c_e_n, c_e_p=c_e_p, i_app=i_app, k_f_avg=5,
                                                 temp=298.15))


class TestP2DM(unittest.TestCase):
    def test_a_s(self):
        # Typical value for the negative electrode containing graphite
        self.assertEqual(340000.0, battery.P2DM.a_s(epsilon=0.68, r=6e-6))

        # Typical value for the positive electrode containing NMC
        self.assertEqual(390000.0, battery.P2DM.a_s(epsilon=0.65, r=5e-6))

    def test_i_0(self):
        # parameter below are from Shangwoo et al
        k_n = 2.3e-10
        k_p = 1.43e-10
        c_s_max_n = 31221
        c_s_max_p = 50179
        c_s_surf_n = 30596.579999999998
        c_s_surf_p = 17562.649999999998
        c_e = 900
        c_e_0 = 1000

        self.assertAlmostEqual(0.09202222696431563,
                               battery.P2DM.i_0(k=k_n, c_s_surf=c_s_surf_n, c_s_max=c_s_max_n, c_e=c_e, c_e_0=c_e_0))
        self.assertAlmostEqual(0.31328442058013517,
                               battery.P2DM.i_0(k=k_p, c_s_surf=c_s_surf_p, c_s_max=c_s_max_p, c_e=c_e, c_e_0=c_e_0))

        # Below tests for the exchange current equation using the Numpy arrays
        c_s_surf_n = np.array([30596.58, 30000, 28000, 25000, 24000])
        c_e = np.array([900, 1000, 990, 1000, 800])
        c_e_0 = np.array([1000, 1000, 1000, 1000, 1000])

        actual_i_0_n = np.array([0.09202222696431563, 0.13431208741651982, 0.20969526054264584,
                                 0.27675580843377356, 0.2613039208124068])

        self.assertTrue(np.allclose(actual_i_0_n,
                                    battery.P2DM.i_0(k=k_n,
                                                     c_s_surf=c_s_surf_n, c_s_max=c_s_max_n,
                                                     c_e=c_e, c_e_0=c_e_0)))

    def test_v_n_minus_v_e(self):
        array_rel_eta_n = np.array([-0.06276411, -0.06191974, -0.06021835, -0.05717165, -0.05351207])
        array_i_0_n = np.array([0.00388174, 0.00388174, 0.00388174, 0.00388174, 0.00388174])
        array_x_n = np.array([8.10e-06, 2.43e-05, 4.05e-05, 5.67e-05, 7.29e-05])

        i_app = 4
        temp = 298.15
        cell_area = 0.06
        a_s_n = 340000
        L_n = 81e-6

        v_e = battery.P2DM.v_n_minus_v_e(array_i_0_n=array_i_0_n,
                                         array_eta_rel_n=array_rel_eta_n,
                                         array_coord_n=array_x_n,
                                         i_app=i_app, temp=temp, a_s_n=a_s_n, cell_area=cell_area, L_n=L_n)
        self.assertEqual(-0.92599193513242, v_e)

    def test_v_p_minum_v_e(self):
        pass

    def test_calc_eta_from_rel_eta(self):
        array_rel_eta: npt.ArrayLike = np.array([-4.2, -4.2, -4.2, -4.2, -4.2])
        v_terminal: float = 4.16
        v_terminal_e: float = 0.92
        array_actual: npt.ArrayLike = np.array([-0.96, -0.96, -0.96, -0.96, -0.96])

        self.assertTrue(np.allclose(array_actual,
                                    battery.P2DM.calc_eta_from_rel_eta(rel_eta=array_rel_eta,
                                                                       v_terminal=v_terminal,
                                                                       v_terminal_e=v_terminal_e)))
