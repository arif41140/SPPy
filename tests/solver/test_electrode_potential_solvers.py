import unittest

import numpy as np

from SPPy.calc_helpers.constants import Constants
from SPPy.solvers.co_ordinates import FVMCoordinates
from SPPy.solvers.electrode_potential import ElectrodePotentialFVMSolver


class TestElectrodePotentialFVMSolver(unittest.TestCase):
    instance_coords = FVMCoordinates(L_n=8e-5, L_s=2.5e-5, L_p=8.8e-5, num_grid_n=5, num_grid_p=5)

    # The instance below has the realistic parameter values for the NMC electrode
    instance_p = ElectrodePotentialFVMSolver(fvm_coords=instance_coords, electrode_type='p',
                                             a_s=39600, sigma_eff=1.57)
    # The instance below has realistic input parameters values for the graphite electrode
    instance_n = ElectrodePotentialFVMSolver(fvm_coords=instance_coords, electrode_type='n',
                                             a_s=340000.0, sigma_eff=56.074)

    def test_constructor(self):
        dx = self.instance_p.coords[2] - self.instance_p.coords[1]
        self.assertEqual(dx, self.instance_p.dx)

    def test_M_phi_s(self):
        actual_p = np.array([[-1., 1., 0., 0., 0.],
                             [1., -2., 1., 0., 0.],
                             [0., 1., -2., 1., 0.],
                             [0., 0., 1., -2., 1.],
                             [0., 0., 0., 1., -3.]])
        actual_n = np.array([[-3., 1., 0., 0., 0.],
                             [1., -2., 1., 0., 0.],
                             [0., 1., -2., 1., 0.],
                             [0., 0., 1., -2., 1.],
                             [0., 0., 0., 1., -1.]])
        self.assertTrue(np.array_equal(actual_p, self.instance_p._M_phi_s))
        self.assertTrue(np.array_equal(actual_n, self.instance_n._M_phi_s))

    def test_array_j(self):
        dx = self.instance_p.dx

        # The flux [mol m-2 s-1] below is when a current of 1A is applied during discharge.
        j = -5.679e-6 * np.ones(5)  # row vector
        self.assertTrue(np.array_equal(dx**2 * Constants.F * self.instance_p.a_s * j.reshape(-1,1) / self.instance_p.sigma_eff,
                                       self.instance_p._array_j(j=j)))

        j = j.reshape(-1, 1)  # column vector
        self.assertTrue(
            np.array_equal(dx ** 2 * Constants.F * self.instance_p.a_s * j.reshape(-1, 1) / self.instance_p.sigma_eff,
                           self.instance_p._array_j(j=j)))

    def test_array_V(self):
        actual_p = np.array([0, 0, 0, 0, 4.2])
        actual_n = np.array([0, 0, 0, 0, 0])

        self.assertTrue(np.array_equal(actual_p, self.instance_p._array_V(terminal_potential=4.2)))
        self.assertTrue(np.array_equal(actual_n, self.instance_n._array_V()))

    def test_solve(self):
        # The flux [mol m-2 s-1] below is when a current of 1A is applied during discharge.
        j_n = 5.679e-6 * np.ones(5)
        j_p = -5.679e-6 * np.ones(5)

        print(self.instance_n.solve_phi_s(j=j_n))
        print(self.instance_p.solve_phi_s(j=j_p, terminal_potential=4.2))
