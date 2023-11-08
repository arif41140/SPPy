import numpy as np
import numpy.typing as npt

from SPPy.calc_helpers.constants import Constants
from SPPy.solvers.co_ordinates import FVMCoordinates


class ElectrolytePotentialFVMSolver:
    def __int__(self, fvm_coords: FVMCoordinates,
                epsilon_en: float, epsilon_esep: float, epsilon_ep: float,
                t_c: float, kappa_e: float, brugg: float, temp: float):
        self.coords = fvm_coords

        self.epsilon_en = epsilon_en
        self.epsilon_esep = epsilon_esep
        self.epsilon_ep = epsilon_ep

        self.t_c = t_c
        self.kappa_e = kappa_e
        self.temp = temp
        self.kappa_D = 2 * Constants.R * self.temp * self.kappa_e * (self.t_c - 1) / Constants.F
        self.brugg = brugg

        n: int = len(self.coords.array_x)  # the number of rows and columns of the matrixself.

    @property
    def array_epsilon_e(self) -> npt.ArrayLike:
        """
        Returns an array containing the volume fraction of the electrolyte at each spatial region
        :return:
        """
        array_epsilon_n = self.epsilon_en * np.ones(len(self.coords.array_x_n))
        array_epsilon_s = self.epsilon_esep * np.ones(len(self.coords.array_x_s))
        array_epsilon_p = self.epsilon_ep * np.ones(len(self.coords.array_x_p))
        return np.append(np.append(array_epsilon_n, array_epsilon_s), array_epsilon_p)

    @property
    def array_kappa_eff(self) -> npt.ArrayLike:
        """
        Returns an array containing the effective electrolyte conductivity at spatial FVM points.
        :return:
        """
        array_D_eff_n = self.kappa_e * (self.epsilon_en ** self.brugg) * np.ones(len(self.coords.array_x_n))
        array_D_eff_s = self.kappa_e * (self.epsilon_esep ** self.brugg) * np.ones(len(self.coords.array_x_s))
        array_D_eff_p = self.kappa_e * (self.epsilon_ep ** self.brugg) * np.ones(len(self.coords.array_x_p))
        return np.append(np.append(array_D_eff_n, array_D_eff_s), array_D_eff_p)

    @property
    def array_kappa_D_eff(self) -> npt.ArrayLike:
        """
        Returns an array containing the effective kappa_D at spatial FVM points.
        :return:
        """
        array_D_eff_n = self.kappa_D * (self.epsilon_en ** self.brugg) * np.ones(len(self.coords.array_x_n))
        array_D_eff_s = self.kappa_D * (self.epsilon_esep ** self.brugg) * np.ones(len(self.coords.array_x_s))
        array_D_eff_p = self.kappa_D * (self.epsilon_ep ** self.brugg) * np.ones(len(self.coords.array_x_p))
        return np.append(np.append(array_D_eff_n, array_D_eff_s), array_D_eff_p)

    def _M_phi_e(self):
        diag_elements = np.zeros(self.n)
        lower_diag_elements = np.zeros(self.n-1)
        upper_diag_elements = np.zeros(self.n-1)
        # setup first row
        k_eff1 = (self.array_kappa_eff[0] + self.array_kappa_eff[1]) / 2
        dx1 = self.coords.array_dx[0]
        diag_elements[0] = -k_eff1 / dx1
        upper_diag_elements[0] = k_eff1 / dx1
        for i in range(1, self.n-1):
            dx1 = self.coords.array_dx[i-1]
            dx2 = self.coords.array_dx[i]
            k_eff1 = (self.array_kappa_eff[i] + self.array_kappa_eff[i-1])/2
            k_eff2 = (self.array_kappa_eff[i+1] + self.array_kappa_eff[i])/2
            A = k_eff1 / dx1
            B = k_eff2 / dx2
            diag_elements[i] = -(A+B)
            lower_diag_elements[i-1] = A
            upper_diag_elements[i] = B
        # set elements for the last row
        k_eff1 = (self.array_kappa_eff[-2] + self.array_kappa_eff[-1]) / 2
        dx1 = self.coords.array_dx
        diag_elements[-1] = -k_eff1 / dx1
        lower_diag_elements[-1] = k_eff1 / dx1
        m_ = np.diag(diag_elements) + np.diag(upper_diag_elements, 1) + np.diag(lower_diag_elements, -1)
        return m_

    def vec_phi_e(self, j: npt.ArrayLike) -> npt.ArrayLike:
        col_vec = np.zeros(self.n)
        col_vec_j = -j
        col_vec_j[self.b_cell.electrolyte.node_x_n] = col_vec_j[self.b_cell.electrolyte.node_x_n] \
                                                      * self.b_cell.n_elec.dx
        col_vec_j[self.b_cell.electrolyte.node_x_p] = col_vec_j[
                                                          self.b_cell.electrolyte.node_x_p] * self.b_cell.p_elec.dx
        dx = self.b_cell.electrolyte.x[1] - self.b_cell.electrolyte.x[0]
        kappa_D = (self.b_cell.electrolyte.kappa_D_eff()[1] + self.b_cell.electrolyte.kappa_D_eff()[0])/2
        c1 = self.b_cell.electrolyte.c_e[0]
        c2 = self.b_cell.electrolyte.c_e[1]
        col_vec[0] = (kappa_D / dx) * ((c2-c1) / (c2 + c1))
        for i in range(1, len(col_vec)-1):
            dx1 = self.b_cell.electrolyte.x[i] - self.b_cell.electrolyte.x[i-1]
            dx2 = self.b_cell.electrolyte.x[i+1] - self.b_cell.electrolyte.x[i]
            kappa_D1 = (self.b_cell.electrolyte.kappa_D_eff()[i] + self.b_cell.electrolyte.kappa_D_eff()[i-1]) / 2
            kappa_D2 = (self.b_cell.electrolyte.kappa_D_eff()[i+1] + self.b_cell.electrolyte.kappa_D_eff()[i]) / 2
            c1 = self.b_cell.electrolyte.c_e[i-1]
            c2 = self.b_cell.electrolyte.c_e[i]
            c3 = self.b_cell.electrolyte.c_e[i+1]
            col_vec[i] = (kappa_D2 / dx2) * ((c3 - c2) / (c3 + c2)) - (kappa_D1 / dx1) * ((c2-c1) / (c2 + c1))
        # update the last col entry
        dx = self.b_cell.electrolyte.x[-1] - self.b_cell.electrolyte.x[-2]
        kappa_D = (self.b_cell.electrolyte.kappa_D_eff()[-2] + self.b_cell.electrolyte.kappa_D_eff()[-1])/2
        c1 = self.b_cell.electrolyte.c_e[-2]
        c2 = self.b_cell.electrolyte.c_e[-1]
        col_vec[-1] = -(kappa_D / dx) * ((c2-c1) / (c2 + c1))
        return 2 * col_vec.reshape(-1,1) + col_vec_j.reshape(-1,1)

    def solve_phi_e(self, j, V_e):
        M = np.linalg.inv(self._M_phi_e())
        b = self.vec_phi_e(j=j)
        vec_V_e = np.zeros(len(self.b_cell.electrolyte.x)).reshape(-1,1)
        vec_V_e[-1,0] = V_e
        b = b - 2* vec_V_e
        return np.ndarray.flatten(M@b)