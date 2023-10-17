from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from SPPy.calc_helpers.matrix_operations import TDMAsolver


@dataclass
class ElectrolyteFVMCoordinates:
    L_p: float  # thickness of the positive electrode region [m]
    L_s: float  # thickness of the seperator region [m]
    L_n: float  # thickness of the negative electrode region [m]

    num_grid_p: int = 10  # number of finite volumes in positive electrode region
    num_grid_s: int = 10  # number of finite volumes in the seperator region
    num_grid_n: int = 10  # number of finite volumes in the negative electrode region

    def __post_init__(self):
        self.dx_n = self.L_n / self.num_grid_n  # dx in the negative electrode region
        self.dx_s = self.L_s / self.num_grid_s  # dx in the seperator region
        self.dx_p = self.L_p / self.num_grid_p  # dx in the positive electrode region

    @property
    def array_x_n(self) -> npt.ArrayLike:
        """
        Returns the location of center of the finite volumes in the negative electrode region.
        :return: array containing the centers of the finite volumes.
        """
        return np.arange(self.dx_n/2, self.L_n, self.dx_n)

    @property
    def array_x_s(self) -> npt.ArrayLike:
        """
        Array containing the location of the nodes in the finite volume in the seperator region.
        :return: Array containing the location of the nodes in the finite volume in the seperator region.
        """
        return np.arange(self.L_n + self.dx_s/2, self.L_n + self.L_s, self.dx_s)

    @property
    def array_x_p(self) -> npt.ArrayLike:
        """
        Array containing the locations of the center of the finite volumes in the positive electrode region.
        :return: Array containing the locations of the center of the finite volumes in the positive electrode region.
        """
        return np.arange(self.L_n + self.L_s + self.dx_p/2, self.L_n + self.L_s + self.L_p, self.dx_p)

    @property
    def array_x(self) -> npt.ArrayLike:
        """
        Array containing the locations of the center of the finite volumes.
        :return: Array containing the locations of the center of the finite volumes.
        """
        return np.append(np.append(self.array_x_n, self.array_x_s), self.array_x_p)

    @property
    def array_dx(self) -> npt.ArrayLike:
        """
        Array containing the width of the finite volumes.
        :return: Array containing the width of the finite volumes.
        """
        array_dx_n = self.dx_n * np.ones(len(self.array_x_n))
        array_dx_s = self.dx_s * np.ones(len(self.array_x_s))
        array_dx_p = self.dx_p * np.ones(len(self.array_x_p))
        return np.append(np.append(array_dx_n, array_dx_s), array_dx_p)


class ElectrolyteConcFVMSolver:
    def __init__(self, fvm_co_ords: ElectrolyteFVMCoordinates, c_e_init: float, transference: float,
                 epsilon_en: float, epsilon_esep: float, epsilon_ep: float,
                 a_sn: float, a_sp: float,
                 D_e: float, brugg: float):
        self.co_ords = fvm_co_ords
        self.t_c = transference
        self.c_e_init = c_e_init
        self.array_c_e_ = self.c_e_init * np.ones(len(self.co_ords.array_x))

        self.epsilon_en = epsilon_en
        self.epsilon_esep = epsilon_esep
        self.epsilon_ep = epsilon_ep

        self.a_sn = a_sn
        self.a_sp = a_sp

        self.D_e = D_e
        self.brugg = brugg

    @property
    def array_epsilon_e(self) -> npt.ArrayLike:
        """
        Returns an array containing the volume fraction of the electrolyte at each spatial region
        :return:
        """
        array_epsilon_n = self.epsilon_en * np.ones(len(self.co_ords.array_x_n))
        array_epsilon_s = self.epsilon_esep * np.ones(len(self.co_ords.array_x_s))
        array_epsilon_p = self.epsilon_ep * np.ones(len(self.co_ords.array_x_p))
        return np.append(np.append(array_epsilon_n, array_epsilon_s), array_epsilon_p)

    @property
    def array_D_eff(self) -> npt.ArrayLike:
        """
        Returns an array containing the effective electrolyte diffusivity at spatial FVM points.
        :return:
        """
        array_D_eff_n = self.D_e * (self.epsilon_en ** self.brugg) * np.ones(len(self.co_ords.array_x_n))
        array_D_eff_s = self.D_e * (self.epsilon_esep ** self.brugg) * np.ones(len(self.co_ords.array_x_s))
        array_D_eff_p = self.D_e * (self.epsilon_ep ** self.brugg) * np.ones(len(self.co_ords.array_x_p))
        return np.append(np.append(array_D_eff_n, array_D_eff_s), array_D_eff_p)

    @property
    def array_a_s(self) -> npt.ArrayLike:
        array_asn = self.a_sn * np.ones(len(self.co_ords.array_x_n))
        array_asep = np.zeros(len(self.co_ords.array_x_s))
        array_asp = self.a_sp * np.ones(len(self.co_ords.array_x_p))
        return np.append(np.append(array_asn, array_asep), array_asp)

    @property
    def array_c_e(self) -> npt.ArrayLike:
        return self.array_c_e_

    @array_c_e.setter
    def array_c_e(self, new_array_c_e_prev):
        self.array_c_e_ = new_array_c_e_prev

    def diags(self, dt: float):
        # initialize the diagonals
        diag_elements = []
        upper_diag_elements = []
        lower_diag_elements = []
        # update first elements
        dx = (self.co_ords.array_x[1] - self.co_ords.array_x[0])
        D1 = self.array_D_eff[0]
        D2 = self.array_D_eff[1]
        A = dt / (2 * self.co_ords.array_dx[0])
        diag_elements.append(self.array_epsilon_e[0] + A * (D2 + D1) / dx)
        upper_diag_elements.append(-A * (D2 + D1) / dx)
        for i in range(1, len(self.co_ords.array_x) - 1):
            dx1 = self.co_ords.array_x[i] - self.co_ords.array_x[i - 1]
            dx2 = self.co_ords.array_x[i + 1] - self.co_ords.array_x[i]
            D1 = self.array_D_eff[i - 1]
            D2 = self.array_D_eff[i]
            D3 = self.array_D_eff[i + 1]
            A = dt / (2 * self.co_ords.array_dx[i])
            diag_elements.append(self.array_epsilon_e[i] + A * ((D1 + D2) / dx1 + (D2 + D3) / dx2))
            upper_diag_elements.append(-A * (D3 + D2) / dx2)
            lower_diag_elements.append(-A * (D1 + D2) / dx1)
        # update last elements
        dx = (self.co_ords.array_x[-1] - self.co_ords.array_x[-2])
        D1 = self.array_D_eff[-1]
        D2 = self.array_D_eff[-1]
        A = dt / (2 * self.co_ords.array_dx[-1])
        diag_elements.append(self.array_epsilon_e[-1] + A * (D2 + D1) / dx)
        lower_diag_elements.append(-A * (D2 + D1) / dx)
        return lower_diag_elements, diag_elements, upper_diag_elements

    def M_ce(self, dt) -> npt.ArrayLike:
        l_diag, diag, u_diag = self.diags(dt)
        return np.diag(diag) + np.diag(u_diag, 1) + np.diag(l_diag, -1)

    def ce_j_vec(self, c_prev: npt.ArrayLike, j: npt.ArrayLike, dt: float) -> npt.ArrayLike:
        ce_j_vec_1_ = (c_prev * self.array_epsilon_e).reshape(-1, 1)
        ce_j_vec_2_ = ((1 - self.t_c) * self.array_a_s * j * dt).reshape(-1, 1)
        return ce_j_vec_1_ + ce_j_vec_2_

    def solve_ce(self, j: npt.ArrayLike, dt: float, solver_method: str = 'TDMA') -> None:
        b = self.ce_j_vec(c_prev=self.array_c_e, j=j, dt=dt)
        if solver_method == 'TDMA':
            l_diag, diag, u_diag = self.diags(dt)
            self.array_c_e = TDMAsolver(l_diag=l_diag, diag=diag, u_diag=u_diag, col_vec=b)
        elif solver_method == 'inverse':
            M = np.linalg.inv(self.M_ce(dt=dt))
            self.array_c_e = np.ndarray.flatten(M @ b)
