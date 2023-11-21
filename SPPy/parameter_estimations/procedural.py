from typing import Union, Callable, Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import scipy

import SPPy
from SPPy.calc_helpers.computational_intelligence_algorithms import GA


class OCVData:
    """
    This class finds the stociometric limits of the positive and the negative electrodes using the low C-rate
    battery cycling.
    Optimization is performed whereby the fitted OCV is compared with the experimental results
    """

    def __init__(self, func_ocp_p: Callable, func_ocp_n: Callable,
                 soc_n_min_1: float, soc_n_min_2: float,
                 soc_n_max_1: float, soc_n_max_2: float,
                 soc_p_min_1: float, soc_p_min_2: float,
                 soc_p_max_1: float, soc_p_max_2: float,
                 charge_or_discharge: str):
        self.func_ocp_p = func_ocp_p
        self.func_ocp_n = func_ocp_n

        self.SOC_N_MIN_1 = soc_n_min_1
        self.SOC_N_MIN_2 = soc_n_min_2
        self.SOC_N_MAX_1 = soc_n_max_1
        self.SOC_N_MAX_2 = soc_n_max_2
        self.SOC_P_MIN_1 = soc_p_min_1
        self.SOC_P_MIN_2 = soc_p_min_2
        self.SOC_P_MAX_1 = soc_p_max_1
        self.SOC_P_MAX_2 = soc_p_max_2

        self.cycling_step: str = charge_or_discharge

        self.SOC_LIB_MIN = 0.0
        self.SOC_LIB_MAX = 1.0

    @property
    def array_soc_lib(self):
        return np.linspace(self.SOC_LIB_MIN, self.SOC_LIB_MAX)

    @staticmethod
    def _func_interp_ocp_exp(array_cap_exp: npt.ArrayLike, array_v_exp: npt.ArrayLike):
        return scipy.interpolate.interp1d(array_cap_exp, array_v_exp)

    def array_soc(self, soc_min: float, soc_max: float) -> npt.ArrayLike:
        return np.linspace(soc_min, soc_max)

    def array_ocp_p(self, soc_min: float, soc_max: float) -> npt.ArrayLike:
        array_soc_p = self.array_soc(soc_min=soc_min, soc_max=soc_max)
        if self.cycling_step == 'discharge':
            return self.func_ocp_p(array_soc_p)
        elif self.cycling_step == 'charge':
            return np.flip(self.func_ocp_p(array_soc_p))
        else:
            raise TypeError('Unknown charging step.')

    def array_ocp_n(self, soc_min: float, soc_max: float) -> npt.ArrayLike:
        array_soc_n = self.array_soc(soc_min=soc_min, soc_max=soc_max)
        if self.cycling_step == 'discharge':
            return np.flip(self.func_ocp_n(array_soc_n))
        elif self.cycling_step == 'charge':
            return self.func_ocp_n(array_soc_n)
        else:
            raise TypeError('Unknown charging step.')

    def _func_interp_ocp(self, soc_min: float, soc_max: float, interpolation_type: str) -> Callable:
        if interpolation_type == 'p':
            array_ocp_p_: npt.ArrayLike = self.array_ocp_p(soc_min=soc_min, soc_max=soc_max)
            return scipy.interpolate.interp1d(self.array_soc_lib, array_ocp_p_)
        elif interpolation_type == 'n':
            array_ocp_n_: npt.ArrayLike = self.array_ocp_n(soc_min=soc_min, soc_max=soc_max)
            return scipy.interpolate.interp1d(self.array_soc_lib, array_ocp_n_)

    def ocv_lib(self, ocp_p: float, ocp_n: float) -> float:
        return ocp_p - ocp_n

    def mse(self, array_v_exp: npt.ArrayLike, array_v_fit: npt.ArrayLike) -> float:
        return np.mean((array_v_exp - array_v_fit) ** 2)

    def find_optimized_parameters(self, array_cap_exp: npt.ArrayLike, array_v_exp_: npt.ArrayLike):
        def func_obj(lst_param: list) -> float:
            # extract the params from the parameter set below
            soc_p_min, soc_p_max, soc_n_min, soc_n_max = lst_param[0], lst_param[1], lst_param[2], lst_param[3]

            # calculate the OCV of the LIB below
            array_ocp_p = self._func_interp_ocp(soc_min=soc_p_min, soc_max=soc_p_max,
                                                interpolation_type='p')(self.array_soc_lib)
            array_ocp_n = self._func_interp_ocp(soc_min=soc_n_min, soc_max=soc_n_max,
                                                interpolation_type='n')(self.array_soc_lib)
            array_ocv = self.ocv_lib(ocp_p=array_ocp_p, ocp_n=array_ocp_n)

            # MSE calculation below
            array_v_exp = self._func_interp_ocp_exp(array_cap_exp=array_cap_exp,
                                                    array_v_exp=array_v_exp_)(self.array_soc_lib)
            mse: float = self.mse(array_v_exp=array_v_exp, array_v_fit=array_ocv)
            return mse

        array_bounds: npt.ArrayLike = np.array([[self.SOC_P_MIN_1, self.SOC_P_MIN_2],
                                                [self.SOC_P_MAX_1, self.SOC_P_MAX_2],
                                                [self.SOC_N_MIN_1, self.SOC_N_MIN_2],
                                                [self.SOC_N_MAX_1, self.SOC_N_MAX_2]])
        return GA(n_chromosomes=10, bounds=array_bounds, obj_func=func_obj,
                  n_pool=7, n_elite=3, n_generations=2).solve()[0]

    def plot_fit(self, soc_p_min: float, soc_p_max: float, soc_n_min: float, soc_n_max: float,
                 cap_exp: Optional[npt.ArrayLike] = None, v_exp: Optional[npt.ArrayLike] = None) -> None:
        array_ocp_p = self._func_interp_ocp(soc_min=soc_p_min, soc_max=soc_p_max,
                                            interpolation_type='p')(self.array_soc_lib)
        array_ocp_n = self._func_interp_ocp(soc_min=soc_n_min, soc_max=soc_n_max,
                                            interpolation_type='n')(self.array_soc_lib)
        array_ocv = self.ocv_lib(ocp_p=array_ocp_p, ocp_n=array_ocp_n)

        fig = plt.figure()

        ax1 = fig.add_subplot(111)
        ax1.plot(self.array_soc_lib, array_ocp_p, '--', label=r'${OCP_p}$')
        ax1.plot(self.array_soc_lib, array_ocp_n, '--', label=r'${OCP_n}$')
        ax1.plot(self.array_soc_lib, array_ocv, label=r'$OCV_{LIB}^{fit}$')

        if cap_exp is not None and v_exp is not None:
            ax1.plot(cap_exp, v_exp, label=r'$OCV_{LIB}^{exp}$')

            # MSE calculation below
            array_v_exp = self._func_interp_ocp_exp(array_cap_exp=cap_exp, array_v_exp=v_exp)(self.array_soc_lib)
            mse: float = self.mse(array_v_exp=array_v_exp, array_v_fit=array_ocv)
            ax1.set_title(f'MSE: {mse}')

        ax1.set_xlabel('Cap. [Ahr]')
        ax1.set_ylabel('V [V]')
        ax1.legend()
        plt.show()


class DriveCycleData:
    def __init__(self, b_cell: SPPy.BatteryCell, sol_exp: SPPy.Solution):
        self.b_cell = b_cell

    def func_obj_isothermal(self, lst_params):
        # Modify the battery cell parameters below
        self.b_cell.elec_n.SOC_init = lst_params[0]
        self.b_cell.elec_p.SOC_init = lst_params[1]
        self.b_cell.elec_n.D_ref = lst_params[2]
        self.b_cell.elec_p.D_ref = lst_params[3]
        self.b_cell.elec_n.R = lst_params[4]
        self.b_cell.elec_p.R = lst_params[5]
        self.b_cell.elec_n.k_ref = lst_params[6]
        self.b_cell.elec_p.k_ref = lst_params[7]
        self.b_cell.R_cell = lst_params[8]

        # set up solver and cycler
        cycler = SPPy.CustomCycler(t_array=0, I_array=0, SOC_LIB=1.0)
        solver = SPPy.SPPySolver(b_cell=self.b_cell, N=5, isothermal=True, degradation=False,
                                 electrode_SOC_solver='poly')

    def ga(self):
        pass
