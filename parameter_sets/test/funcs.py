from typing import Union

import numpy as np
import numpy.typing as npt


def OCP_ref_p(SOC):
    return 4.04596 + np.exp(-42.30027 * SOC + 16.56714) - 0.04880 * np.arctan(50.01833 * SOC - 26.48897) \
              - 0.05447 * np.arctan(18.99678 * SOC - 12.32362) - np.exp(78.24095 * SOC - 78.68074)


def dOCPdT_p(SOC):
    num = -0.19952 + 0.92837*SOC - 1.36455 * SOC ** 2 + 0.61154 * SOC ** 3
    dem = 1 - 5.66148 * SOC + 11.47636 * SOC**2 - 9.82431 * SOC**3 + 3.04876 * SOC**4
    return (num/dem) * 1e-3 # since the original unit are of mV/K


def OCP_ref_n(SOC):
    return 0.13966 + 0.68920 * np.exp(-49.20361 * SOC) + 0.41903 * np.exp(-254.40067 * SOC) \
            - np.exp(49.97886 * SOC - 43.37888) - 0.028221 * np.arctan(22.52300 * SOC - 3.65328) \
            -0.01308 * np.arctan(28.34801* SOC - 13.43960)


def dOCPdT_n(SOC):
    num = 0.00527 + 3.29927 * SOC - 91.79326 * SOC ** 2 + 1004.91101 * SOC ** 3 - \
          5812.27813 * SOC ** 4 + 19329.75490 * SOC ** 5 - 37147.89470 * SOC ** 6 + \
          38379.18127 * SOC ** 7 - 16515.05308 * SOC ** 8
    dem = 1 - 48.09287 * SOC + 1017.23480 * SOC**2 - 10481.80419 * SOC**3 + \
          59431.30001 * SOC**4 - 195881.64880 * SOC**5 + 374577.31520 * SOC**6 - \
          385821.16070 * SOC**7 + 165705.85970 * SOC**8
    return (num/dem) * 1e-3 # since the original unit are of mV/K


def func_D_e(c_e: Union[float, npt.ArrayLike], temp: float) -> float:
    """
    Calculates the lithium-ion diffusivity in the electrolyte as a func of lithium-ion concentration and temperature
    Reference: Han et al. A numerically efficient method for solving the full-order pseudo-2D Li-ion cell model.
    2021. Journal of Power Sources. 490

    :param c_e: lithium-ion concentration [mol/m3]
    :param temp: electrolyte temp [K]
    :return: (float) diffusivity of the lithium-ion [m2/s]
    """
    c_e = 0.001 * c_e  # in the original work the concentration was in mol/l
    return (10 ** (-4.43 - 54 / (temp - (229+5*c_e)) - 0.22 * c_e)) * 1e-4  # the original D_e was in cm2/s


def func_kappa_e(c_e: Union[float, npt.ArrayLike], temp: float) -> Union[float, npt.ArrayLike]:
    """
    Calculates the conductivity [S/m] for the electrolyte.
    Reference: Han et al. A numerically efficient method for solving the full-order pseudo-2D Li-ion cell model.
    2021. Journal of Power Sources. 490

    :param c_e:
    :param temp:
    :return:
    """
    c_e = c_e * 0.001  # Original work used the units of mol/l for the concentrations
    kappa_e_ = c_e * (
                -10.5 + 0.074 * temp - 6.96e-5 * (temp ** 2) + 0.668 * c_e - 0.0178 * c_e * temp + 2.8e-5 * c_e * (
                    temp ** 2) + 0.494 * (c_e ** 2) - \
                8.86e-4 * (c_e ** 2) * temp) ** 2
    return kappa_e_ * 1e-3 * 100  # Original work used the units of mS/cm for the conductivity


def func_dlnf(c_e: Union[float, npt.ArrayLike], temp: float, t_c: float) -> Union[float, npt.ArrayLike]:
    """
    Calculates the expression 1+dlnf/dlnc_e

    Reference: Han et al. A numerically efficient method for solving the full-order pseudo-2D Li-ion cell model.
    2021. Journal of Power Sources. 490
    :param c_e:
    :param temp:
    :param t_c:
    :return:
    """
    c_e = c_e * 0.001
    return (0.601-0.24*c_e+(0.982-5.1064e-3*(temp-294.15))*c_e**1.5)/(1-t_c)


