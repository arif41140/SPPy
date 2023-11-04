__author__ = 'Moin Ahmed'
__copywrite__ = 'Copywrite 2023 by Moin Ahmed. All rights are reserved.'
__status__ = 'deployed'


from typing import Union

import numpy as np


class Thevenin1RC:
    """
    This class creates a first order Thevenin model object for a lithium-ion battery cell. It contains relevant model
    parameters as class attributes and methods to calculate SOC and terminal voltage.

    Thevenin first order model is a phenomenological model that can be used to simulate the terminal voltage across a
    lithium-ion battery cell. It has representative electrical components that represent the open-circuit voltage,
    internal resistance, and diffusion voltages. The set of differential and algebraic equations are:

    dz/dt = -eta(t) * i_app(t) / capacity
    di_R1/dt = -i_R1/(R1*C1) + i_app(t)/(R1*C1)
    v(t) = OCV(z(t)) - R1*i_R1(t) - R0*i_app(t)

    Where the second equation is a non-homogenous linear first-order differential equation. Furthermore, the variables
    are:
    z: state of charge (SOC)
    R0: resistance of the resistor that represents the battery cell's internal resistance
    R1: resistance of the resistor in the RC pair.
    C1: capacitance of the capacitor in the RC pair.
    i_R1: current through R1
    i_app: applied current
    eta: Colombic efficiency

    Note that the RC pair represents the diffusion voltage in the battery cell.


    After time discretization, the set of algebraic equations are:

    z[k+1] = z[k] - delta_t*eta[k]*i_app[k]/capacity
    i_R1[k+1] = exp(-delta_t/(R1*C1))*i_R1[k] + (1-exp(-delta_t/(R1*C1))) * i_app[k]
    v[k] = OCV(z[k]) - R1*i_R1[k] - R0*i_app[k]

    Where k represents the time-point and delta_t represents the time-step between z[k+1] and z[k].

    Code Notes:
    1. It is assumed for now that eta is a function of applied current only.
    2. Discharge currrent is positve and charge current is negative by convention.

    Reference:
    Hariharan, K. S. (2013). A coupled nonlinear equivalent circuit – Thermal model for lithium ion cells.
    In Journal of Power Sources (Vol. 227, pp. 171–176). Elsevier BV.
    https://doi.org/10.1016/j.jpowsour.2012.11.044
    """
    @classmethod
    def soc_next(cls, dt: float, i_app: float, SOC_prev: float, Q: float, eta: float):
        """
        This methods calculates the SOC at the next time-step
        :param dt: time difference between the current and previous time steps [s]
        :param i_app: Applied current [A]
        :param SOC_prev: SOC at the previous time step
        :param Q: battery cell capacity [Ahr]
        :param eta: Columbic efficiency
        :return: SOC at the current time step
        """
        return SOC_prev - dt * eta * i_app / (3600 * Q)

    @classmethod
    def i_R1_next(cls, dt: float, i_app: float, i_R1_prev: float, R1: float, C1: float):
        """
        Measures the current through R1 (i_R1) at the current time step.
        :param dt: time difference between the current and the previous time step [s]
        :param i_app: applied current [A]
        :param i_R1_prev: current through the RC branch at the previous time step [A]
        :param R1: resistance of R1 [ohms]
        :param C1: capacitance of C1 [F]
        :return: current through the RC branch at the current time step
        """
        return np.exp(-dt/(R1*C1)) * i_R1_prev + (1-np.exp(-dt/(R1*C1))) * i_app

    @classmethod
    def v(cls, i_app, OCV: float, R0: float, R1: float, i_R1: float):
        """
        This method calculates the cell terminal voltage.
        :param i_app: (float) applied current at current time step, k
        :return: (float) terminal voltage at the current time step, k
        """
        return OCV - R1 * i_R1 - R0 * i_app


class ESC:
    """
    Class oject contains the relevant functions to perform the Enhanched-self-correcting ECM model.
    Notes:
        The discharge current is assumed to be positive values. Meanwhile, the charge current is negative by convention.
    """

    @classmethod
    def sign(cls, num: Union[int, float]) -> int:
        if num < 0:
            return -1
        elif num == 0:
            return 0
        else:
            return 1

    @classmethod
    def s(cls, i_app: float, s_prev: float) -> float:
        """
        Returns the value for the instantaneous hysteresis
        :param i_app: Applied battery cell current [A]
        :return: value for the instantaneous hysteresis
        """
        if abs(i_app) > 0:
            return ESC.sign(num=i_app)
        else:
            return s_prev

    @classmethod
    def soc_next(cls, dt: float, i_app: float, SOC_prev: float, Q: float, eta: float):
        """
        This methods calculates the SOC at the next time-step
        :param dt: time difference between the current and previous time steps [s]
        :param i_app: Applied current [A]
        :param SOC_prev: SOC at the previous time step
        :param Q: battery cell capacity [Ahr]
        :param eta: Columbic efficiency
        :return: SOC at the current time step
        """
        return Thevenin1RC.soc_next(dt=dt, i_app=i_app, SOC_prev=SOC_prev, Q=Q, eta=eta)

    @classmethod
    def i_R1_next(cls, dt: float, i_app: float, i_R1_prev: float, R1: float, C1: float):
        """
        Measures the current through R1 (i_R1) at the current time step.
        :param dt: time difference between the current and the previous time step [s]
        :param i_app: applied current [A]
        :param i_R1_prev: current through the RC branch at the previous time step [A]
        :param R1: resistance of R1 [ohms]
        :param C1: capacitance of C1 [F]
        :return: current through the RC branch at the current time step
        """
        return Thevenin1RC.i_R1_next(dt=dt, i_app=i_app, i_R1_prev=i_R1_prev, R1=R1, C1=C1)

    @classmethod
    def h_next(cls, dt: float, i_app: float, eta: float, gamma: float, cap: float, h_prev: float) -> float:
        exp_term = np.exp(-np.abs((eta * i_app * gamma * dt) / (3600 * cap)))
        return exp_term * h_prev - (1 - exp_term) * ESC.sign(i_app)

    @classmethod
    def v(cls, i_app, ocv: float, R0: float, R1: float, i_R1: float,
          m_0: float, m: float, h: float, s_prev: float) -> float:
        return ocv - R1 * i_R1 - R0 * i_app + m * h + m_0 * ESC.s(i_app=i_app, s_prev=s_prev)


