import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import interpolate
from SPPy.cycler.base import BaseCycler


class CustomCycler(BaseCycler):
    def __init__(self, t_array: npt.ArrayLike, I_array: npt.ArrayLike, SOC_LIB: float=0.0):
        """
        CustomCycler constructor.
        :param t_array: numpy array containing the time values in sequence [s].
        :param I_array: numpy array containing the current values.
        :param SOC_LIB:
        """
        super().__init__(SOC_LIB=SOC_LIB)
        # check is t_array and I_array are numpy arrays.
        if (not isinstance(t_array, np.ndarray)) and (not isinstance(I_array, np.ndarray)):
            raise TypeError("t_array and I_array needs to be a numpy array.")

        # t_array and I_array needs to be of equal sizes
        if t_array.shape[0] != I_array.shape[0]:
            raise ValueError("t_array and I_array are not of equal sizes.")

        self.t_array = t_array
        self.I_array = I_array

    @property
    def t_max(self):
        """
        Returns the time value at the last iteration.
        :return: (float) time value at the last iteration
        """
        return self.t_array[-1]

    def get_current(self, t: float):
        """
        Returns the current value from the inputted time value. This current value is interpolation based on the
        current value at the previous time step.
        :param t: time [s]
        :returns: current value [A]
        """
        return interpolate.interp1d(self.t_array, self.I_array, kind='previous', fill_value='extrapolate')(t)

    def plot(self):
        """
        Plots the cycler's instance time [s] vs. current [A]. According to the convention, the discharge current is
        negative.
        :return:
        """
        plt.plot(self.t_array, self.I_array)
        plt.xlabel('Time [s]')
        plt.ylabel('I [A]')
        plt.show()
