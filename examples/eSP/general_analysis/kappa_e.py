from typing import Union, Optional

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def func_kappa_e(c_e: Union[float, npt.ArrayLike], temp: float) -> Union[float, npt.ArrayLike]:
    c_e = c_e * 0.001  # Original work used the units of mol/l for the concentrations
    kappa_e_ = c_e * (
                -10.5 + 0.074 * temp - 6.96e-5 * (temp ** 2) + 0.668 * c_e - 0.0178 * c_e * temp + 2.8e-5 * c_e * (
                    temp ** 2) + 0.494 * (c_e ** 2) - \
                8.86e-4 * (c_e ** 2) * temp) ** 2
    return kappa_e_ * 1e-3 * 100  # Original work used the units of mS/cm for the conductivity


temp1 = 298.15
array_c_e = np.linspace(0, 5000)
array_kappa_e1 = func_kappa_e(c_e=array_c_e, temp=temp1)

plt.plot(array_c_e, array_kappa_e1, label=temp1)

plt.legend()
plt.tight_layout()
plt.show()
