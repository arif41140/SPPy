import numpy as np


def ocp_p(soc_n: float):
    return -10.74*soc_n**4 + 23.88*soc_n**3 - 16.77*soc_n**2 + 2.595*soc_n + 4.563


def ocp_n(soc_n: float):
    return 0.1493 + 0.8493 * np.exp(-61.79 * soc_n) + 0.3824 * np.exp(-665.8 * soc_n) - np.exp(39.42 * soc_n - 41.92) - \
           0.03131 * np.arctan(25.59 * soc_n - 4.099) - 0.009434 * np.arctan(32.49 * soc_n - 15.74)

