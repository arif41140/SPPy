"""
Note the performance for SPM is not great at high C-rates
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import SPPy
from SPPy.solution import Solution


sol_exp = Solution.upload_exp_data(file_name='BJDST_25.csv')
t_exp = sol_exp.t[np.argwhere(sol_exp.cycle_step > 3)].flatten()
t_exp = t_exp - t_exp[0]
V_exp = sol_exp.V[np.argwhere(sol_exp.cycle_step > 3)].flatten()
I_exp = sol_exp.I[np.argwhere(sol_exp.cycle_step > 3)].flatten()

# Modelling parameters
SOC_n_min=0.01073305
SOC_n_max=0.75
SOC_p_min=0.41553058
SOC_p_max=0.90207103

SOC_init_p, SOC_init_n = SOC_p_min, SOC_n_max
T = 298.15  # in K

# Setup cycler and battery cell
cycler = SPPy.CustomCycler(t_array=t_exp, I_array=I_exp, SOC_LIB=1.0)
cycler.plot()
cell = SPPy.BatteryCell(parameter_set_name='Calce_NMC_18650', SOC_init_p=SOC_init_p, SOC_init_n=SOC_init_n, T=T)
solver = SPPy.SPPySolver(b_cell=cell, N=5, isothermal=True, degradation=False, electrode_SOC_solver='poly')

# simulate and plot
sol = solver.solve(cycler_instance=cycler, verbose=True, t_increment=1)

sol_exp = Solution.upload_exp_data(file_name='FUDS_25.csv')
plt.plot(t_exp, V_exp)
plt.plot(sol.t, sol.V)
plt.show()

