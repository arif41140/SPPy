"""
This script contains the example usage of the single particle model for the discharge operation.
"""

__author__ = 'Moin Ahmed'
__copywrite__ = 'Copywrite 2023 by Moin Ahmed. All rights reserved.'
__status__ = 'deployed'

import SPPy

# Operating parameters
I = 1.656
T = 298.15
V_min = 3
SOC_min = 0.1
SOC_LIB = 0.9

# Modelling parameters
SOC_init_p, SOC_init_n = 0.4956, 0.7568  # conditions in the literature source. Guo et al

# Setup battery components
cell = SPPy.BatteryCell.read_from_parametersets(parameter_set_name='test', SOC_init_p=SOC_init_p, SOC_init_n=SOC_init_n,
                                                temp_init=T)

# set-up cycler and solver
dc = SPPy.Discharge(discharge_current=I, v_min=V_min, SOC_LIB_min=SOC_min, SOC_LIB=SOC_LIB)
solver = SPPy.SPPySolver(b_cell=cell, N=5, isothermal=True, degradation=False, electrode_SOC_solver='poly')

# simulate
sol = solver.solve(cycler_instance=dc)

print(sol.cycle_summary)

# Plot
sol.comprehensive_isothermal_plot()
