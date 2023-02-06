import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate


from file_path_variables import *
from data.Calce_PL import funcs
from data.general_OCP.LCO import OCP_ref_p
from data.general_OCP.hard_carbon import OCP_ref_n
from src.battery_components.battery_cell import BatteryCell
from src.models.single_particle_model import SPModel
from src.solvers.eigen_func_exp import EigenFuncExp
from src.cycler.discharge import Discharge, DischargeRest


# function to correct time in exp data
def correct_time(x, t_init):
    return x - t_init

# Calce data
df_exp = pd.read_csv("C:/Users/Moin/PycharmProjects/CalceData/PL/PL21/FirstDischarge.csv")
df_exp = df_exp[df_exp['Time_sec'] > 60]
# df_exp = df_exp[(df_exp['Current_Amp'] != 0)]
# t_init = df_exp['Time_sec'].iloc[0]
# df_exp['Time_sec'] = df_exp['Time_sec'].apply(lambda x: correct_time(x, t_init=t_init))
t_exp = df_exp['Time_sec'].to_numpy()
V_exp = df_exp['Voltage_Volt'].to_numpy()
I_exp = -df_exp['Current_Amp'].to_numpy()


# Operating parameters
T = 298.15
V_min = 2.8
V_max = 4.2
num_cycles = 1
charge_current = 0
discharge_current = 0.75
rest_time = 3600

# Modelling parameters
# SOC_init_p, SOC_init_n = 0.35, 0.7568
SOC_init_p, SOC_init_n = 0.35, 0.7252

# Setup battery components
cell = BatteryCell(filepath_p=TEST_POS_ELEC_DIR, SOC_init_p=SOC_init_p, func_OCP_p= OCP_ref_p,
                        func_dOCPdT_p=funcs.dOCPdT_p, filepath_n = TEST_NEG_ELEC_DIR, SOC_init_n=SOC_init_n,
                        func_OCP_n= funcs.OCP_ref_n, func_dOCPdT_n=funcs.dOCPdT_n,
                        filepath_electrolyte = TEST_ELECTROLYTE_DIR, filepath_cell = TEST_BATTERY_CELL_DIR, T=T)
# cell.R_cell = 0.13389
cell.R_cell = 0.3
# cell.elec_n.max_conc = 40874
cell.elec_n.max_conc = 46000
cell.elec_n.R = 1e-5
cell.elec_p.max_conc = 60000
model = SPModel(isothermal=False, degradation=False)

# set-up solver and solve
# cycler = Discharge(discharge_current=discharge_current, V_min=V_min)
cycler = DischargeRest(discharge_current=discharge_current, rest_time=rest_time, V_min=V_min)
solver = EigenFuncExp(b_cell=cell, b_model=model, N=5)
sol = solver.solve(cycler=cycler, verbose=False, t_increment=1)
tV_func_sim = interpolate.interp1d(sol.t, sol.V)
# V_sim = tV_func_sim(t_exp)

# mse
# print(np.mean(np.square(V_exp-V_sim)))

# Plot
plt.plot(t_exp, V_exp, label="exp")
plt.plot(sol.t, sol.V, label="sim")
plt.legend()
plt.show()
# sol.plot_tV()
# sol.comprehensive_plot()