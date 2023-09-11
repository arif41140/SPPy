import SPPy

import matplotlib.pyplot as plt


# Operating parameters
I = 1.65
T = 298.15
V_min = 2.5
SOC_min = 0.1
SOC_LIB = 1

# Modelling parameters
SOC_init_p, SOC_init_n = 0.4956, 0.7568  # conditions in the literature source. Guo et al

# Setup battery components
cell = SPPy.BatteryCell(parameter_set_name='test', SOC_init_p=SOC_init_p, SOC_init_n=SOC_init_n, T=T)


# set-up cycler and solver
dc = SPPy.DischargeRest(discharge_current=I, V_min=V_min, SOC_LIB=SOC_LIB, SOC_LIB_min=SOC_min, SOC_LIB_max=1,
                        rest_time=500)
solver_poly = SPPy.SPPySolver(b_cell=cell, N=5, isothermal=True, degradation=False, electrode_SOC_solver='poly')


# simulate
sol_poly = solver_poly.solve(cycler_instance=dc)

# Plot
fig = plt.figure(figsize=(10, 3), dpi=300)
ax1 = fig.add_subplot(121)
ax1.plot(sol_poly.t, sol_poly.I)
ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Current [A]')

ax2 = fig.add_subplot(122)
ax2.plot(sol_poly.t, sol_poly.V, label="Polynomial Approximation", linewidth=2)
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Cell Terminal Voltage [V]')

plt.tight_layout()
plt.savefig('C:\\Users\\Moin\\Desktop\\diffusion_voltage_example.png')
plt.show()
