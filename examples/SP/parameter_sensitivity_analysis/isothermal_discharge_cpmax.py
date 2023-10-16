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
lst_sol = []
for i in range(1, 7):
    cell = SPPy.BatteryCell.read_from_parametersets(parameter_set_name='test',
                                                    SOC_init_p=SOC_init_p, SOC_init_n=SOC_init_n, temp_init=T)
    cell.elec_p.max_conc = i * 10000
    dc = SPPy.Discharge(discharge_current=I, V_min=V_min, SOC_LIB_min=SOC_min, SOC_LIB=SOC_LIB)
    solver = SPPy.SPPySolver(b_cell=cell, N=5, isothermal=True, degradation=False, electrode_SOC_solver='poly')
    sol = solver.solve(cycler_instance=dc, sol_name=f"$c_{{s,p,max}}={cell.elec_p.max_conc}$")
    lst_sol.append(sol)

# Plot
SPPy.Plots(lst_sol[0],
           lst_sol[1],
           lst_sol[2],
           lst_sol[3],
           lst_sol[4],
           lst_sol[5]).comprehensive_plot()