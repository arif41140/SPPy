import numpy as np

from SPPy.calc_helpers.constants import Constants
from SPPy.models.battery import P2DM
from SPPy.solvers.co_ordinates import FVMCoordinates
from SPPy.solvers.electrode_potential import ElectrodePotentialFVMSolver
from SPPy.solvers.electrolyte_potential import ElectrolytePotentialFVMSolver

from examples.P2D.general_analysis.parameters_Shangwoo import ocp_p, ocp_n

# Parameters values are obtained from the Shangwoo et al.
L_n = 81e-6
L_s = 20e-6
L_p = 78e-6

e_s_n = 0.68
e_s_p = 0.65
R_n = 6e-6
R_p = 5e-6
a_s_n = 3 * e_s_n / R_n
a_s_p = 3 * e_s_p / R_p
epsilon_en = 0.264
epsilon_esep = 0.46
epsilon_ep = 0.281
k_n = 2.3e-10
k_p = 1.43e-10
c_s_max_n = 31221
c_s_max_p = 50179

area_cell = 0.06  # in m2

t_c = 0.38
kappa_e = 1.194
brugg = 1.5
temp = 298.15

# Simulation Parameters below
num_grid_n = 5
num_grid_s = 5
num_grid_p = 5
total_num_grid = num_grid_n + num_grid_s + num_grid_p

i_app = 4.0  # in A. This value is roughly equal to 1.5C according to Shangwoo's parameters.
soc_init_n, soc_init_p = 0.98, 0.35
array_c_e_init = array_c_e = 1000 * np.ones(total_num_grid)

array_ocp_n = ocp_n(soc_init_n) * np.ones(5)
array_ocp_p = ocp_p(soc_init_p) * np.ones(5)

array_c_s_n = soc_init_n * np.ones(num_grid_n)
array_c_s_p = soc_init_p * np.ones(num_grid_p)

# solvers instances below
coords = instance_coords = FVMCoordinates(L_n=L_n, L_s=L_s, L_p=L_p, num_grid_n=5, num_grid_s=5, num_grid_p=5)
electrode_phi_n = ElectrodePotentialFVMSolver(fvm_coords=instance_coords, electrode_type='n',
                                              a_s=a_s_n, sigma_eff=56.074)
electrode_phi_p = ElectrodePotentialFVMSolver(fvm_coords=instance_coords, electrode_type='p',
                                              a_s=a_s_p, sigma_eff=1.57)
instance_solver = ElectrolytePotentialFVMSolver(fvm_coords=instance_coords,
                                                epsilon_en=epsilon_en, epsilon_esep=epsilon_esep,
                                                epsilon_ep=epsilon_ep,
                                                a_s_n=a_s_n, a_s_p=a_s_p,
                                                t_c=t_c, kappa_e=kappa_e, brugg=brugg, temp=temp)

i_n_v = i_app / (area_cell * L_n)  # in A/m3
i_p_v = -i_app / (area_cell * L_p)  # in A/m3
j_n = i_n_v / (Constants.F * a_s_n)  # in mol/m2
j_p = i_p_v / (Constants.F * a_s_p)  # in mol/m2

array_j_n = j_n * np.ones(5)
array_j_p = j_p * np.ones(5)
array_j = np.append(array_j_n, np.append(np.zeros(5), array_j_p))

phi_n = electrode_phi_n.solve_phi_s(j=array_j_n, terminal_potential=0.0)
phi_p = electrode_phi_p.solve_phi_s(j=array_j_p, terminal_potential=0.0)
terminal_phi_e, array_phi_e, array_rel_phi_e = instance_solver.solve_phi_e(j=array_j, c_e=array_c_e)

eta_n_rel = np.ndarray.flatten(phi_n)-array_phi_e[:5]-array_ocp_n
i_0_n = P2DM.i_0(k=k_n, c_s_surf=array_c_s_n, c_s_max=c_s_max_n,
                 c_e=array_c_e[:num_grid_n], c_e_0=array_c_e_init[:num_grid_n])
v_e = P2DM.v_n_minus_v_e(array_i_0=i_0_n, array_eta_rel_n=eta_n_rel, array_coord_n=coords.array_x_n,
                         i_app=i_app, temp=temp, a_s_n=a_s_n, cell_area=area_cell, L_n=L_n)
print(eta_n_rel, i_0_n)
print(coords.array_x_n)
print(a_s_n)
print(v_e)

