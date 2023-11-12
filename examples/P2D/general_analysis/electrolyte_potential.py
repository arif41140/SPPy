import numpy as np
import matplotlib.pyplot as plt
import scipy

from SPPy.solvers.co_ordinates import FVMCoordinates
from SPPy.solvers.electrolyte_potential import ElectrolytePotentialFVMSolver


# Parameters values are obtained from the Shangwoo et al.
L_n = 81e-6
L_s = 20e-6
L_p = 78e-6
L_tot = L_n + L_s + L_p

epsilon_en = 0.264
epsilon_esep = 0.46
epsilon_ep = 0.281
e_s_n = 0.68
e_s_p = 0.65
R_n = 6e-6
R_p = 5e-6
a_s_n = 3 * e_s_n / R_n
a_s_p = 3 * e_s_p / R_p
t_c = 0.38
kappa_e = 1.194
brugg = 1.5
temp = 298.15

instance_coords = FVMCoordinates(L_n=L_n, L_s=L_s, L_p=L_p, num_grid_n=5, num_grid_s=5, num_grid_p=5)
instance_solver = ElectrolytePotentialFVMSolver(fvm_coords=instance_coords,
                                                epsilon_en=epsilon_en, epsilon_esep=epsilon_esep,
                                                epsilon_ep=epsilon_ep,
                                                a_s_n=a_s_n, a_s_p=a_s_p,
                                                t_c=t_c, kappa_e=kappa_e, brugg=brugg, temp=temp)

array_c_e = 1000 * np.ones(len(instance_coords.array_x))

j_n = 2.51e-5 * np.ones(5)
j_p = -2.27e-5 * np.ones(5)
array_j = np.append(j_n, np.append(np.zeros(5), j_p))

terminal_phi_e, array_phi_e, array_rel_phi_e = instance_solver.solve_phi_e(j=array_j, c_e=array_c_e)

plt.plot(instance_coords.array_x, array_rel_phi_e)
plt.show()
