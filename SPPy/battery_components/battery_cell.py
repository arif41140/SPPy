from dataclasses import dataclass
from typing import Callable

import numpy as np

from SPPy.calc_helpers import constants
from SPPy.battery_components.parameter_set_manager import ParameterSets
from SPPy.battery_components import electrolyte, electrode


@dataclass
class BatteryCellBase:
    T_: float  # battery cell temperature, K
    rho: float  # battery density (mostly for thermal modelling), kg/m3
    Vol: float  # battery cell volume, m3
    C_p: float  # specific heat capacity, J / (K kg)
    h: float  # heat transfer coefficient, J / (S K)
    A: float  # surface area, m2
    cap: float  # capacity, Ah
    V_max: float  # maximum potential
    V_min: float  # minimum potential

    elec_p: electrode.PElectrode  # electrode class object
    elec_n: electrode.NElectrode  # electrode class object
    electrolyte: electrolyte.Electrolyte  # electrolyte class object

    def __post_init__(self):
        # self.T_ = self.T
        self.T_amb_ = self.T  # initial condition
        # initialize internal cell resistance
        self.R_cell = (self.elec_p.L / self.elec_p.kappa_eff + self.electrolyte.L / self.electrolyte.kappa_eff + \
                       self.elec_n.L / self.elec_n.kappa_eff) / self.elec_n.A
        self.R_cell_init = self.R_cell

    @property
    def T(self):
        return self.T_

    @T.setter
    def T(self, new_T):
        self.T_ = new_T
        self.elec_p.T = new_T
        self.elec_n.T = new_T

    @property
    def T_amb(self):
        return self.T_amb_


class BatteryCell(BatteryCellBase):
    """
    Class for the BatteryCell object and contains the relevant parameters.
    """
    def __init__(self, parameter_set_name: str, SOC_init_p: float, SOC_init_n: float, T: float):
        param_set = ParameterSets(name=parameter_set_name)
        df = ParameterSets.parse_csv(file_path=param_set.BATTERY_CELL_DIR)
        rho = df['Density [kg m^-3]']
        Vol = df['Volume [m^3]']
        C_p = df['Specific Heat [J K^-1 kg^-1]']
        h = df['Heat Transfer Coefficient [J s^-1 K^-1]']
        A = df['Surface Area [m^2]']
        cap = df['Capacity [A hr]']
        V_max = df['Maximum Potential Cut-off [V]']
        V_min = df['Minimum Potential Cut-off [V]']
        # initialize electrodes and electrolyte
        obj_elec_p = electrode.NElectrode(L=param_set.L_p, A=param_set.A_p, kappa=param_set.kappa_p,
                                          epsilon=param_set.epsilon_p, S=param_set.S_p, max_conc=param_set.max_conc_p,
                                          R=param_set.R_p, k_ref=param_set.k_ref_p, D_ref=param_set.D_ref_p,
                                          Ea_R=param_set.Ea_R_p, Ea_D=param_set.Ea_D_p, alpha=param_set.alpha_p,
                                          T_ref=param_set.T_ref_p, brugg=param_set.brugg_p,
                                          func_OCP=param_set.OCP_ref_p_, func_dOCPdT=param_set.dOCPdT_p_,
                                          SOC_init=SOC_init_p, T=T)
        obj_elec_n = electrode.NElectrode(L=param_set.L_n, A=param_set.A_n, kappa=param_set.kappa_n,
                                          epsilon=param_set.epsilon_n, S=param_set.S_n, max_conc=param_set.max_conc_n,
                                          R=param_set.R_n, k_ref=param_set.k_ref_n, D_ref=param_set.D_ref_n,
                                          Ea_R=param_set.Ea_R_n, Ea_D=param_set.Ea_D_n, alpha=param_set.alpha_n,
                                          T_ref=param_set.T_ref_n, brugg=param_set.brugg_n,
                                          func_OCP=param_set.OCP_ref_n_, func_dOCPdT=param_set.dOCPdT_n_,
                                          U_s=param_set.U_s, i_s=param_set.i_s, MW_SEI=param_set.MW_SEI,
                                          rho_SEI=param_set.rho_SEI, kappa_SEI=param_set.kappa_SEI,
                                          SOC_init=SOC_init_n, T=T)
        obj_electrolyte = electrolyte.Electrolyte(L=param_set.L_es, conc=param_set.conc_es, kappa=param_set.kappa_es,
                                                  epsilon=param_set.epsilon_es, brugg=param_set.brugg_es)
        super().__init__(T_=T, rho=rho, Vol=Vol, C_p=C_p, h=h, A=A, cap=cap, V_max=V_max, V_min=V_min,
                         elec_p=obj_elec_p, elec_n=obj_elec_n, electrolyte=obj_electrolyte)


@dataclass
class ECMBatteryCell:
    R0_ref: float  # resistance value of R0 [ohm]
    R1_ref: float  # resistance value of R1 [ohm]
    C1: float  # capacitance of capacitor in RC circuit [ohm]
    T_ref: float  # reference temperature for R0_ref and R1_ref
    Ea_R0: float  # activation energy for R0 [J/mol]
    Ea_R1: float  # activation energy for R1 [J/mol]

    rho: float  # battery density (mostly for thermal modelling), kg/m3
    Vol: float  # battery cell volume, m3
    C_p: float  # specific heat capacity, J / (K kg)
    h: float  # heat transfer coefficient, J / (S K)
    A: float  # surface area, m2
    cap: float  # capacity, Ah
    V_max: float  # maximum potential
    V_min: float  # minimum potential

    SOC_init: float  # initial SOC
    T_init: float  # initial battery cell temperature, K

    func_eta: Callable  # func for the Columbic efficiency as a func of SOC and temp
    func_OCV: Callable  # func which outputs the battery OCV from its SOC
    func_dOCVdT: Callable  # function which outputs the change of OCV with respect to temperature from its SOC

    def __post_init__(self):
        self.T_ = self.T_init
        self.SOC_ = self.SOC_init

    @property
    def T(self):
        """
        Represents the current temperature of the battery cell [K]
        :return: (float) current battery cell temperature [K]
        """
        return self.T_

    @T.setter
    def T(self, T_new: float):
        self.T_ = T_new

    @property
    def SOC(self):
        """
        Represents the current battery cell SOC
        :return: (float) returns the current battery cell SOC
        """
        return self.SOC_

    @SOC.setter
    def SOC(self, SOC_new: float):
        self.SOC_ = SOC_new

    @property
    def R0(self):
        return self.R0_ref * np.exp(-1 * self.Ea_R0 / constants.Constants.R * (1 / self.T - 1 / self.T_ref))

    @property
    def R1(self):
        return self.R0_ref * np.exp(-1 * self.Ea_R1 / constants.Constants.R * (1 / self.T - 1 / self.T_ref))

    @property
    def dOCPdT(self):
        return self.func_dOCVdT(self.SOC)

    @property
    def OCV(self):
        return self.func_OCV(self.SOC) + self.dOCPdT * (self.T - self.T_ref)

    @property
    def eta(self):
        return self.func_eta(self.SOC, self.T)
