from typing import Callable
from networkx import selfloop_edges
import numpy as np
import numpy.typing as npt
import math
import scipy as sp


'''Scheme for Electrolyte Potential'''

class ElectrolytePotential:
    def __init__(self, t: float, x: float, ln: float, lp: float, ls: float,elp: float, eln: float, els: float, cl_init: float
                        diff_sn: float, diff_sp:float, diff_ln: float, diff_lp: float, diff_ls: float, tr_num: float) -> None:
        
        # Battary Parameters 
        self.t = t # Time
        self.x = x # Length variable
        self.elp = elp  # Electrolyte Volume fraction coefficient in positive electrode
        self.eln = eln  # Electrolyte Volume fraction coefficient in nagative electrode
        self.els = els # Electrolyte volume fraction coefficinet in seperatior
        self.ln = ln  # length of negative electrode in m
        self.lp = lp  # length of positive electrode in m
        self.ls = ls # Length of seperator
        self.diff_sn = diff_sn # (Solid phase)Diffusion coefficient in negative electrode
        self.diff_sp = diff_sp # (Solid phase)Diffusion coefficient in positive electrode
        self.diff_ln = diff_ln # Liquid phase diffusion coefficenit of negative electrode
        self.diff_lp = diff_lp # Liquid phase diffusion coefficenit of positive electrode
        self.dif_ls = diff_ls # Liquid phase diffusion coefficenit of sepeartor region
        self.tr_num = tr_num # Transferrance number
        self.cl_init = cl_init # Initial Liquid Concentration 
    
    #Length of cell
    def L(self):
        return self.lp + self.ln + self.ls
    
    # Active surface area of posoitive electrode
    def a_n(self):
        pass
    
    # Active surface area of negative electrode
    def a_p(self):
        pass
    
    # Uniform Current density of negative electrode
    def j_n(self):
        pass
    
    # Uniform Current density of positive electrode
    def j_p(self):
        pass
       
    #Involved Constants in Model
    def alpha_in(self):
        return (-( self.ln * self.ls * self.eln ) / ( 2 * self.diff_ls) - ( (self.ls**2) * self.els ) / ( 6 * self.diff_ls) - ( (self.ln**2) * self.eln ) / ( 3 * self.diff_ln)) / \
                (self.eln * self.ln + self.elp * self.lp + self.els * self.ls)
    
    
    def alpha_ip(self):
        return (-( self.lp * self.ls * self.elp ) / ( 2 * self.diff_ls) - ( (self.ls**2) * self.els ) / ( 6 * self.diff_ls) - ( (self.lp **2) * self.elp ) / ( 3 * self.diff_lp)) / \
                (self.eln * self.ln + self.elp * self.lp + self.els * self.ls)
                    
    def A_1(self):
        return self.ln * self.eln *self.alpha_in() + (self.ls * self.ln * self.eln )/(2*self.diff_ls) + ( (self.ln**2) * self.eln ) / ( 3 * self.diff_ln)
    
    def A_2(self):
        return self.lp * self.elp *self.alpha_ip() - ( (self.lp **2) * self.elp ) / ( 3 * self.diff_lp)
    
    def A_3(self):
        return (1-self.tr_num) * self.ln * self.a_n()
    
    def B_1(self):
        return self.lp * self.elp * self.alpha_in()
    
    def B_2(self):
        return  self.lp * self.elp * self.alpha_ip() - ( (self.lp **2) * self.elp ) / ( 3 * self.diff_lp)
    
    def B_3(self):
        return (1-self.tr_num) * self.lp * self.a_p()
    
    def D(self):
        return self.A_1() * self.B_2() - self.A_2() * self.B_1()
    
    # Liquid concentration flux at negative electrode interface
    def qlin(self):
        A1=self.A_1()
        A2=self.A_2()
        A3=self.A_3()
        B1=self.B_1()
        B2=self.B_2()
        B3=self.B_3()
        D=self.D()
        Jn=self.j_n()
        Jp=self.j_p()
        return -2*A2*B1*(-2*A1*A2*B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2) \
                + A2*(2*A3*B1*Jn*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2) \
                - B3*Jp*(A1**2 + A1*(2*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))\
                - 4*A2*B1 + B2**2 - B2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)))\
                + A3*B2*Jn*(A1**2 + A1*(2*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))\
                - 4*A2*B1 + B2**2 - B2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)))*Piecewise((-2*D*exp(-t*(A1 - B2\
                + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))/(2*D))/(A1 - B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)),\
                Ne(A1 - B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2), 0)),\
                (t, True))*exp(t*(A1 - B2 + sqrt(A1**2 + 2*A1*B2 - \
                4*A2*B1 + B2**2))/(2*D))/(D*(A1 + B2 + sqrt(A1**2 + 2*A1*B2 \
                - 4*A2*B1 + B2**2))*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)*(A1**2 \
                + A1*(2*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) - 4*A2*B1 + B2**2 \
                - B2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)))\
                + 2*A2*B1*(A1**2*(A2*B3*Jp - A3*B2*Jn) \
                + A1*(A2*B3*Jp*(2*B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) \
                - A3*B2*Jn*(2*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)))\
                - 4*A2**2*B1*B3*Jp + A2*(2*A3*B1*Jn*(2*B2 \
                - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) + B2**2*B3*Jp - \
                B2*B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) - A3*B2**2*Jn*(B2 -\
                sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)))*Piecewise((-2*D/(A1 - \
                B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)),\
                 Ne(A1, A2*B1/B2)), (-2*B2*D/(A2*B1 - B2**2\
                + B2*sqrt(A2**2*B1**2/B2**2 - 2*A2*B1+ B2**2)), True))*exp(t*(A1 - B2 + sqrt(A1**2 + \
                2*A1*B2 - 4*A2*B1 + B2**2))/(2*D))/(D*(A1 + B2 +\
                sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))*(A1**3 \
                + A1**2*(3*B2 - sqrt(A1**2 + 2*A1*B2- 4*A2*B1 + B2**2)) \
                - A1*(4*A2*B1 - 3*B2**2 + 2*B2*sqrt(A1**2 \
                + 2*A1*B2 - 4*A2*B1 + B2**2)) - 4*A2*B1*(B2 - sqrt(A1**2 \
                + 2*A1*B2 - 4*A2*B1 + B2**2))+ B2**3 - B2**2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))) \
                - A2*(A1**2*B3*Jp - A1*(A3*B1*Jn - B2*B3*Jp + B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))\
                - 2*A2*B1*B3*Jp + A3*B1*Jn*(B2 + sqrt(A1**2 + 2*A1*B2 \
                - 4*A2*B1 + B2**2)))*Piecewise((2*D/(-A1 + B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)),\
                Ne(A1, A2*B1/B2)), (2*B2*D/(-A2*B1 + B2**2 +B2*sqrt(A2**2*B1**2/B2**2 - 2*A2*B1 + B2**2)), True))*exp(-t*(-A1 + B2 + sqrt(A1**2 + 2*A1*B2 \
                - 4*A2*B1 + B2**2))/(2*D))/(D*(A1 + B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))\
                *sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) - 2*A2*(A1**3*B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)\
                - A1**2*(A3*B1*Jn*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2) - 2*B2*B3*Jp\
                 *sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2) + B3*Jp*(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) \
                + A1*(-2*A2*B1*B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2) + A3*B1*B2*Jn\
                *sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2) + A3*B1*Jn*(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2 \
                - 2*B2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) + B2**2*B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)\
                - B2*B3*Jp*(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) + A2*B1*(2*A3*B1*Jn\
                *sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2) + B3*Jp*(A1**2 + A1*(2*B2\
                - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) - 4*A2*B1 + B2**2 - B2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))))\
                *Piecewise((2*D*exp(t*(-A1 + B2 + sqrt(A1**2 + 2*A1*B2 \
                - 4*A2*B1 + B2**2))/(2*D))/(-A1 + B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)),\
                Ne(-A1 + B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2), 0)), \
                (t, True))*exp(-t*(-A1 + B2 + sqrt(A1**2 + 2*A1*B2 \
                - 4*A2*B1 + B2**2))/(2*D))/(D*(A1 + B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))\
                *sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)*(A1**2 + A1*(2*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))\
                - 4*A2*B1 + B2**2 - B2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)))                                                                                                                                                                                             
    
    # Liquid concentration flux at positive electrode interface
    def qlip(self):
        A1=self.A_1()
        A2=self.A_2()
        A3=self.A_3()
        B1=self.B_1()
        B2=self.B_2()
        B3=self.B_3()
        D=self.D()
        Jn=self.j_n()
        Jp=self.j_p()
        return B1*(-2*A1*A2*B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2) + A2*(2*A3*B1*Jn*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)\
            - B3*Jp*(A1**2 + A1*(2*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) - 4*A2*B1 + B2**2 - B2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)))\
            + A3*B2*Jn*(A1**2 + A1*(2*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) - 4*A2*B1 + B2**2 - B2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)))\
            *Piecewise((-2*D*exp(-t*(A1 - B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))/(2*D))/(A1 - B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)),\
            Ne(A1 - B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2), 0)), (t, True))*exp(t*(A1 - B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 \
            + B2**2))/(2*D))/(D*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)*(A1**2 + A1*(2*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))\
            - 4*A2*B1 + B2**2 - B2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))) - B1*(A1**2*(A2*B3*Jp - A3*B2*Jn)\
            + A1*(A2*B3*Jp*(2*B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) - A3*B2*Jn*(2*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)))\
            - 4*A2**2*B1*B3*Jp + A2*(2*A3*B1*Jn*(2*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) + B2**2*B3*Jp - B2*B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))\
            - A3*B2**2*Jn*(B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)))*Piecewise((-2*D/(A1 - B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)),\
            Ne(A1, A2*B1/B2)), (-2*B2*D/(A2*B1 - B2**2 + B2*sqrt(A2**2*B1**2/B2**2 - 2*A2*B1 + B2**2)), True))*exp(t*(A1 - B2 + sqrt(A1**2 + 2*A1*B2\
            - 4*A2*B1 + B2**2))/(2*D))/(D*(A1**3 + A1**2*(3*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) - A1*(4*A2*B1 - 3*B2**2 + 2*B2*sqrt(A1**2 + 2*A1*B2\
            - 4*A2*B1 + B2**2)) - 4*A2*B1*(B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) + B2**3 - B2**2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)))\
            + (A1**2*B3*Jp - A1*(A3*B1*Jn - B2*B3*Jp + B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) - 2*A2*B1*B3*Jp + A3*B1*Jn*(B2 + sqrt(A1**2\
            + 2*A1*B2 - 4*A2*B1 + B2**2)))*Piecewise((2*D/(-A1 + B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)), Ne(A1, A2*B1/B2)), (2*B2*D/(-A2*B1 + B2**2\
            + B2*sqrt(A2**2*B1**2/B2**2 - 2*A2*B1 + B2**2)), True))*exp(-t*(-A1 + B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))/(2*D))/(2*D*sqrt(A1**2 + 2*A1*B2\
            - 4*A2*B1 + B2**2)) + (A1**3*B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2) - A1**2*(A3*B1*Jn*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)\
            - 2*B2*B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2) + B3*Jp*(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) + A1*(-2*A2*B1*B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)\
            + A3*B1*B2*Jn*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2) + A3*B1*Jn*(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2 - 2*B2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))\
            + B2**2*B3*Jp*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2) - B2*B3*Jp*(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) + A2*B1*(2*A3*B1*Jn*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)\
            + B3*Jp*(A1**2 + A1*(2*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) - 4*A2*B1 + B2**2 - B2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))))\
            *Piecewise((2*D*exp(t*(-A1 + B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))/(2*D))/(-A1 + B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)),\
            Ne(-A1 + B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2), 0)), (t, True))*exp(-t*(-A1 + B2 + sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2))/(2*D))/(D*sqrt(A1**2 + 2*A1*B2\
            - 4*A2*B1 + B2**2)*(A1**2 + A1*(2*B2 - sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)) - 4*A2*B1 + B2**2 - B2*sqrt(A1**2 + 2*A1*B2 - 4*A2*B1 + B2**2)))
    
    # Liquid concentration at negative electrode interface
    def clin(self):
        return self.cl_init * self.alpha_in() * self.qlin() + self.alpha_ip()* self.qlip()
    
    # Liquid concentration at negative electrode interface
    def clip(self):
        return self.clip() + self.ls * (self.qlin() + self.qlip())/(2*self.diff_ls)
    
    # Liquid concentration at negative current-collector end
    def cl0(self):
        return self.clin() + self.ln * self.qlin()/(2*self.diff_ln)
    
    #Liquid concentration at mid of cell
    def clmid(self):
        return self.clin() - 3*self.ls*self.qlin() / (8*self.diff_ls) - self.ls*self.qlip() / (8*self.diff_ls)
    
    # Liquid concentration at negative current-collector end
    def clL(self):
        return self.clip() - self.lp * self.qlip()/ (2 * self.diff_lp)
    
    # Liquid concentration at negative electrode
    def Cln(self):
        return self.clin() + self.qlin() * (self.ln**2-self.x**2)/(2*self.ln * self.diff_ln)
    
    # Liquid concentration at positive electrode
    def Clp(self):
        return self.clip() + self.qlip() * (self.lp**2 - (self.L - self.x)**2) / (2* self.lp * self.diff_lp)
    
    #Liquid concentration in seperator region
    def Cls(self):
        return self.clin() + self.qlin()* (self.x - self.ln) / (self.diff_ls) + (self.qlin() - self.qlip()) * ((self.x - self.L)**2) / ((2* self.ls * self.diff_ls)
                                                                                                                                        

'''Scheme for Electrolyte Potential'''
class ElectrolytePotential:
                                                                                                                                        
    def __init__(self, x: float, ln: float, lp: float, ls: float,klp: float, kln: float, kls:
                       tr_num: float, R: float, T: float, F: float, I: float) -> None:
        
        # Battary Parameters 
        self.x = x # Length variable
        self.klp = klp  # Ionic conductivity of positive electrode
        self.kln = kln  # Ionic conductivity of nagative electrode
        self.kls = kls # Ionic conductivity of seperatior
        self.ln = ln  # length of negative electrode in m
        self.lp = lp  # length of positive electrode in m
        self.ls = ls # Length of seperator
        self.tr_num = tr_num # Transferrance number
        self.R = R # Universal gas constant
        self.F = F # Faraday number
        self.T = T # Temperature
        self.I = I # Applied Current     
    
    # Electrolyte Potential of seperator region
    def Phi_Ls(self):
        return ((2* self.R * self.T) /(self.F)) * (1-self.tr_num) * math.log(self.Cls()/self.clmid())  + self.I * (self.x - self.ln - 0.5*self.ls)  / (self.kls)
        
    # Electrolyte Potential at interface of negative electrode
    def phi_ln(self):
        return  ((2* self.R * self.T) /(self.F)) * (1-self.tr_num) * math.log(self.clin()/self.clmid())  + (self.I * self.ls) /(2*self.kls)    
        
    # Electrolyte Potential at interface of positive electrode
    def phi_lp(self):
        return  ((2* self.R * self.T) /(self.F)) * (1-self.tr_num) * math.log(self.clip()/self.clmid())  - (self.I * self.ls) /(2*self.kls)    
        
    # Electrolyte concentraton at negative elctrode
    def Phi_Ln(self):
        return self.phi_ln() + ((2* self.R * self.T) /(self.F)) * (1-self.tr_num) * math.log(self.clin()/self.clmid()) + (self.I * (self.ln - self.x)) /(self.kln)\
            (self.I * (self.ln - self.x)**2)/(2*self.ln*self.kls)
            
    # Electrolyte concentraton at positive elctrode
    def Phi_Ln(self):
        return self.phi_lp() + ((2* self.R * self.T) /(self.F)) * (1-self.tr_num) * math.log(self.clip()/self.clmid()) + (self.I * (self.x-self.ln -self.ls)) /(self.klp)\
            (self.I * (self.x-self.ln-self.ls)**2)/(2*self.lp*self.klp)
            
    # Electrolyte Potential at current collector end of negative electrode
    def phi_n0(self):
        retun  self.phi_ln() + ((2* self.R * self.T) /(self.F)) * (1-self.tr_num) * math.log(self.cl0()/self.clmid()) + (self.I * self.ln) /(2*self.kln) 
        
    # Electrolyte Potential at current collector end of positive electrode
    def phi_pL(self):
        retun  self.phi_lp() + ((2* self.R * self.T) /(self.F)) * (1-self.tr_num) * math.log(self.clL()/self.clmid()) - (self.I * self.lp) /(2*self.klp)    
                                                                                                          
                                                                                                          
'''Scheme for Solid Phase Potential''' 
class SolidPhasePotential:
                                                                                                            
    def __init__(self, R: float, T: float, F: float) -> None:
        
        # Battary Parameters 
        self.x = x # Length variable
        self.klp = klp  # Ionic conductivity of positive electrode
        self.kln = kln  # Ionic conductivity of nagative electrode
        self.kls = kls # Ionic conductivity of seperatior
        self.ln = ln  # length of negative electrode in m
        self.lp = lp  # length of positive electrode in m
        self.ls = ls # Length of seperator
        self.tr_num = tr_num # Transferrance number
        self.R = R # Universal gas constant
        self.F = F # Faraday number
        self.T = T # Temperature
        self.I = I # Applied Current     
    
    # Uniform Current density of negative electrode
    def J_n(self):
        pass
    
    # Uniform Current density of positive electrode
    def J_p(self):
        pass
    
    # Excahnge Current density of positive electrode
    def Jp0(self):
        pass
        
    # Excahnge Current density of negative electrode
    def Jn0(self):
        pass
        
    # OCP of positive electrode
    def Up(self):
        pass
        
    #OCP of negative electrode
    def Un(self):
        pass
    
    # Solid Phase Potential of Negative electrode
    def Phi_Sn(self):
        return self.Un() + self.Phi_Ln() + ((2* self.R * self.T) /(self.F))* math.asinh (self.J_n/(2*self.Jn0))
        
    
    # Solid Phase Potential of Negative electrode
    def Phi_Sp(self):
        return self.Up() + self.Phi_Lp() + ((2* self.R * self.T) /(self.F))* math.asinh (self.J_p/(2*self.Jp0))
    
    
    