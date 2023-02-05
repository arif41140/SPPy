import matplotlib.pyplot as plt
import matplotlib as mpl

from src.solution import Solution
from src.calc_helpers.constants import Constants


class Plots:
    def __init__(self, *args):
        for sol in args:
            if (not isinstance(sol, Solution)) and (not isinstance(sol, list)):
                if isinstance(sol, list):
                    for list_item in list:
                        if not isinstance(list_item, Solution):
                            raise TypeError("List needs to hold Solution object.")
                raise TypeError("input needs to be a solution object or a solution containing list objects")
        if isinstance(args, list):
            self.sols = args[0]
        else:
            self.sols = args

    def plot_in_axis(self, ax, sol, x_var, y_var):
        if sol.name is not None:
            ax.plot(x_var, y_var, label=sol.name)
        elif sol.name is None:
            ax.plot(x_var, y_var)

    def comprehensive_plot(self):
        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color='bgrcmyk')
        mpl.rcParams['lines.linewidth'] = 3
        plt.rc('axes', titlesize= 15)
        plt.rc('axes', labelsize= 15)

        fig = plt.figure(figsize=(12, 10), dpi=300)
        num_col = 2
        num_row = 3

        ax1 = fig.add_subplot(num_row, num_col, 1)
        ax1.set_xlabel('Time [s]')
        ax1.set_ylabel('V [V]')
        ax1.set_title('V vs. Time')

        ax2 = fig.add_subplot(num_row, num_col, 2)
        ax2.set_xlabel('Capacity [Ahr]')
        ax2.set_ylabel('V [V]')
        ax2.set_title('V vs. Capacity')

        ax3 = fig.add_subplot(num_row, num_col, 3)
        ax3.set_xlabel('Time [s]')
        ax3.set_ylabel('SOC')
        ax3.set_title('Positive Electrode SOC')

        ax4 = fig.add_subplot(num_row, num_col, 4)
        ax4.set_xlabel('Time [s]')
        ax4.set_ylabel('SOC')
        ax4.set_title('Negative Electrode SOC')

        ax5 = fig.add_subplot(num_row, num_col, 5)
        ax5.set_xlabel('Time [s]')
        ax5.set_ylabel(r'T [$^O$C]')
        ax5.set_title('Battery Surface Temp')

        for sol in self.sols:
            self.plot_in_axis(ax1, sol, sol.t, sol.V)
            self.plot_in_axis(ax2, sol, sol.cap, sol.V)
            self.plot_in_axis(ax3, sol, sol.t, sol.x_surf_p)
            self.plot_in_axis(ax4, sol, sol.t, sol.x_surf_n)
            self.plot_in_axis(ax5, sol, sol.t, sol.T - Constants.T_abs)

        lines_labels = [ax1.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        fig.legend(lines, labels)
        plt.tight_layout()
        plt.show()
