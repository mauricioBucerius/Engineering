import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# from calculations import extract_data
import os


f_min = 1
f_max = 4

# Messdaten als Excel Tabelle
# FILENAME_MEASUREMENTS_POWER_CLASS_AB = 'class_ab_single_large_freq_in_pwr_sweep.txt'
# FILENAME_MEASUREMENTS_FREQ_CLASS_AB = 'class_ab_single_large_freq_freq_sweep.txt'

FILENAME_MEASUREMENTS_POWER_CLASS_AB = 'class_ab_single_u_gs_0_5_v_in_pwr_sweep.txt'
FILENAME_MEASUREMENTS_FREQ_CLASS_AB = 'class_ab_single_u_gs_0_5_v_freq_sweep.txt'

# FILENAME_MEASUREMENTS_POWER_CLASS_AB = 'class_ab_4_7_pF_in_pwr_sweep.txt'
# FILENAME_MEASUREMENTS_FREQ_CLASS_AB = 'class_ab_4_7_pF_freq_sweep.txt'

FILENAME_MEASUREMENTS_POWER_CLASS_AB_DEFAULT = 'class_ab_red_4_7_pF_in_pwr_sweep.txt'
FILENAME_MEASUREMENTS_FREQ_CLASS_AB_DEFAULT = 'class_ab_red_4_7_pF_freq_sweep.txt'

# Simulationsergebnisse Eingangsleistungs Sweep
FILENAME_PAE_CLASS_AB = 'PAE_vs_input_Class_AB.txt'
FILENAME_POWER_CLASS_AB = 'OUTPUT_vs_input_Class_AB.txt'
FILENAME_GAIN_CLASS_AB = 'GAIN_vs_input_Class_AB.txt'
FILENAME_DC_POWER_CLASS_AB = 'DC_vs_input_Class_AB.txt'

# Simulationsergebnisse Frequenz Sweep
FILENAME_SIMULATION_FREQ = 'Large_Signal_freq_sweep.csv'

# Einstellungen für
DPI = 300
# LABEL_SIZE = 25
FIGURE_WIDTH = 6.4
FIGURE_HEIGTH = 4.8

# print(mpl.rcParams.keys())
# mpl.rcParams['axes.linewidth'] = 2
# mpl.rcParams['lines.linewidth'] = 2.5
# mpl.rcParams['xtick.labelsize'] = LABEL_SIZE
# mpl.rcParams['ytick.labelsize'] = LABEL_SIZE
# mpl.rcParams['font.size'] = LABEL_SIZE
# mpl.rcParams['grid.linewidth'] = 2
# mpl.rcParams['savefig.dpi'] = DPI
# mpl.rcParams['legend.fontsize'] = 'medium'
plt.style.use('D:\\MA_Bucerius\\07_Python\\ChipDesign\\two_plots_aligned_large_signal.mplstyle')

data_measurements = pd.read_csv(FILENAME_MEASUREMENTS_POWER_CLASS_AB)
header = data_measurements.columns

pwr_meas = data_measurements[header[0]]
output_pwr_meas = data_measurements[header[1]]
gain_pwr_meas = data_measurements[header[2]]
pae_pwr_meas = data_measurements[header[3]] * 100
dc_power_pwr_meas = data_measurements[header[4]] * 1e3


data_pae_sim = pd.read_csv(FILENAME_PAE_CLASS_AB, delimiter='\t')
header_sim = data_pae_sim.columns
pae_pwr_sim = data_pae_sim[header_sim[1]]
pwr_sim = data_pae_sim[header_sim[0]]

data_output_sim = pd.read_csv(FILENAME_POWER_CLASS_AB, delimiter='\t')
output_pwr_sim = data_output_sim[data_output_sim.columns[1]]

data_gain_sim = pd.read_csv(FILENAME_GAIN_CLASS_AB, delimiter='\t')
gain_pwr_sim = data_gain_sim[data_gain_sim.columns[1]]

data_dc_sim = pd.read_csv(FILENAME_DC_POWER_CLASS_AB, delimiter='\t')
dc_power_pwr_sim = data_dc_sim[data_dc_sim.columns[1]]

########################################################################################################################
# Auswertung für Eingangsleistungs Sweep
pwr = 13

# Power Sweep
idx_max_pae_sim = np.argmax(pae_pwr_sim)
# pae_design = np.interp(pwr, pwr_sim, pae_pwr_sim)
# output_design = np.interp(pwr, pwr_sim, output_pwr_sim)
# gain_design = np.interp(pwr, pwr_sim, gain_pwr_sim)
# dc_design = np.interp(pwr, pwr_sim, dc_power_pwr_sim)

idx_max_pae_pwr_meas = np.argmax(pae_pwr_meas)
# pae_design = np.interp(pwr, pwr_sim, pae_pwr_sim)
# output_design = np.interp(pwr, pwr_sim, output_pwr_sim)
# gain_design = np.interp(pwr, pwr_sim, gain_pwr_sim)
# dc_design = np.interp(pwr, pwr_sim, dc_power_pwr_sim)

print('----------------------------------------')
print('Power Sweep')
print('----------------------------------------\n')
print('Simulation'.upper())
print(f'Max PAE: {round(pae_pwr_sim[idx_max_pae_sim], 1)} % @ {round(pwr_sim[idx_max_pae_sim], 1)} dBm')
print(f'Output Power: {round(output_pwr_sim[idx_max_pae_sim], 1)} dBm @ {round(pwr_sim[idx_max_pae_sim], 1)} dBm')
print(f'Gain: {round(gain_pwr_sim[idx_max_pae_sim], 1)} dBm @ {round(pwr_sim[idx_max_pae_sim], 1)} dBm')
print(f'Kompression: {round(gain_pwr_sim[0] - gain_pwr_sim[idx_max_pae_sim], 1)} dB @ {round(pwr_sim[idx_max_pae_sim], 1)} dBm')

print('\nMessung'.upper())
print(f'Max PAE: {round(pae_pwr_meas[idx_max_pae_pwr_meas], 1)} % @ {round(pwr_meas[idx_max_pae_pwr_meas], 1)} dBm')
print(f'Output Power: {round(output_pwr_meas[idx_max_pae_pwr_meas], 1)} dBm @ {round(pwr_meas[idx_max_pae_pwr_meas],1 )} dBm')
print(f'Gain: {round(gain_pwr_meas[idx_max_pae_pwr_meas], 1)} dBm @ {round(pwr_meas[idx_max_pae_pwr_meas], 1)} dBm')
print(f'Kompression: {round(gain_pwr_meas[0] - gain_pwr_meas[idx_max_pae_pwr_meas], 1)} dB @ {round(pwr_meas[idx_max_pae_pwr_meas], 1)} dBm')
print(f'Unterschied im Gain: {round(abs(gain_pwr_meas[0]-gain_pwr_sim[0]), 1)} dB')
print(f'Unterschied in der Ausgangsleistung: {round(abs(output_pwr_meas[0]-output_pwr_sim[0]), 1)} dB')

########################################################################################################################
# Auswertung für Eingangsleistungs Sweep

# plt.figure()
# plt.plot(pwr_meas, output_pwr_meas, color='tab:blue', label='Messung')
# plt.plot(pwr_sim, output_pwr_sim, color='tab:red', label='Simulation')
#
# plt.xlabel('Eingangsleistung / dBm')
# plt.ylabel('Ausgangsleistung / dBm')
# plt.xlim([-20, 15])
# plt.ylim([-5, 30])
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'{FILENAME_POWER_CLASS_AB.split(".")[0]}.png')
#
# plt.figure()
# plt.plot(pwr_meas, gain_pwr_meas, color='tab:blue', label='Messung')
# plt.plot(pwr_sim, gain_pwr_sim, color='tab:red', label='Simulation')
#
# plt.xlabel('Eingangsleistung / dBm')
# plt.ylabel('Verstärkung / dB')
# plt.xlim([-20, 15])
# plt.ylim([8, 20])
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'{FILENAME_GAIN_CLASS_AB.split(".")[0]}.png')
#
# plt.figure()
# plt.plot(pwr_meas, dc_power_pwr_meas, color='tab:blue', label='Messung')
# plt.plot(pwr_sim, dc_power_pwr_sim, color='tab:red', label='Simulation')
#
# plt.xlabel('Eingangsleistung / dBm')
# plt.ylabel('Leistungsaufnahme / mW')
# plt.xlim([-20, 17])
# plt.ylim([0, 1000])
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'{FILENAME_DC_POWER_CLASS_AB.split(".")[0]}.png')
#
# plt.figure()
# plt.plot(pwr_meas, pae_pwr_meas, color='tab:blue', label='Messung')
# plt.plot(pwr_sim, pae_pwr_sim, color='tab:red', label='Simulation')
#
# plt.xlabel('Eingangsleistung / dBm')
# plt.ylabel('PAE / %')
# plt.xlim([-20, 15])
# plt.ylim([0, 70])
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'{FILENAME_PAE_CLASS_AB.split(".")[0]}.png')

# plt.show()
########################################################################################################################
# Auswertung für Frequenz Sweep
data_measurements = pd.read_csv(FILENAME_MEASUREMENTS_FREQ_CLASS_AB)
header = data_measurements.columns

freq_meas = data_measurements[header[0]]/1e9

# dc_power_pwr_meas = data_measurements[header[5]]
output_freq_meas = data_measurements[header[1]]
gain_freq_meas = data_measurements[header[2]]
pae_freq_meas = data_measurements[header[3]] * 100
dc_power_freq_meas = data_measurements[header[4]] * 1e3

# data_measurements = pd.read_csv(FILENAME_MEASUREMENTS_FREQ_CLASS_AB_INPUT)
# header = data_measurements.columns
#
# freq_meas_input = data_measurements[header[0]]/1e9
#
# # dc_power_pwr_meas = data_measurements[header[5]]
# output_meas_input = data_measurements[header[1]]
# gain_pwr_meas_input = data_measurements[header[2]]
# pae_pwr_meas_input = data_measurements[header[3]] * 100

# data_measurements = pd.read_csv()
# header = data_measurements.columns
#
# freq_meas = data_measurements[header[0]]/1e9
#
# dc_power_pwr_meas = data_measurements[header[4]]
# output_meas = data_measurements[header[1]]
# gain_pwr_meas = data_measurements[header[2]]
# pae_pwr_meas = data_measurements[header[3]] * 100

data_simulation = pd.read_csv(FILENAME_SIMULATION_FREQ)
header_sim = data_simulation.columns

freq_sim = data_simulation[header_sim[0]]
pae_freq_sim = data_simulation[header_sim[1]]
output_freq_sim = data_simulation[header_sim[2]]
dc_power_freq_sim = data_simulation[header_sim[3]]*1e3
gain_freq_sim = data_simulation[header_sim[4]]

idx_max_pae_sim_freq = np.argmax(pae_freq_sim)
idx_max_pae_meas_freq = np.argmax(pae_freq_meas)

print('\nPower Sweep:')
print(f'Max Gain (sim): {round(gain_pwr_sim[0], 1)} dB @ {pwr_sim[0]} dBm')
print(f'Max Gain (meas): {round(gain_pwr_meas[0], 1)} dB @ {pwr_sim[0]} dBm')

print('\nFrequenz Sweep:')
print(f'Max PAE (sim): {round(pae_freq_sim[idx_max_pae_sim_freq], 1)} % \t @ {round(freq_sim[idx_max_pae_sim_freq], 2)} GHz')
print(f'Max PAE (mess): {round(pae_freq_meas[idx_max_pae_meas_freq], 1)} % \t @ {round(freq_meas[idx_max_pae_meas_freq], 2)} GHz')
# plt.figure()
# plt.plot(freq_meas, output_freq_meas, color='tab:blue', label='Messung')
# plt.plot(freq_sim, output_freq_sim, color='tab:red', label='Simulation')
#
# plt.xlabel('Frequenz / GHz')
# plt.ylabel('Ausgangsleistung / dBm')
# plt.ylim([-5, 30])
# plt.xlim([f_min, f_max])
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'OUTPUT_vs_freq_Class_AB.png')
#
# plt.figure()
# plt.plot(freq_meas, gain_freq_meas, color='tab:blue', label='Messung')
# plt.plot(freq_sim, gain_freq_sim, color='tab:red', label='Simulation')
#
# plt.xlabel('Frequenz / GHz')
# plt.ylabel('Verstärkung / dB')
# plt.ylim([-20, 20])
# plt.xlim([f_min, f_max])
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'GAIN_vs_freq_Class_AB.png')
#
# plt.figure()
# plt.plot(freq_meas, dc_power_freq_meas, color='tab:blue', label='Messung')
# plt.plot(freq_sim, dc_power_freq_sim, color='tab:red', label='Simulation')
#
# plt.xlabel('Frequenz / GHz')
# plt.ylabel('Leistungsaufnahme / mW')
# plt.xlim([1.7, 2.9])
# plt.ylim([0, 1000])
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'DC_vs_freq_Class_AB.png')
#
# plt.figure()
# plt.plot(freq_meas, pae_freq_meas, color='tab:blue', label='Messung')
# plt.plot(freq_sim, pae_freq_sim, color='tab:red', label='Simulation')
#
# plt.xlabel('Frequenz / GHz')
# plt.ylabel('PAE / %')
# plt.xlim([f_min, f_max])
# plt.ylim([0, 70])
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'PAE_vs_freq_Class_AB.png')

# =============================================================================
# %% Measurement Results - single plot
# =============================================================================

# Einstellungen für
DPI = 500
LABEL_SIZE = 20
FIGURE_WIDTH = 9
FIGURE_HEIGTH = 7.2

# print(mpl.rcParams.keys())
# mpl.rcParams['axes.linewidth'] = 2
# mpl.rcParams['lines.linewidth'] = 2.5
# mpl.rcParams['xtick.labelsize'] = LABEL_SIZE
# mpl.rcParams['ytick.labelsize'] = LABEL_SIZE
# mpl.rcParams['font.size'] = LABEL_SIZE
# mpl.rcParams['grid.linewidth'] = 2
# mpl.rcParams['savefig.dpi'] = DPI

plt.style.use('D:\\Benutzer\\OneDrive\\OneDrive - bwedu\\KIT\\Master\\5. Master\\07_Python\\ChipDesign\\two_plots_aligned_large_signal.mplstyle')

y_min = 0
y_max = 60
y_step = 2.5
x_min = 1
x_max = 4
x_step = 0.1
fig, ax_pae = plt.subplots(dpi=DPI, figsize=[FIGURE_WIDTH, FIGURE_HEIGTH])
# fig.subplots_adjust(right=0.8)

# creates one more y-axis on the right side
ax_gain_output = ax_pae.twinx()
ax_pae.spines.left.set_position(("axes", 0.0))
ax_gain_output.spines.right.set_position(("axes", 1.0))

# limits of the plots and setting the labels of the axis
ax_pae.set(xlim=(1, 4), ylim=(y_min, y_max), xlabel='Frequenz / GHz', ylabel='PAE / %',
           xticks=np.arange(x_min, x_max+0.5, 0.5))
ax_gain_output.set(ylim=(-20, 40), ylabel='G / dB, $P_{out}$ / dBm')

# plotting the data
p1_meas, = ax_gain_output.plot(freq_meas, output_freq_meas, color='tab:blue', linestyle='--', label='$P_\mathrm{out, meas}$')
p1_sim, = ax_gain_output.plot(freq_sim, output_freq_sim, color='tab:blue', label='$P_\mathrm{out, sim}$')
p2_meas, = ax_pae.plot(freq_meas, pae_freq_meas, color='tab:orange', linestyle='--', label='$PAE_\mathrm{meas}$')
p2_sim, = ax_pae.plot(freq_sim, pae_freq_sim, color='tab:orange', label='$PAE_\mathrm{sim}$')
p3_meas, = ax_gain_output.plot(freq_meas, gain_freq_meas, color='tab:red', linestyle='--', label='$G_\mathrm{meas}$')
p3_sim, = ax_gain_output.plot(freq_sim, gain_freq_sim, color='tab:red', label='$G_\mathrm{sim}$')

# Minor Ticks
minor_x_ticks = np.arange(x_min,  x_max + x_step, x_step)
minor_y_ticks = np.arange(y_min, y_max, y_step)
ax_pae.set_xticks(minor_x_ticks, minor=True)
ax_pae.set_yticks(minor_y_ticks, minor=True)
ax_pae.grid(axis='x', which='minor', alpha=0.2)
ax_pae.grid(axis='y', which='minor', alpha=0.2)
ax_pae.grid()

plt.legend(handles=[p1_meas, p1_sim, p2_meas, p2_sim, p3_meas, p3_sim],
           bbox_to_anchor=(0., 1.02, 1., .102),
           loc='lower left',
           mode="expand",
           borderaxespad=0.,
           ncols=3)

# ax_pae.grid()
plt.tight_layout()
plt.savefig('single_class_ab_vs_freq.png')

y_min = 0
y_max = 60
y_step = 2.5
x_min = -20
x_max = 17
x_step = 2
# Plotting the Input Power Sweep Results
fig, ax_pae = plt.subplots(dpi=DPI, figsize=[FIGURE_WIDTH, FIGURE_HEIGTH])
fig.subplots_adjust(right=0.8)

# creates two more y-axis
ax_gain_output = ax_pae.twinx()
ax_gain_output.spines.right.set_position(("axes", 1.0))

# limits of the plots and setting the labels of the axis
ax_pae.set(xlim=(-20, 17), ylim=(y_min, y_max), xlabel='Eingangsleistung / dBm', ylabel='PAE / %')
ax_gain_output.set(ylim=(-20, 40), ylabel='G / dB, $P_{out}$ / dBm')

p1_meas, = ax_gain_output.plot(pwr_meas, output_pwr_meas, color='tab:blue', linestyle='--', label='$P_\mathrm{out, meas}$')
p1_sim, = ax_gain_output.plot(pwr_sim, output_pwr_sim, color='tab:blue', label='$P_\mathrm{out, sim}$')
p2_meas, = ax_pae.plot(pwr_meas, pae_pwr_meas, color='tab:orange', linestyle='--', label='$PAE_\mathrm{meas}$')
p2_sim, = ax_pae.plot(pwr_sim, pae_pwr_sim, color='tab:orange', label='$PAE_\mathrm{sim}$')
p3_meas, = ax_gain_output.plot(pwr_meas, gain_pwr_meas, color='tab:red', linestyle='--', label='$G_\mathrm{meas}$')
p3_sim, = ax_gain_output.plot(pwr_sim, gain_pwr_sim, color='tab:red', label='$G_\mathrm{sim}$')

# Minor Ticks
minor_x_ticks = np.arange(x_min,  x_max + x_step, x_step)
# minor_x_ticks = np.arange(-20, 18 + x_step, x_step)
minor_y_ticks = np.arange(y_min, y_max, y_step)
ax_pae.set_xticks(minor_x_ticks, minor=True)
ax_pae.set_yticks(minor_y_ticks, minor=True)
ax_pae.grid(axis='x', which='minor', alpha=0.2)
ax_pae.grid(axis='y', which='minor', alpha=0.2)
ax_pae.grid()

# ax_pae.legend(handles=[p1_meas, p1_sim, p2_meas, p2_sim, p3_meas, p3_sim],
#                  bbox_to_anchor=(1.02, 1.3),
#                  ncols=3)

plt.legend(ncols=3,
           handles=[p1_meas, p1_sim, p2_meas, p2_sim, p3_meas, p3_sim],
           bbox_to_anchor=(0., 1.02, 1., .102),
           loc='lower left',
           mode="expand",
           borderaxespad=0.)
# ax_pae.grid()
plt.tight_layout()
plt.savefig('single_class_ab_vs_input.png')

# Beispiel Plot für Power Point Präsentation
plt.style.use('D:\\Benutzer\\OneDrive\\OneDrive - bwedu\\KIT\\Master\\5. Master\\07_Python\\ChipDesign\\power_point.mplstyle')

y_min = 0
y_max = 60
y_step = 2.5
x_min = -20
x_max = 17
x_step = 5

fig, ax_pae = plt.subplots(dpi=DPI, figsize=[FIGURE_WIDTH, FIGURE_HEIGTH])

# limits of the plots and setting the labels of the axis
ax_pae.set(xlim=(1, 4), ylim=(y_min, y_max), xlabel='Eingangsleistung / dBm', ylabel='PAE / %',
           xticks=np.arange(x_min, x_max+5, 5))

# plotting the data
ax_pae.plot(pwr_sim, pae_pwr_sim, color='tab:blue', label='$PAE_\mathrm{sim}$')
# ax_pae.hist(pwr_sim[], pae_pwr_sim[], color='tab:red', marker='x')

# Minor Ticks
# minor_x_ticks = np.arange(x_min,  x_max + x_step, x_step)
# minor_y_ticks = np.arange(y_min, y_max, y_step)
# ax_pae.set_xticks(minor_x_ticks, minor=True)
# ax_pae.set_yticks(minor_y_ticks, minor=True)
# ax_pae.grid(axis='x', which='minor', alpha=0.2)
# ax_pae.grid(axis='y', which='minor', alpha=0.2)
ax_pae.grid()
plt.tight_layout()

plt.savefig('single_class_ab_power_point_plot.png')

print('Done')