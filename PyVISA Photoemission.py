# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 10:16:40 2023

@author: kyebchoo
"""

# has to start wtih "pip install pyserial" & "pip install pygame" in the kernal
''' loading packages 

    Has to start with entering the following lines in the kernal:         
         1) "pip install pyserial"
         2) "pip install pygame"
         3) 'pip install keyboard'
    
    

'''

import numpy as np
import scipy as sci
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import math
import pandas as pd
import pyvisa
import time
from matplotlib import cbook
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import serial
import re
from pygame import mixer
from time import sleep
from tqdm import tqdm
# from tqdm.auto import tqdm
import keyboard
from datetime import date, datetime
from matplotlib.pyplot import figure

#%%

''' initiating program '''

rm = pyvisa.ResourceManager()
# print(rm.list_resources())

oscilloscope = rm.open_resource('TCPIP0::192.168.0.2::inst0::INSTR')
# oscilloscope.query('*IDN?')

picoammeter = rm.open_resource('ASRL9::INSTR', baud_rate = 9600, data_bits = 8, write_termination = '\r', read_termination = '\r')
# picoammeter.query('*IDN?\r\n')

#%%

''' loading effects '''

mixer.init()
beep_short = mixer.Sound("beep.wav")
beep_error = mixer.Sound("beep_end.wav")



#%%

''' declaring global constants '''



c0 = 299792458
e = -1.602 * (10**-19)



#%% 

''' loading light profiles '''



light_profile = pd.read_excel("2023_10_13_light_profile.xlsx", engine = "openpyxl")
# light_profile = pd.read_excel("2023_10_25_light_profile.xlsx", engine = "openpyxl")
filter_list = np.array(light_profile)[:,0]
centroid_lambda = np.array(light_profile)[:,1]
peak_lambda = np.array(light_profile)[:,2]
FWHM_lambda = np.array(light_profile)[:,3]



#%%

''' defining general functions '''

def timestamp():
    string = str(datetime.now()).replace('-', '_').replace(' ', '_')\
        .replace(':', '_')
    return string




def line(x, m, c):
    return m*x + c

def quad(x, a, b, c):
    return a*(x**2) + b*x + c



def fit_line(x, y, ystddev):
    popt, pcov = curve_fit(line, x, y, sigma = ystddev)
    m = popt[0]
    c = popt[1]
    mmin = popt[0] - np.sqrt(pcov[0, 0])
    mmax = popt[0] + np.sqrt(pcov[0, 0])
    cmin = popt[1] - np.sqrt(pcov[1, 1])
    cmax = popt[1] + np.sqrt(pcov[1, 1])
    
    return m, c, mmin, mmax, cmin, cmax


def fit_line_unweighted(x, y):
    popt, pcov = curve_fit(line, x, y)
    m = popt[0]
    c = popt[1]
    mmin = popt[0] - np.sqrt(pcov[0, 0])
    mmax = popt[0] + np.sqrt(pcov[0, 0])
    cmin = popt[1] - np.sqrt(pcov[1, 1])
    cmax = popt[1] + np.sqrt(pcov[1, 1])
    
    return m, c, mmin, mmax, cmin, cmax


def fit_quad(x, y, ystddev):
    popt, pcov = curve_fit(quad, x, y, ystddev)
    a = popt[0]
    b = popt[1]
    c = popt[2]
    amax = popt[0] + np.sqrt(pcov[0, 0])
    amin = popt[0] - np.sqrt(pcov[0, 0])
    bmax = popt[1] + np.sqrt(pcov[1, 1])
    bmin = popt[1] - np.sqrt(pcov[1, 1])
    cmax = popt[2] + np.sqrt(pcov[2, 2])
    cmin = popt[2] - np.sqrt(pcov[2, 2])
    return a, b, c, amin, amax, bmin, bmax, cmin, cmax



def plot(filter_choice):
    "plotting"
    x = eval("df" + str(filter_choice) + "[:,0]")
    y = eval("df" + str(filter_choice) + "[:,2]")
    
    plt.figure()
    plt.scatter(x, y)
    plt.title("Current against Voltage (Filter " + str(filter_choice) + ")")
    plt.xlabel("Voltage/V")
    plt.ylabel("Current/A")
    plt.grid()
    plt.show()
    pass



def smallest_above_zero(filter_choice):
    "loading data and extracting smallest value above zero with its corresponding index"
    array = eval("df" + str(filter_choice) + "[:,2]")
    smallest_number = 999
    smallest_index = 0
    for i in range(0, len(array)):
        if array[i] > 0 and array[i] < smallest_number:
            smallest_number = array[i]
            smallest_index = i
    return smallest_number, smallest_index
    
    

def search_zero_intersect(filter_choice, pool = 3, diag = False, fit_mode = 'quad'):
    
    smallest_number, smallest_index = smallest_above_zero(filter_choice)
    extract_x     = eval("df" + str(filter_choice) + "[:,0]")
    # extract_x_err = eval("df" + str(filter_choice) + "[:,1]") # not used
    extract_y     = eval("df" + str(filter_choice) + "[:,2]")
    # extract_y_err = eval("df" + str(filter_choice) + "[:,3]") # not used
    # list_x = extract_x[(smallest_index - pool + 1):(smallest_index + 1)]
    # list_y = extract_y[(smallest_index - pool + 1):(smallest_index + 1)]
    list_x = extract_x[(smallest_index):(smallest_index + pool)]
    list_y = extract_y[(smallest_index):(smallest_index + pool)]
    # list_y_err = extract_y_err[(smallest_index):(smallest_index + pool)]
    # print(list_y)
    
    if fit_mode == 'line':
        # m, c, mmin, mmax, cmin, cmax = fit_line(list_x, list_y, ystddev = list_y_err)
        m, c, mmin, mmax, cmin, cmax = fit_line_unweighted(list_x, list_y)
        x_intersect = -c/m
        x_intersect_max = -cmax/mmin
        x_intersect_min = -cmin/mmax
        pass
    
    elif fit_mode == 'quad':
        # a, b, c, amin, amax, bmin, bmax, cmin, cmax = fit_quad(list_x, list_y)
        # x_intersect = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
        # x_intersect_min = (-bmin + np.sqrt(bmax**2 - 4*amin*cmin))/(2*amax)
        # x_intersect_max = (-bmax + np.sqrt(bmin**2 - 4*amax*cmax))/(2*amin)
        list_y = np.sqrt(list_y)
        # m, c, mmin, mmax, cmin, cmax = fit_line(list_x, list_y, ystddev = list_y_err)
        m, c, mmin, mmax, cmin, cmax = fit_line_unweighted(list_x, list_y)
        x_intersect = -c/m
        x_intersect_max = -cmax/mmin
        x_intersect_min = -cmin/mmax
        pass        
    
    if diag == True:
        plot(filter_choice)
        
    # print([x_intersect, x_intersect_max, x_intersect_min])
    
    return [x_intersect, x_intersect_max, x_intersect_min]



def calculate_planck(pool = 3, diag = False, fit_mode = 'quad'):
    V_ec = []
    V_ec_max = []
    V_ec_min = []

    frequency = c0/(centroid_lambda * (10**-9))
    
    for i in range(0, len(filter_list)):
        array = search_zero_intersect(int(filter_list[i]), pool, diag, fit_mode)
        V_ec.append(array[0])
        V_ec_max.append(array[1])
        V_ec_min.append(array[2])
        pass
    
    # print(V_ec)
    # print(frequency)
    
    m, c, mmin, mmax, cmin, cmax = fit_line_unweighted(frequency, V_ec)
    m2, c2, mmin2, mmax2, cmin2, cmax2 = fit_line_unweighted(frequency, V_ec_max)
    m3, c3, mmin3, mmax3, cmin3, cmax3 = fit_line_unweighted(frequency, V_ec_min)
    
    plt.figure()
    plt.scatter(frequency, V_ec)
    plt.scatter(frequency, V_ec_max)
    plt.scatter(frequency, V_ec_min)
    plt.ylabel('Cut-off Voltage, $(V_{EC})$/V')
    plt.xlabel('Frequency, $\omega$/Hz')
    plt.grid()
    plt.show()

    
    h  = m  * e
    h2 = m2 * e
    h3 = m3 * e
    print(h, h2, h3)

#%%

''' data aquisition



start/end  # starting and ending voltage, DO NOT CHANGE
steps      # number of steps vetween high and low voltage
wait       # time span between changing voltage and starting data aquisition
pause      # time for each data point
sample     # sample size for each data point (to compute error)
'''

def take_measurements(start = -2.5, end = 2.5, steps = 100, wait = 1, pause = 1, sample = 5, beep = True):
    df = []
    pause_sample = pause/sample
    step_size = (end - start)/steps
    
    # picoammeter.write('*RST')
    # picoammeter.write('ARM:SOURce IMMediate')
    # picoammeter.write('ARM:COUNt 1')
    # picoammeter.write('TRIGger:SOURce IMMediate')
    # picoammeter.write('TRIGger:COUNt 1')
    
    # picoammeter.write('SYST:ZCH ON')
    # # picoammeter.write('')
    # # picoammeter.write('')
    # # picoammeter.write('')
    # # picoammeter.write('')

    for i in tqdm(range(0, steps + 1), position = 0, leave = True):
        
        voltage = start + i * step_size
        voltage_err = 0
        
        oscilloscope.write('WGENerator:VOLTage:OFFSet ' + str(voltage))
        time.sleep(wait)
        samples = []
        
        # picoammeter.write('*RST')
        # picoammeter.write('ARM:SOURce IMMediate')
        # picoammeter.write('ARM:COUNt 1')
        # picoammeter.write('TRIGger:SOURce IMMediate')
        # picoammeter.write('TRIGger:COUNt 1')
        
        # picoammeter.write('SYST:ZCH OFF')
        
        for j in range(0, sample):
            
            picoammeter.write('*RST')
            picoammeter.write('ARM:SOURce IMMediate')
            picoammeter.write('ARM:COUNt 1')
            picoammeter.write('TRIGger:SOURce IMMediate')
            picoammeter.write('TRIGger:COUNt 1')
            
            picoammeter.write('SYST:ZCH OFF')
            
            reading = float(picoammeter.query('READ?\r\n')[0:13])
            samples.append(reading)
            time.sleep(pause_sample)
            pass
        
        current = np.average(samples)
        current_err = np.std(samples)
        df.append([voltage, voltage_err, current, current_err])
        
        if beep == True:
            beep_short.play()
        
        pass
    
    for i in range(0, 3):
        beep_error.play()
        time.sleep(1)
    
    # remove after testing
    
    
    return df

''' Example Output:
    
[[-2.5, 0, -1.3667410000000002e-10, 8.889671476494505e-12],
 [-2.0, 0, -1.4799408e-10, 1.6130881439028586e-13],
 [-1.5, 0, -1.4554786e-10, 2.2029027758846197e-13],
 [-1.0, 0, -1.3123745999999999e-10, 9.28295559584339e-12],
 [-0.5, 0, 1.6367472e-09, 1.129458254916933e-11],
 [0.0, 0, 2.7525048e-08, 6.828350881435455e-11],
 [0.5, 0, 8.2781114e-08, 5.1791043472785785e-11],
 [1.0, 0, 1.0633418000000001e-07, 9.175570608959293e-11],
 [1.5, 0, 1.1358767999999999e-07, 1.289140395767674e-10],
 [2.0, 0, 1.1853054e-07, 1.38091978043624e-10]]

'''

def run_experiment(filter_array, start = -2.5, end = 2.5, steps = 100, wait = 1, pause = 1, sample = 5, beep = True):
    
    filenames = []
    
    global experiment_stamp
    
    experiment_stamp = timestamp()
    
    for i in range(len(filter_array)):
        
        print('\n=========================================================' + 
              '\nCollecting data for Filter ' + str(filter_array[i]) + 
              '\n=========================================================')
        
        run_stamp = str(timestamp())
        
        x = take_measurements(start, end, steps, wait, pause, sample, beep)
        
        # # to be removed
        # x = [[-2.5, 0, -1.3667410000000002e-10, 8.889671476494505e-12],
        #  [-2.0, 0, -1.4799408e-10, 1.6130881439028586e-13],
        #  [-1.5, 0, -1.4554786e-10, 2.2029027758846197e-13],
        #  [-1.0, 0, -1.3123745999999999e-10, 9.28295559584339e-12],
        #  [-0.5, 0, 1.6367472e-09, 1.129458254916933e-11],
        #  [0.0, 0, 2.7525048e-08, 6.828350881435455e-11],
        #  [0.5, 0, 8.2781114e-08, 5.1791043472785785e-11],
        #  [1.0, 0, 1.0633418000000001e-07, 9.175570608959293e-11],
        #  [1.5, 0, 1.1358767999999999e-07, 1.289140395767674e-10],
        #  [2.0, 0, 1.1853054e-07, 1.38091978043624e-10]]
        
        x = pd.DataFrame(x, columns = ['Voltage/V', 'Uncertainty/V', \
                                       'Current/A', 'Uncertainty/A'])
        
        print()
        print(x)
        print()
        
        x.to_excel(str(run_stamp + '_Filter_' + str(filter_array[i]) + \
                       '.xlsx'), index=False)
            
        filenames.append([str(run_stamp + '_Filter_' + str(filter_array[i]) +\
                       '.xlsx'), str(filter_array[i])])
        
        print('Data saved as:\n ' , str(run_stamp + '_Filter_' + \
                                        str(filter_array[i]) + '.xlsx'))
        print('=========================================================\n')
            
        if i != len(filter_array) -1 :
            print("\nPlease change to Filter " + str(filter_array[i + 1]) + \
                  " and press \"Enter\" to continue\n")
            keyboard.wait('enter')
    
    filenames = pd.DataFrame(filenames, columns = ['Filename', 'Filter'])
    filenames.to_excel(str(experiment_stamp + '_Run.xlsx'), index=False)
    


#%%

''' load data '''

# df1raw  = []
# df2raw = []
# df3raw = []
# df4raw = []
# df5raw = []
# df6raw = []
# df1 = []
# df2 = []
# df3 = []
# df4 = []
# df5 = []
# df6 = []

def load(filename_input = ''):    
    
    global filenamesraw
    global filenames
    global filters
    
    if filename_input == '':
        filenamesraw = pd.read_excel(str(experiment_stamp + '_Run.xlsx'), engine = "openpyxl", skiprows = 0)
        filenames = np.array(filenamesraw)
        filters   = filenames[:,1]
        pass
    else:
        filenamesraw = pd.read_excel(str(filename_input), engine = "openpyxl", skiprows = 0)
        filenames = np.array(filenamesraw)
        filters   = filenames[:,1]
        
        
    if 1 in filters:
        global df1raw
        x = np.where(filters == 1)
        df1raw = pd.read_excel(str(filenames[x,0]).replace('[', '').replace(']', '').replace("'", ""), engine = "openpyxl", skiprows = 0)
        global df1
        df1 = np.array(df1raw)
        pass
    if 2 in filters:
        global df2raw
        x = np.where(filters == 2)
        df2raw = pd.read_excel(str(filenames[x,0]).replace('[', '').replace(']', '').replace("'", ""), engine = "openpyxl", skiprows = 0)
        global df2
        df2 = np.array(df2raw)
        pass
    if 3 in filters:
        global df3raw
        x = np.where(filters == 3)
        df3raw = pd.read_excel(str(filenames[x,0]).replace('[', '').replace(']', '').replace("'", ""), engine = "openpyxl", skiprows = 0)
        global df3
        df3 = np.array(df3raw)
        pass
    if 4 in filters:
        global df4raw
        x = np.where(filters == 4)
        df4raw = pd.read_excel(str(filenames[x,0]).replace('[', '').replace(']', '').replace("'", ""), engine = "openpyxl", skiprows = 0)
        global df4
        df4 = np.array(df4raw)
        pass
    if 5 in filters:
        global df5raw
        x = np.where(filters == 5)
        df5raw = pd.read_excel(str(filenames[x,0]).replace('[', '').replace(']', '').replace("'", ""), engine = "openpyxl", skiprows = 0)
        global df5
        df5 = np.array(df5raw)
        pass
    if 6 in filters:
        global df6raw
        x = np.where(filters == 6)
        df6raw = pd.read_excel(str(filenames[x,0]).replace('[', '').replace(']', '').replace("'", ""), engine = "openpyxl", skiprows = 0)
        global df6
        df6 = np.array(df6raw)
        pass

    
#%%

''' run '''

# filter_array = [2, 3, 4, 5, 6]   # choose filters WARNING: make sure that the filter list containing the wavelengths is change appropriately
# filter_array = [2, 3]
# filter_list = filter_array

# run_experiment(filter_array, start = -2.5, end = 2.5, steps = 5, wait = 0.0, pause = 0.25, sample = 2, beep = False)
# run_experiment(filter_array, start = -2.0, end = 0, steps = 40, wait = 5.0, pause = 0.05, sample = 10, beep = False)





# load()


# manual
# load(filename_input = '2023_10_18_Run.xlsx')
# load(filename_input = '2023_10_19_Run.xlsx')
# load(filename_input = '2023_10_20_Run.xlsx')
load(filename_input = '2023_10_23_Final_Run.xlsx')



# calculate_planck(pool = 3)

# calculate_planck(pool = 10, diag = False, fit_mode = 'quad')
# calculate_planck(pool = 5, diag = True, fit_mode = 'line')

# calculate_planck(pool = 20, diag = False, fit_mode = 'quad')


#%%

def plot_final_1(filter_choice = 3):
    
    x = eval("df" + str(filter_choice) + "[:,0]")
    y = eval("df" + str(filter_choice) + "[:,2]")
    
    pool = 5
    smallest_number, smallest_index = smallest_above_zero(filter_choice)
    list_x = x[(smallest_index):(smallest_index + pool)]
    list_y = y[(smallest_index):(smallest_index + pool)]
    
    m, c, mmin, mmax, cmin, cmax = fit_line_unweighted(list_x, list_y)
    
    
    x_intersect = -c/m
    x_intersect_max = -cmax/mmin
    x_intersect_min = -cmin/mmax
    
    
    # Plotting
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    plt.style.use('default')
    
    ax.scatter(x, y, marker = ".", s = 8, label = "Measurements", color = 'blue')
    ax.axhline(0, color = 'black', label = "Zero-crossing")
    ax.plot([-99, -98], [line(-99, m, c), line(-98, m, c)], color = 'red', label = 'Linear Fit')
    
    plt.title("Current against Voltage (Filter " + str(filter_choice) + ")")
    plt.xlabel("Voltage/V")
    plt.ylabel("Current/A")
    plt.grid()
    
    
    
    axins = zoomed_inset_axes(ax, 8, loc = 6) # zoom = 6
    axins.grid()
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="gray", )
    
    axins.set_xlim(-1.73, -1.58) # Limit the region for zoom
    axins.set_ylim(-1.5E-9, 1.5E-9)
    axins.yaxis.tick_right()
    axins.xaxis.tick_bottom()
    axins.axhline(0, color = 'black', label = "Zero-crossing")
    
    axins.plot([-1.74, -1.54], [line(-1.74, m, c), line(-1.54, m, c)], color = 'red', label = 'Linear Fit')

    
    
    axins.scatter(x, y, marker = ".", s = 20, label = "Measurements", color = 'blue')
    # axins.text(521, 0.498, "50x")
    axins.set_facecolor('white')
    
    ax.legend(loc = 2)
    ax.set_xlim(-2.1, 0.1)
    ax.set_ylim(-4.2E-9, 7.4E-8)
    plt.show()
    pass

#%%
x = np.array([-1.995, -1.985, -1.975, -1.965, -1.955, -1.945, -1.935, -1.925,
       -1.915, -1.905, -1.895, -1.885, -1.875, -1.865, -1.855, -1.845,
       -1.835, -1.825, -1.815, -1.805, -1.795, -1.785, -1.775, -1.765,
       -1.755, -1.745, -1.735, -1.725, -1.715, -1.705, -1.695, -1.685,
       -1.675, -1.665, -1.655, -1.645, -1.635, -1.625, -1.615, -1.605,
       -1.595, -1.585, -1.575, -1.565, -1.555, -1.545, -1.535, -1.525,
       -1.515, -1.505, -1.495, -1.485, -1.475, -1.465, -1.455, -1.445,
       -1.435, -1.425, -1.415, -1.405, -1.395, -1.385, -1.375, -1.365,
       -1.355, -1.345, -1.335, -1.325, -1.315, -1.305, -1.295, -1.285,
       -1.275, -1.265, -1.255, -1.245, -1.235, -1.225, -1.215, -1.205,
       -1.195, -1.185, -1.175, -1.165, -1.155, -1.145, -1.135, -1.125,
       -1.115, -1.105, -1.095, -1.085, -1.075, -1.065, -1.055, -1.045,
       -1.035, -1.025, -1.015, -1.005, -0.995, -0.985, -0.975, -0.965,
       -0.955, -0.945, -0.935, -0.925, -0.915, -0.905, -0.895, -0.885,
       -0.875, -0.865, -0.855, -0.845, -0.835, -0.825, -0.815, -0.805,
       -0.795, -0.785, -0.775, -0.765, -0.755, -0.745, -0.735, -0.725,
       -0.715, -0.705, -0.695, -0.685, -0.675, -0.665, -0.655, -0.645,
       -0.635, -0.625, -0.615, -0.605, -0.595, -0.585, -0.575, -0.565,
       -0.555, -0.545, -0.535, -0.525, -0.515, -0.505, -0.495, -0.485,
       -0.475, -0.465, -0.455, -0.445, -0.435, -0.425, -0.415, -0.405,
       -0.395, -0.385, -0.375, -0.365, -0.355, -0.345, -0.335, -0.325,
       -0.315, -0.305, -0.295, -0.285, -0.275, -0.265, -0.255, -0.245,
       -0.235, -0.225, -0.215, -0.205, -0.195, -0.185, -0.175, -0.165,
       -0.155, -0.145, -0.135, -0.125, -0.115, -0.105, -0.095, -0.085,
       -0.075, -0.065, -0.055, -0.045, -0.035, -0.025, -0.015, -0.005])

y = np.array([ 4.250800e-10,  8.711800e-10, -2.421200e-10,  3.459000e-11,
        2.047870e-09,  1.118940e-09, -1.917400e-09,  6.778000e-10,
        2.529560e-09,  7.606600e-10, -1.673350e-09,  2.656280e-09,
        1.287890e-09,  8.437700e-10,  7.766400e-10,  1.228450e-09,
       -9.913400e-10,  1.816430e-09,  1.808670e-09,  1.788190e-09,
        1.286930e-09,  2.330860e-09,  2.110000e-09,  2.218720e-09,
        2.743010e-09,  2.319710e-09,  3.231570e-09,  3.141800e-09,
        3.607660e-09,  3.303270e-09,  4.135630e-09,  3.572030e-09,
        4.643722e-09,  5.468397e-09,  3.627993e-09,  4.844614e-09,
        5.684124e-09,  6.360090e-09,  4.540770e-09,  5.779680e-09,
        7.969600e-09,  5.601130e-09,  6.295700e-09,  7.879530e-09,
        7.250230e-09,  7.381120e-09,  7.313160e-09,  8.898130e-09,
        7.651620e-09,  8.683970e-09,  9.244200e-09,  9.503900e-09,
        8.683600e-09,  1.109590e-08,  9.192200e-09,  1.155880e-08,
        9.096300e-09,  1.170770e-08,  1.185620e-08,  1.102160e-08,
        1.261430e-08,  1.286690e-08,  1.073630e-08,  1.481720e-08,
        1.149350e-08,  1.350940e-08,  1.285890e-08,  1.571780e-08,
        1.541280e-08,  1.469560e-08,  1.379480e-08,  1.322740e-08,
        1.954500e-08,  1.611590e-08,  1.404380e-08,  1.601720e-08,
        2.205250e-08,  1.658650e-08,  1.951990e-08,  1.585320e-08,
        2.070040e-08,  2.209670e-08,  1.493920e-08,  1.682040e-08,
        3.113840e-08,  1.699050e-08,  1.750630e-08,  1.799950e-08,
        2.334480e-08,  3.571220e-08,  1.521470e-08,  3.273070e-08,
        2.680640e-08,  1.332010e-08,  2.073270e-08,  4.095590e-08,
        1.883410e-08,  2.253390e-08,  3.610060e-08,  2.879810e-08,
        2.168620e-08,  2.868000e-08,  2.315900e-08,  4.701800e-08,
        3.069300e-08,  1.839000e-08,  2.660600e-08,  3.640900e-08,
        3.086900e-08,  4.234400e-08,  4.568900e-08,  2.851600e-08,
        2.575100e-08,  2.515300e-08,  2.986000e-08,  7.189600e-08,
        2.975400e-08,  2.194800e-08,  3.111100e-08,  3.470700e-08,
        5.531200e-08,  3.732100e-08,  3.824600e-08,  3.586600e-08,
        4.682600e-08,  1.955300e-08,  4.623300e-08,  3.712100e-08,
        8.714400e-08,  2.936500e-08,  6.014500e-08,  4.711500e-08,
        2.969400e-08,  6.848000e-08,  3.744100e-08,  5.812000e-08,
        3.401000e-08,  6.404600e-08,  4.497000e-08,  3.997700e-08,
        5.648200e-08,  4.150100e-08,  9.094700e-08,  5.758200e-08,
        5.128600e-08,  5.526400e-08,  5.926300e-08,  5.809600e-08,
        3.616600e-08,  9.368400e-08,  4.671900e-08,  2.951100e-08,
        1.258770e-07,  1.633600e-08,  2.860200e-08,  8.315600e-08,
        6.849300e-08,  6.921700e-08,  5.594700e-08,  7.960600e-08,
        4.476500e-08,  6.559800e-08,  1.009250e-07,  6.943000e-08,
        3.418600e-08,  2.825800e-08,  1.585830e-07,  7.116800e-08,
        7.000400e-08,  7.095200e-08,  8.028400e-08,  4.379000e-08,
        1.020150e-07,  5.879800e-08,  1.115180e-07,  3.392800e-08,
        1.166270e-07,  9.092200e-08,  1.964600e-08,  1.180130e-07,
        6.617600e-08,  1.086840e-07,  8.062200e-08,  6.235400e-08,
        1.110440e-07,  1.068720e-07,  6.673100e-08,  5.166800e-08,
        1.209300e-07,  6.805600e-08,  9.520200e-08,  8.790400e-08,
        6.667800e-08,  1.226070e-07,  3.271100e-08,  4.024700e-08,
        2.144950e-07,  7.122900e-08,  9.611100e-08,  1.100530e-07])



# plt.figure()
# plt.plot(x, y)
# plt.plot(x, df3[0:-1,2])
# plt.axhline(0)
# plt.show()

def plot_final_2(filter_choice = 3):
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    plt.style.use('default')
    
    ax.fill_between([-99, -1.825], [-999, -999], [999, 999], color = "grey", alpha = 0.25, label = "Insignificant Area")
    ax.axhline(0, color = 'black', label = "Zero-crossing", linewidth = 1.0)
    ax.scatter(x, df3[0:-1,2], marker = ".", s = 8, label = "Measurements", color = 'blue')
    ax.plot(x, y, color = 'red', linewidth = 1.0, label = 'Derivative')
    
    plt.title("Derivatives Method of Determining Planck's Constant (Filter " + str(filter_choice) + ")")
    plt.xlabel("Voltage/V")
    plt.ylabel("Current/A")
    plt.grid()
    
    
    # window
    axins = zoomed_inset_axes(ax, 5, loc = 6) # zoom = 6
    axins.grid()
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="gray", )
    
    axins.set_xlim(-1.93, -1.71) # Limit the region for zoom
    axins.set_ylim(-5.5E-9, 5.5E-9)
    axins.yaxis.tick_right()
    axins.xaxis.tick_bottom()
    axins.set_facecolor('white')
    
    axins.fill_between([-99, -1.825], [-999, -999], [999, 999], color = "grey", alpha = 0.25, label = "Insignificant Area")
    axins.axhline(0, color = 'black', label = "Zero-crossing", linewidth = 1.0)
    axins.scatter(x, df3[0:-1,2], marker = ".", s = 8, label = "Measurements", color = 'blue')
    axins.plot(x, y, color = 'red', linewidth = 1.0, label = 'Derivative')
    
    ax.legend(loc = 2, framealpha = 1.00)
    ax.set_xlim(-2.1, 0.1)
    ax.set_ylim(-1.3E-8, 2.25E-7)
    plt.show()
    pass
    
    
    
#%%

calculate_planck(pool = 5, diag = False, fit_mode = 'line')
a  = "Straight line fit with extrapolation from 5 points"
a0 = 5.687E-34
aU = 5.729E-34 - a0
aL = a0 - 5.628E-34


calculate_planck(pool = 10, diag = False, fit_mode = 'line')
b  = "Quadriatic fit with extrapolation from 5 points"
b0 = 5.678E-34
bU = 6.788E-34 - b0
bL = b0 - 4.747E-34


calculate_planck(pool = 10, diag = False, fit_mode = 'quad')
c  = "Quadriatic fit with extrapolation from 10 points"
c0 = 5.772E-34
cU = 6.366E-34 - c0
cL = c0 - 5.231E-34


d  = "Derivative method with 10 significant points"
d0 = 6.712E-34
dU = 6.7851E-34 - d0
dL = d0 - 6.573E-34


def plot_final_3():
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    plt.style.use('default')
    
    ax.scatter(0, a0, label = a, zorder = 10, color = 'red', marker = '_')
    ax.errorbar(0, a0, yerr = ([aL], [aU]), fmt = '', capsize = 5.0, zorder = 5, color = 'red')
    
    ax.scatter(5, b0, label = b, zorder = 10, color = 'darkgreen', marker = '_')
    ax.errorbar(5, b0, yerr = ([bL], [bU]), fmt = '', capsize = 5.0, zorder = 5, color = 'darkgreen')
    
    ax.scatter(10, c0, label = c, zorder = 10, color = 'mediumblue', marker = '_')
    ax.errorbar(10, c0, yerr = ([cL], [cU]), fmt = '', capsize = 5.0, zorder = 5, color = 'mediumblue')
    
    ax.scatter(15, d0, label = d, zorder = 10, color = 'darkorange', marker = '_')
    ax.errorbar(15, d0, yerr = ([dL], [dU]), fmt = '', capsize = 5.0, zorder = 5, color = 'darkorange')
    
    ax.axhline(6.626E-34, label = "Actual value", color = 'black', linestyle = 'dashed')
    
    ax.set_xlim(-3.00, 18.00)
    ax.set_ylim(4.60E-34, 8.00E-34)
    
    
    plt.xticks(ticks = [0, 5, 10, 15], labels = ["SL-5", "Q-5", "Q-10", "Derv"])
    
    plt.title("Summary of Results")
    plt.grid()
    plt.legend(loc = 2, framealpha = 1.00)
    plt.ylabel("Resulting Value of Planck's Constant/ $Js$")
    plt.show()
    














#%%

# # run_experiment(filter_array, start = -2.0, end = 0.0, steps = 200, wait = 1.0, pause = 0.0, sample = 1, beep = False)


# load(filename_input = '2023_10_20_Run_2.xlsx') 


calculate_planck(pool = 5, diag = False, fit_mode = 'line')
calculate_planck(pool = 10, diag = False, fit_mode = 'line')
calculate_planck(pool = 10, diag = False, fit_mode = 'quad')
#%%





