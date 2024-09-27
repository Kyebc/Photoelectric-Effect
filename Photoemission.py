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