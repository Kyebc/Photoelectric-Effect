# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 11:56:56 2023

@author: kyebchoo
"""

import numpy as np
import scipy as sci
import scipy.signal as sig
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import math
import pandas as pd
from matplotlib import cbook
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

#%%
'''
Part A) Initiation of code ####################################################
'''



'loading light profiles'
light_profile = pd.read_excel("2023_10_13_light_profile.xlsx", engine = "openpyxl")
filter_list = np.array(light_profile)[:,0]
centroid_lambda = np.array(light_profile)[:,1]
peak_lambda = np.array(light_profile)[:,2]
FWHM_lambda = np.array(light_profile)[:,3]



'loading data'
df2raw = pd.read_excel("2023_10_13_Filter_2.xlsx", engine = "openpyxl", skiprows = 1)
df3raw = pd.read_excel("2023_10_13_Filter_3.xlsx", engine = "openpyxl", skiprows = 1)
df4raw = pd.read_excel("2023_10_13_Filter_4.xlsx", engine = "openpyxl", skiprows = 1)
df5raw = pd.read_excel("2023_10_13_Filter_5.xlsx", engine = "openpyxl", skiprows = 1)
df6raw = pd.read_excel("2023_10_13_Filter_6.xlsx", engine = "openpyxl", skiprows = 1)
df2 = np.array(df2raw)
df3 = np.array(df3raw)
df4 = np.array(df4raw)
df5 = np.array(df5raw)
df6 = np.array(df6raw)



'declaring global constants'
c0 = 299792458
e = -1.602 * (10**-19)



#%%
'''
Part B) Define common functions with reference to my own previous work ########
'''



def line(x, m, c):
    return m*x + c

def fit_line(x, y):
    popt, pcov = curve_fit(line, x, y)
    m = popt[0]
    c = popt[1]
    mmin = popt[0] - np.sqrt(pcov[0, 0])
    mmax = popt[0] + np.sqrt(pcov[0, 0])
    cmin = popt[1] - np.sqrt(pcov[1, 1])
    cmax = popt[1] + np.sqrt(pcov[1, 1])
    
    return m, c, mmin, mmax, cmin, cmax



#%%

'''
Part C) Defining functions for use ############################################
'''


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



def search_zero_intersect(filter_choice, pool = 3, diag = False):
    
    smallest_number, smallest_index = smallest_above_zero(filter_choice)
    extract_x     = eval("df" + str(filter_choice) + "[:,0]")
    extract_x_err = eval("df" + str(filter_choice) + "[:,1]") # not used
    extract_y     = eval("df" + str(filter_choice) + "[:,2]")
    extract_y_err = eval("df" + str(filter_choice) + "[:,3]") # not used
    list_x = extract_x[(smallest_index - pool + 1):(smallest_index + 1)]
    list_y = extract_y[(smallest_index - pool + 1):(smallest_index + 1)]
    
    m, c, mmin, mmax, cmin, cmax = fit_line(list_x, list_y)
    x_intersect = -c/m
    x_intersect_max = -cmax/mmin
    x_intersect_min = -cmin/mmax
    
    if diag == True:
        plot(filter_choice)
    
    return x_intersect, x_intersect_max, x_intersect_min

    
    

def plot(filter_choice):
    "plotting"
    x = eval("df" + str(filter_choice) + "[:,0]")
    y = eval("df" + str(filter_choice) + "[:,2]")
    
    plt.figure()
    plt.scatter(x, y)
    plt.title("Current against Voltage (Filter " + str(filter_choice) + ")")
    plt.xlabel("Voltage/V")
    plt.ylabel("Current/nA")
    plt.grid()
    plt.show()

def calculate_planck():
    V_ec = []

    frequency = c0/(centroid_lambda * (10**-9))
    
    for i in range(0, len(filter_list)):
        V_ec.append(search_zero_intersect(int(filter_list[i]), pool = 3)[0])
        
    plt.figure()
    plt.scatter(frequency, V_ec)
    plt.show()
    
    m, c, mmin, mmax, cmin, cmax = fit_line(frequency, V_ec)
    
    h = m * e
    print(h)


  
