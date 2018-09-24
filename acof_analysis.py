# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:07:17 2018

@author: J. Monroe

"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    '''
    OUTLINE:
        load data
        correlate tomography
        make theory curves
        evaluate scores
        select on scores
        calculate probabilities
    '''
    data_dir = r"C:\Data\Spring2018\reversed_feedback\data_version3\\"
    data_file = "fullData_26Rot_-0.45,0.35Amp_5420Start_145nsInteg_f1.5_xyzTomo_p2"
    
    weak_measurement, strong_measurement = np.loadtxt(data_dir+data_file)
    
    ''' Check input
    fig, ax1 = plt.subplots()
    ax1.plot(weak_measurement, ',')
    fig, ax2 = plt.subplots()
    ax2.plot(strong_measurement, ",")
    
    plt.show()
    '''
    
    ## select zero angle rotation and ch
    
    check_corrTomo(weak_measurement, strong_measurement)
##END main()
    
    
def check_corrTomo(weak, strong):
    num_rotations= 27
    weak_zero_angle_xyz = weak[::num_rotations//2]
    strong_zero_angle_xyz = strong[::num_rotations//2]
    
    ## x,y,z tomo
    #for i in range(3):
    if True:
        i = 2
        single_axis_weak = weak_zero_angle_xyz[i::3]
        single_axis_strong = strong_zero_angle_xyz[i::3]
        
        threshold = -173 # eye-balling based on strong signal display
        coord, tomo, tomo_err = correlate_tomography(single_axis_weak, single_axis_strong, threshold)
    ## just look at z for now.
    
    plt.plot(coord, tomo)
    plt.show()
##END check_corrTomo()
    
    
def correlate_tomography(to_bin, tomographic, threshold, bin_min=None, bin_max=None,num_bins=30):
    '''
    TODO: better names for "to_bin", "readout"
    '''
    
    ## bin the array to be binned
    if (bin_min is not None) and (bin_max is not None):
        bins = np.linspace(bin_min, bin_max, num_bins)
    else:
        bins = num_bins
    hist_values, bin_edges = np.histogram(to_bin, bins=bins)
    bin_xs = bin_edges[1:] # skip the left-most bin

    ## calculate average outcome for tomography in each bin
    tomo = np.zeros(num_bins)
    tomo_err = np.zeros(num_bins)
    sorted_toBin = np.sort(to_bin)
    data_index = 0
    point = sorted_toBin[0]
    readout = tomo[0]  
    for bin_index, bin_thresh in enumerate(bin_xs):
        while (point < bin_thresh):
            tomo[bin_index] += np.sign( readout - threshold)
            data_index +=1
            point = sorted_toBin[data_index]
            readout = tomographic[data_index]
            
        ##END loop through points belonging to bin
        N = hist_values[bin_index]
        tomo[bin_index] /= N
        ## calculate 95% CI for binomial error
        p = 0.5*tomo[bin_index] + 0.5 # convert from expectation value to probability
        tomo_err[bin_index] = 1.96 *np.sqrt(p*(1-p)/N) /2
        tomo_err[bin_index] *= 2 # convert back to expectation values.
    ##END loop through bins
    
    ## output
    return bin_xs, tomo, tomo_err
        
##END correlate_tomography
    

if __name__ == '__main__':
    main()