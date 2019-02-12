# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:07:17 2018

@author: J. Monroe

"""
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.getcwd())
import util
import tests 

global num_rotations
num_rotations=26

def main():
    '''
    OUTLINE: load data correlate tomography
        make theory curves
        evaluate scores
        select on scores
        calculate probabilities
        profit!
    '''
    ## load data
    data_dir = "../data/"
    use_plusX_initial = True
    if use_plusX_initial:
        data_file = "fullData_26Rot_-0.45,0.35Amp_5420Start_145nsInteg_f1.5_xyzTomo_p1"
        z0 = 0.
    else:
        data_file = "fullData_prepZ,0.5_26Rot_-0.45,0.35Amp_5420Start_145nsInteg_f1.5_xyzTomo_p1"
        z0=1./np.sqrt(2)
    weak_measurement, strong_measurement = np.loadtxt(data_dir+data_file)
    global num_rotations

    ## clean data
    #   trim to integer number of trajectories per rotation angle
    #   substract mean
    weak, strong = clean_data(weak_measurement,strong_measurement)

    ##  calculate tomographic outcomes
    #tests.find_readout_threshold(weak, strong) 
    readout_threshold = 4 ## tuned to make corr tomo match
    if not use_plusX_initial: readout_threshold = 6.5
    #readout_threshold = -5 ## ignore above, make uncorrelated tomography average to zero
    tomo = measurement_to_tomo(strong, readout_threshold)
    #tests.check_sequence_reading(tomo, num_rotations)

    ## select zero angle rotation and check readout tomography
    #tests.check_corrTomo(weak, tomo, z0, num_rotations)

    ## evaluate feedback
    scores = get_scores(weak)
    #tests.check_scores_as_appliedAngle(scores, tomo) 
    #tests.check_scoreThreshold(scores, tomo)
    
    lowError_outcomes = filter_by_scores(weak, scores, threshold=0.1)

    ## calcualte arrow of time
    calculate_AoT(lowError_outcomes, z0)

    return weak, tomo
##END main()


def clean_data(raw_weak, raw_strong):
    ## removes a few inconvenient features of data
    
    ## throw away last incomplete set of tomography
    num = len(raw_weak)
    global num_rotations 
    num_extra = num%(num_rotations*3)
    weak = np.copy(raw_weak[:-num_extra])
    strong= np.copy(raw_strong[:-num_extra])

    ## subtract mean
    weak_mean = np.mean(weak)
    strong_mean = np.mean(strong)
    weak -= weak_mean
    strong -= strong_mean

    return weak, strong
## clean_data


def measurement_to_tomo(strong_measurement,threshold):
    return  np.sign(threshold-strong_measurement)
##END measurement_to_tomo


def get_scores(weak):
    ## score weak measurement outcomes based on accuracy of applied feedback

    global num_rotations
    feedback_angle_list = np.linspace(-np.pi/4, np.pi/4, num_rotations)
    scores = np.zeros(len(weak))

    x,z = util.theory_xz(weak)
    traj_angle = np.arctan(z/x)
    app_angle = np.tile(feedback_angle_list, 3*len(weak)//(num_rotations*3))
    scores = abs(traj_angle-app_angle)/(np.pi/2)

    return scores
    ## test cases for tests.check_scores_as_appliedAngle()
    #app_angle[app_angle<0] = 2*np.pi
    #return np.abs(app_angle/np.pi)
    #return np.abs(traj_angle/np.pi)
      
##END get_scores 


def filter_by_scores(to_filter, criteria, threshold=0.1):
    min_args = np.where(criteria < threshold)
    return to_filter[min_args]
##END filter_by_scores


def calculate_AoT(weak_measurement, z0=0):
    '''
    DESCRIPTION: estimates arrow of time ratio infered from measurement outcomes
    INPUT:  weak_measurement: filtered measurement values for all tomographic axes
    OUTPUT: plot
    '''

    ## setting offsets:
    ## used data from "acof_analysis_030218_noon.pxp" and 
    ## compared mean of this data set (v3_gooder) to the mean of pi/no pi waves in the above Igor file
    ## signs are made up
    ground_val = 1.556
    #excited_val = -1.355
    excited_val = -ground_val
    S = 0.41  # "copied '' from "calibrate readout" ''" <-- copied from util.theory_xz()
    dV = 3.31 # "copied '' from "calibrate readout" ''" <-- copied from util.theory_xz()
    num_measurements = len(weak_measurement)
    num_bins = 100

    ## forward probability
    gnd_gauss = np.exp( -(weak_measurement - ground_val)**2 *S/2/dV )
    ex_gauss = np.exp( -(weak_measurement - excited_val)**2 *S/2/dV )
    for_log_prob = np.log( (1+z0)/2 *gnd_gauss  +  (1-z0)/2*ex_gauss )

    ## backwards probability
    gnd_gauss2 = np.exp( -(-weak_measurement - ground_val)**2 *S/2/dV )
    ex_gauss2 = np.exp( -(-weak_measurement - excited_val)**2 *S/2/dV )
    x_final, z_final = util.theory_xz(weak_measurement)
    # active transformation: flip coordiate of z
    back_log_prob = np.log( (1+z_final)/2 *gnd_gauss2  +  (1-z_final)/2*ex_gauss2) 

    ## log ratio
    Q = for_log_prob - back_log_prob
    hist_counts, hist_bins= np.histogram(Q, bins=num_bins)
    Q_analytic_prob  = get_analytic_probQ(hist_bins[1:],z0=0)
    Q_analytic_prob *= num_measurements # scale prob. to occurances
    weak_sample = weak_measurement[:500]
    Q_analytic = get_analytic_q(weak_sample ,z0=0)
    ## plots 
    #plt.hist(for_log_prob, bins=30, color='b', alpha=0.4, label='Forward')
    #plt.hist(back_log_prob, bins=30, color='r', alpha=0.4, label='Back')
    #plt.legend()
    
    fig, tmp_cf = plt.subplots()
    plt.plot(weak_sample, Q[:500],'.k',label="data")
    plt.plot(weak_sample, Q_analytic, 'r.',label="thy")
    plt.legend()

    fig, q_hist_ax = plt.subplots() 
    q_hist_ax.semilogy( hist_bins[1:], hist_counts,'k-')
    q_hist_ax.semilogy( hist_bins[1:], Q_analytic_prob/30,'r--')
    #@@@ WHY NOT NORMALIZED?
    #q_hist_ax.fill_between( hist_bins[1:], hist_counts,color='r', alpha=0.3)
    q_hist_ax.set_xlabel("Q", fontsize=20)
    q_hist_ax.set_ylabel("Counts", fontsize=20)
    q_hist_ax.set_xlim(-2,2)
    plt.show()
##END calculate_AoT
    

def get_analytic_q(weak,z0=0):
    S = 0.41  # see calc_AoT()
    dV = 3.31 # see calc_AoT()

    return 2*np.log( np.cosh(weak/dV**2)) #+ z0*np.sinh(weak/dV**2))
##END get_analytic_q

def get_analytic_probQ(qs, z0=0):
    '''
    DESCRIPTION: calculates Jordan Dressel's calculation (2017) of the arrow of time in QND measurement
    INPUT: array of q values for which to calculate prob.
            z0 !=0 is current not understood
    OUTPUT: calculated probability density
    '''
    
    S = 0.41  # see calc_AoT()
    dV = 3.31 # see calc_AoT()
    tT = dV**2/S ## via equating Gaussian variance
    #tT = 1
    

    ## three terms: P(Q) = norm *factor *exponential
    norm = np.sqrt(tT/2/np.pi)
    factor = np.sqrt( np.exp(2*qs) /(np.exp(qs)-1) )
    exponent_arg = -0.5/tT - tT* (np.arccosh( np.exp(qs) ))**2
    
    return norm *factor *np.exp(exponent_arg)

##END get_analytic_q

if __name__ == '__main__':
    main()
