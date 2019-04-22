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
    #tests.check_sequence_average(tomo, num_rotations)

    ## select zero angle rotation and check readout tomography
    #tests.check_corrTomo(weak, tomo, z0, num_rotations)

    ## evaluate feedback
    scores = get_scores(weak)
    #tests.check_scores_as_appliedAngle(scores, tomo) 
    #tests.check_scoreThreshold(scores, tomo)
    
    lowError_weak_outcomes = filter_by_scores(weak, scores, threshold=0.1)

    ## calcualte arrow of time
    calculate_AoT(lowError_weak_outcomes, z0)
    plt.show(); return 0;

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
    ##@ is substracting respective means valid?
    weak_mean = np.mean(weak)
    strong_mean = np.mean(strong)
    weak -= weak_mean
    strong -= strong_mean
    # weak, strong means: -173.5, -175.8

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
    #ground_val = 1.556
    #excited_val = -1.355
    ground_val = 1.4 ## tuned to get close agreement between get_analytic_q() and data
    excited_val = -ground_val
    ## 04/04/19: starting over, these are from util.theory ##@@ delete after debugging
    #S = 0.41  # "copied '' from "calibrate readout" ''" <-- copied from util.theory_xz()
    #dV = 3.31 # "copied '' from "calibrate readout" ''" <-- copied from util.theory_xz()
    # below: copied from util.make_theory (tuned to match correlated tomo)
    S = 0.357
    dV = 2.5
    
    num_measurements = len(weak_measurement)
    num_bins = 100
    sig2 = dV**2/S ## sigma squared

    ## forward probability
    gnd_gauss = np.exp( -(weak_measurement - ground_val)**2 /2/sig2)
    ex_gauss = np.exp( -(weak_measurement - excited_val)**2 /2/sig2)
    for_log_prob = np.log( (1+z0)/2 *gnd_gauss  +  (1-z0)/2*ex_gauss )

    ## backwards probability
    ## passive transformation: use negative record
    gnd_gauss_back = np.exp( -(-weak_measurement - ground_val)**2 /2/sig2)
    ex_gauss_back = np.exp( -(-weak_measurement - excited_val)**2 /2/sig2)
    
    #x_final, z_final = util.theory_xz(weak_measurement)
    gamma = weak_measurement/sig2*ground_val
    #z_final = np.tanh(gamma)
    z_final = ( np.sinh(gamma) + z0*np.cosh(gamma) ) /( np.cosh(gamma) + z0*np.sinh(gamma) )
    #z_final = np.tanh(weak_measurement/sig2*ground_val) ##@ use above line instead; note scaling
    back_log_prob = np.log( (1+z_final)/2 *gnd_gauss_back +  (1-z_final)/2*ex_gauss_back)

    ## log ratio
    Q = for_log_prob - back_log_prob
    Q_prob, bins = np.histogram(Q, bins=num_bins)
    Q_prob_xs = bins[1:] ## use right-hand side of bin as x position

    ## compare to theory
    Q_analytic_prob  = get_analytic_probQ(Q_prob_xs, z0)
    Q_analytic_prob *= num_measurements # scale prob. to occurances
    #Q_analytic_prob *= 0.25 ##aribtrary to get "better looking" agreement
 
    fig, q_hist_ax = plt.subplots() 
    q_hist_ax.semilogy( Q_prob_xs, Q_prob,'k-', label="Hist counts")
    q_hist_ax.semilogy( Q_prob_xs, Q_analytic_prob,'r--', label="Analytic")
    q_hist_ax.set_xlabel("Q", fontsize=20)
    q_hist_ax.set_ylabel("Counts", fontsize=20)
    q_hist_ax.set_xlim(-2,2)
    plt.legend()
  
    ## DEBUGGING 
    ################################################ 
    # compare analytic Q to exp. Q
    '''
    ## compare to analytic results
    weak_sample = weak_measurement[:1500]
    gamma_sample = gamma[:1500]
    z_sample = z_final[:1500]
    pf_sample = np.exp(for_log_prob[:1500])
    pf_sample /= np.sqrt(2*np.pi*sig2)
    Q_sample = Q[:1500]

    fig, tmp_cf = plt.subplots()
    tmp_cf.plot(Q_sample, z_sample, '.k', label="data")
    tmp_cf.plot(Q_prob_xs, Q_analytic_prob, label='thy')
    tmp_cf.set_xlabel("Q")
    tmp_cf.set_ylabel("$P_f(\gamma$|Q)")
    plt.legend()
    #'''
##END calculate_AoT
    

def get_analytic_q(weak,z0=0):
    #S = 0.41  # see calc_AoT()
    #dV = 3.31 # see calc_AoT()
    S = 0.357 # from util. from theory
    dV = 2.5
    tT = dV**2/S ## tau over T via equating Gaussian variance

    return 2*np.log( np.cosh(weak/tT)) #+ z0*np.sinh(weak/dV**2))
##END get_analytic_q


def get_analytic_probQ(Q, z0=0):
    '''
    DESCRIPTION: recreates Jordan Dressel's (2017) calculation of Q in QND measurement
    INPUT: array of Q values for which to calculate probability
            z0 !=0 is current not understood
    OUTPUT: calculated probability density
    '''
   
    # see calc_AoT() for values 
    #S = 0.41  
    #dV = 3.31 
    S = 0.357 # from calc_
    dV = 2.5
    tT = dV**2/S ## tau over T; equal to Gaussian variance

    gamma_Q = np.arccosh(np.exp(Q/2))
    gamma_Q /= 1.4 # scale for ground state values
    # below not used but included for completeness
    weak = gamma_Q*tT/1.4 # scale for dV

    zf_Q = np.sqrt(np.exp(Q)-1)*np.exp(-Q/2.)
    arg = -0.5/tT -0.5*tT*gamma_Q**2 +Q/2.
    pf_Q = np.sqrt(1/2/np.pi/tT)*0.5 *np.exp(arg)*2
   
    return pf_Q/2/zf_Q
##END get_analytic_q


if __name__ == '__main__':
    main()
