import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.getcwd())
import util

def find_readout_threshold(weak,strong):
    num_rotations= 27
    weak_zero_angle_xyz = weak[::num_rotations//2]
    strong_zero_angle_xyz = strong[::num_rotations//2]
    i=2  ## 0,1,2 is x,y,z
    single_axis_weak = weak_zero_angle_xyz[i::3]
    single_axis_strong = strong_zero_angle_xyz[i::3]

    ## simply look at histogram
    fig, hist_ax = plt.subplots()
    hist_ax.hist(single_axis_strong,bins=100)
    hist_ax.set_xlabel(r"Strong measurement outcome ($\theta_{app}=0$)")
  
    plt.show()
##END find_thresh 


def check_corrTomo(weak, tomo, z0=0.):
    num_rotations= 27
    weak_zero_angle_xyz = weak[::num_rotations//2]
    tomo_zero_angle_xyz = tomo[::num_rotations//2]
    
    ## x,y,z tomo
    all_tomo_dict = {"x":[], "y":[], "z":[]}
    all_tomoErr_dict = {"x":[], "y":[], "z":[]}

    fig, tomo_ax = plt.subplots()
    #for i,label in enumerate("xyz"):
    if True:
    ## just look at z for now.
        i = 2
        label = 'z'
        single_axis_weak = weak_zero_angle_xyz[i::3]
        single_axis_tomo = tomo_zero_angle_xyz[i::3]

        coord, bin_tomo, bin_tomo_err = util.correlate_tomography(single_axis_weak, single_axis_tomo)        
        all_tomo_dict[label] = bin_tomo
        all_tomoErr_dict[label] = bin_tomo_err
            
        tomo_ax.errorbar(coord, bin_tomo, fmt='o', yerr=bin_tomo_err, label=label)
    x,z = util.theory_xz(coord, z0)
        
    #plt.plot(all_tomo_dict["x"], all_tomo_dict["z"], 'ok')
    tomo_ax.plot(coord, z, label='Theory z')
    #tomo_ax.plot(coord, -x, label='(-)Theory x')
    tomo_ax.legend(loc=3)
    tomo_ax.set_xlabel("Record, $r$")
    tomo_ax.set_ylabel("Average Value")
    tomo_ax.set_ylim(-1,1)
    plt.show()
        
    return all_tomo_dict, all_tomoErr_dict
##END check_corrTomo


def check_scores(scores, weak, tomo):
    for score_thresh in np.linspace(0.1,1.2, 10):
        filtered_weak = weak[scores<score_thresh]
        filtered_tomo = tomo[scores<score_thresh]

        x_tomo = tomo[0::3]
        avg_x = np.mean( filtered_tomo )
        err_x = binomial_error( filtered_tomo )

        plt.errorbar(score_thresh, avg_x, yerr=err_x, color='k',fmt='o')

    plt.xlabel("Score threshold")
    plt.ylabel("Average X Tomography")
    #plt.ylim(-0.1,1.1)
    plt.show()
##END check_scores


def check_sequence_reading(tomo, num_rotations):
    #'''
    shifted = np.copy(tomo)
    num_repeats = len(tomo)//(num_rotations*3)
    shifted.shape = num_rotations*3, num_repeats
    plt.plot(np.mean(shifted,axis=1), 'ok')
    #''' 

    '''
    num_repeats = len(tomo)//(num_rotations*3) 
    bins = np.zeros(num_rotations) 
    for i in range(3*num_rotations):
        setOf_angle_tomo = tomo[i::81]
        plt.plot(i, np.mean(setOf_angle_tomo), 'ok') 
    #'''
    plt.xlabel("Sequence step")
    plt.ylabel("Average Tomography")
    plt.ylim(-1,1)
    plt.show()
##END check_sequence_reading
