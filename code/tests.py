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
  
    '''  ## old code before thresholding was done by a separate function
    ## sweep threshold to see what tomography contrast results
    fig, thresh_ax = plt.subplots()
    num_thresh = 50
    max_thresh = 10
    tomo_vs_thresh = np.zeros( (num_thresh,30) )

    for i,threshold in enumerate(np.linspace(-10,10,num_thresh)):
        coord, tomo, tomo_err = correlate_tomography(single_axis_weak, single_axis_strong, threshold)
        tomo_vs_thresh[i] = tomo
        
    thresh_ax.imshow(tomo_vs_thresh, extent= [coord[0],coord[-1],-10,10], vmin=-1,vmax=1, cmap='bwr')
    thresh_ax.set_xlabel("r")
    thresh_ax.set_ylabel("Threshold value")
    '''
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
        label = 'z' 
        i=2 
        single_axis_weak = weak_zero_angle_xyz[i::3]
        single_axis_tomo = tomo_zero_angle_xyz[i::3]

        coord, bin_tomo, bin_tomo_err = util.correlate_tomography(single_axis_weak, single_axis_tomo)        
        all_tomo_dict[label] = bin_tomo
        all_tomoErr_dict[label] = bin_tomo_err
            
    x,z = util.theory_xz(coord, z0)
        
    #plt.plot(all_tomo_dict["x"], all_tomo_dict["z"], 'ok')
    tomo_ax.errorbar(coord, bin_tomo, fmt='o', yerr=bin_tomo_err)
    tomo_ax.plot(coord, z, label='Theory')
    tomo_ax.legend(loc=2)
    tomo_ax.set_xlabel("Record, $r$")
    tomo_ax.set_ylabel("$<Z>$")
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


