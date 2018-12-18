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


def check_corrTomo(weak, tomo, z0=0., num_rotations=26):
    weak_zero_angle_xyz = weak[num_rotations//2::num_rotations]
    tomo_zero_angle_xyz = tomo[num_rotations//2::num_rotations]
    
    ## x,y,z tomo
    all_tomo_dict = {"x":[], "y":[], "z":[]}
    all_tomoErr_dict = {"x":[], "y":[], "z":[]}
    fig, tomo_ax = plt.subplots()
    for i,label in enumerate("xyz"):
        single_axis_weak = weak_zero_angle_xyz[i::3]
        single_axis_tomo = tomo_zero_angle_xyz[i::3]

        print("corr tomo "+label)
        coord, binned_tomo, binned_tomo_err = util.correlate_tomography(single_axis_weak, single_axis_tomo\
                ,bin_min=-18, bin_max=18)
        all_tomo_dict[label] = binned_tomo
        all_tomoErr_dict[label] = binned_tomo_err
        #binned_tomo = binned_tomo[1:]
        #binned_tomo_err = binned_tomo_err[1:]
      
        if label=='y': continue      
        tomo_ax.errorbar(coord, binned_tomo, fmt='o', yerr=binned_tomo_err, label=label)
    x,z = util.theory_xz(coord, z0)
        
    #plt.plot(all_tomo_dict["x"], all_tomo_dict["z"], 'ok')
    tomo_ax.plot(coord, z, label='Theory z')
    tomo_ax.plot(coord, x, label='Theory x')
    tomo_ax.legend(loc=4)
    tomo_ax.set_xlabel("Record, $r$")
    tomo_ax.set_ylabel("Average Value")
    tomo_ax.set_ylim(-1,1)
    plt.show()
        
    return all_tomo_dict, all_tomoErr_dict
##END check_corrTomo


def check_scores_v2(scores, weak, tomo,num_rotations=26):
    ## don't look at score threshold, but make sure tomo matches the applied angle
    fig, bloch = plt.subplots()

    ## get blocks of x or z tomography (all angles)
    N = len(tomo)
    intrablock_indices = np.arange(N)%(3*num_rotations)
    z_indices = intrablock_indices >= 2*num_rotations
    x_indices = intrablock_indices < num_rotations

    ## with scores set to applied angle, separate tomographic average with applied angle
    z_scores = scores[z_indices]
    z_tomo = tomo[z_indices]
    bins, score_tomo, score_tomo_err = util.correlate_tomography(z_scores, z_tomo, bin_min=-0.25,bin_max=0.25)
    bloch.plot(bins, score_tomo, 'ok')
    bloch.set_ylim(-1,1)

    ## compare to expected rotation
    r = 0.7 ## from zero-angle maximum x-coordinate
    angles = bins
    plt.plot( angles, r*np.sin(np.pi*angles))# - np.pi/4) )
    plt.show()

##END check_scores_v2


def check_scores(scores, weak, tomo):
    fig, ax = plt.subplots()
    fig, bloch = plt.subplots()
    for score_thresh in np.linspace(0.1,1.2, 12):
        filtered_weak = weak[scores<score_thresh]
        filtered_tomo = tomo[scores<score_thresh]

        max_n = len(filtered_tomo) - len(filtered_tomo)%3
        x_weak = filtered_weak[0:max_n:3]
        z_weak = filtered_weak[2:max_n:3]
        x_tomo = filtered_tomo[0:max_n:3]
        z_tomo = filtered_tomo[2:max_n:3]

        bins, x_corrTomo,x_corrTomo_err =  util.correlate_tomography(x_weak, x_tomo)
        bins, z_corrTomo ,z_corrTomo_err =  util.correlate_tomography(z_weak, z_tomo)
        #ax.plot(score_thresh, max(z_corrTomo), 'ok')
        #bloch.plot(x_tomo, z_tomo)

        avg_x = np.mean( x_tomo )
        avg_z = np.mean( z_tomo )
        err_x = util.binomial_error( x_tomo )
        err_z = util.binomial_error( z_tomo )
        ax.errorbar(score_thresh, avg_x, yerr=err_x, color='k',fmt='o')
        #bloch.plot(avg_x, avg_z, 'ok')
        bloch.plot(x_corrTomo, z_corrTomo, 'o', markeredgecolor=None)

    util.make_bloch(bloch)
    ax.set_xlabel("Score threshold")
    ax.set_ylabel("Average X Tomography")
    ax.set_ylim(-1,1)
    #plt.ylim(-0.1,1.1)
    plt.show()
##END check_scores


def check_sequence_reading(tomo, num_rotations):
    '''
    shifted = np.copy(tomo)
    num_repeats = len(tomo)//(num_rotations*3)
    shifted.shape = num_rotations*3, num_repeats
    plt.plot(np.mean(shifted,axis=1), 'ok')
    #''' 

    #'''
    #num_repeats = len(tomo)//(num_rotations*3) 
    #bins = np.zeros(num_rotations) 
    #tomo = tomo[:1000] ## cutoff most of the datea
    for i in range(num_rotations*3):
        setOf_angle_tomo = tomo[i::num_rotations*3]
        plt.plot(i, np.mean(setOf_angle_tomo), 'ok') 
    #'''
    plt.xlabel("Sequence step")
    plt.ylabel("Average Tomography")
    plt.ylim(-1,1)
    plt.show()
##END check_sequence_reading
