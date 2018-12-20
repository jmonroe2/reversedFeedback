import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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


def check_scores_as_appliedAngle(scores, tomo,num_rotations=26):
    ## with score set to the applied angle. Do we correctly index the rotated state?
    fig, bloch = plt.subplots()

    ## get blocks of x or z tomography (all angles)
    N = len(tomo)
    intrablock_indices = np.arange(N)%(3*num_rotations)
    z_indices = intrablock_indices >= 2*num_rotations
    x_indices = intrablock_indices < num_rotations
    x_scores = scores[x_indices]
    z_scores = scores[z_indices]
    x_tomo = tomo[x_indices]
    z_tomo = tomo[z_indices]

    ## correlated tomography connects score values with average tomography
    bins, binned_xTomo, binned_xTomo_err= util.correlate_tomography(x_scores, x_tomo, bin_min=-0.25,bin_max=0.25)
    bins, binned_zTomo, binned_zTomo_err= util.correlate_tomography(z_scores, z_tomo, bin_min=-0.25,bin_max=0.25)

    ## plots
    plt.scatter(binned_xTomo, binned_zTomo,s=20, c=bins, cmap='nipy_spectral')
    plt.colorbar(label=r"$\theta_{app}/\pi$")
    util.make_bloch(bloch)

    bloch.set_ylim(-1,1)

    ## compare to expected rotation
    fig, rot_ax = plt.subplots()
    using_plusX_init = True
    if using_plusX_init:
        r = 0.6 ## from zero-angle maximum x-coordinate
        theta = 0
    else:
        r = 0.7
        theta = np.pi/4
        
    angles = bins
    rot_ax.errorbar( bins, binned_zTomo,yerr=binned_zTomo_err, color='k',fmt='o')
    rot_ax.plot( angles, r*np.sin(np.pi*angles - theta))
    rot_ax.set_xlabel("Applied Angle [$\pi$]")
    rot_ax.set_ylabel("Z Tomo")
    plt.show()
##END check_scores_as_appliedAngle


def check_scoreThreshold(scores, tomo, num_rotations=26):
    fig, bloch = plt.subplots()
    fig, ax = plt.subplots()
    
    ## get blocks of x or z tomography (all angles, scores)
    N = len(tomo)
    intrablock_indices = np.arange(N)%(3*num_rotations)
    z_indices = intrablock_indices >= 2*num_rotations
    x_indices = intrablock_indices < num_rotations
    x_scores = scores[x_indices]
    z_scores = scores[z_indices]
    x_tomo = tomo[x_indices]
    z_tomo = tomo[z_indices]
      
    cmap = cm.get_cmap('Spectral')
    #score_threshold = 0.6
    score_min, score_max = 0.1, 1
    for score_threshold in np.linspace(score_min, score_max, 10):
        filtered_z_tomo = z_tomo[ z_scores < score_threshold ]     
        filtered_x_tomo = x_tomo[ x_scores < score_threshold ]     
        filtered_z_scores = z_scores[ z_scores < score_threshold ]     
        filtered_x_scores = x_scores[ x_scores < score_threshold ]     

        col = cmap((score_threshold-score_min)/(score_max-score_min))
        ax.plot(score_threshold, np.mean(filtered_z_tomo), 'o', color=col)
        ax.plot(score_threshold, np.mean(filtered_x_tomo), 'o', color=col)

        ## at each score threshold, how does tomographic average distribute
        bins, corrTomo_x, err = util.correlate_tomography(filtered_x_scores, filtered_x_tomo, bin_min=score_min, bin_max=score_max)
        bins, corrTomo_z, err = util.correlate_tomography(filtered_z_scores, filtered_z_tomo, bin_min=score_min, bin_max=score_max)
        r = (score_threshold-score_min) /(score_max-score_min)
        bloch.plot( r*corrTomo_x, r*corrTomo_z, 'o', color=col, ms=3)
    

    ax.set_xlabel("Score threshold", fontsize=20)
    ax.set_ylabel("Avg Tomo", fontsize=20)
    util.make_bloch(bloch)

    plt.show()

##END check_scoreThreshold


