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


def check_sequence_average(tomo, num_rotations):
    '''
    DESC: view reps of the full sequence: prep, (weak), rot. by 'feedback' angle, tomo; 
            26 angles in x, 26 in y, 26 in z
    #''' 

    fig, axes = plt.subplots()
    for i in range(num_rotations*3):
        setOf_angle_tomo = tomo[i::num_rotations*3]
        plt.plot(i, np.mean(setOf_angle_tomo), 'ok') 
    axes.set_xlabel("Sequence step")
    axes.set_ylabel("Average Tomography")
    axes.set_ylim(-1,1)
    axes.set_title("Check Sequence Averaging")
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
    tomo_ax.set_title("Correleted Tomography")
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
    fig.sca(bloch)
    plt.scatter(binned_xTomo, binned_zTomo,s=20, c=bins, cmap='nipy_spectral')
    plt.colorbar(label=r"$\theta_{app}/\pi$")
    util.make_bloch(bloch)

    bloch.set_ylim(-1,1)
    #bloch.set_title("Scores as Applied angle") ##@@ THIS GRAPH IS UNLABELED UNTIL THE FUNC IS CLEANED UP

    ## compare to expected rotation
    using_plusX_init = True
    if using_plusX_init:
        r = 0.6 ## from zero-angle maximum x-coordinate
        theta = 0
    else:
        r = 0.7
        theta = np.pi/4
        
    angles = bins
    fig, rot_ax = plt.subplots()
    rot_ax.errorbar( bins, binned_zTomo,yerr=binned_zTomo_err, color='k',fmt='o')
    rot_ax.plot( angles, r*np.sin(np.pi*angles - theta))
    rot_ax.set_xlabel("Applied Angle [$\pi$]")
    rot_ax.set_ylabel("Z Tomo")
##END check_scores_as_appliedAngle


def check_scoreThreshold(scores, tomo, num_rotations=26):
    '''
    '''
    fig1, bloch = plt.subplots()
    fig2, ax = plt.subplots()
    
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
    bloch.set_title(f"Score Threshold {score_min}-{score_max}")
    ax.set_title("Score Threshold")
##END check_scoreThreshold


def check_probQ_theory_consistency(show=True):
    qs = np.linspace(0.1,0.3, 100)

    # find inverse gamma via Q
    gamma = np.arccosh(np.exp(qs/2))
    zf = np.tanh(gamma)

    S = 0.41  # see calc_AoT()
    dV = 3.31 # see calc_AoT()
    tT = dV**2/S ## tau over T via equating Gaussian variance
    r = gamma *tT
    Pf = 0.5*( np.exp(-(r-1)**2/2/tT) + np.exp(-(r+1)**2/2/tT) )
    Pf /= np.sqrt(2*np.pi/tT)#*2 ## I had an extra factor of 2...
    lhs = Pf/2/zf
    
    
    ## check other function's conversion
    #equiv to rhs = get_analytic_probQ(qs)
    arg = -0.5/tT -0.5*tT*(np.arccosh(np.exp(qs/2)))**2
    zf_q = np.sqrt(np.exp(2*qs)/(np.exp(qs)-1))
    norm = 0.5*np.sqrt(tT/2/np.pi) 
    rhs = norm*zf_q*np.exp(arg)
    
    ## results from below insanity check
    #rhs = np.cosh(gamma)*np.exp(-0.5/tT -0.5*tT*gamma**2)
    #rhs /= 2np.tanh(gamma)*np.sqrt(2*np.pi*tT) 
    
    ## insanity check
    #lhs = -(r-1)**2/2/tT
    # check definition of r
    #lhs = -(r-1)**2/2/tT
    #rhs = -0.5*(tT*gamma-1)**2/tT 
    # check expansion
    #lhs = -(r-1)**2/2/tT
    #rhs = -0.5*( (tT*gamma)**2 - 2*tT*gamma +1)/tT
    # check simplification
    #lhs = -(r-1)**2/2/tT
    #rhs = -0.5*tT*gamma**2 +gamma -0.5/tT
    # check exponentiation
    #lhs = np.exp(-(r-1)**2/2/tT)
    #rhs = np.exp(-0.5*tT*gamma**2 +gamma -0.5/tT)
    # check double components
    #lhs = np.exp(-(r-1)**2/2/tT) + np.exp(-(r+1)**2/2/tT)
    #rhs = np.exp(-0.5*tT*gamma**2 +gamma -0.5/tT)
    #rhs += np.exp(-0.5*tT*gamma**2 -gamma -0.5/tT)
    # check factoring
    #lhs = np.exp(-(r-1)**2/2/tT) + np.exp(-(r+1)**2/2/tT)
    #rhs = (np.exp(gamma)+np.exp(-gamma))*np.exp(-0.5*tT*gamma**2 -0.5/tT) 
    # check cosh equivalence
    #lhs = 2*np.cosh(gamma)
    #rhs = np.exp(gamma) + np.exp(-gamma)
    #lhs = np.exp(-(r-1)**2/2/tT) + np.exp(-(r+1)**2/2/tT)
    #rhs = np.cosh(gamma)*np.exp(-0.5*tT*gamma**2 -0.5/tT) 
    
    fig, ax = plt.subplots()
    ax.plot(qs, lhs, 'k-')
    ax.plot(qs, rhs, 'r-')
    ax.set_title("check_probQ_theory_consistency()")
    if show: plt.show()
##END check_probQ_theory_consistency


