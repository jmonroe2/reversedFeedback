import numpy as np

def theory_xz(rs, z0=0):
    S = 0.41 # copied '' from "calibrate readout" ''
    dV = 3.31 # copied '' from "calibrate readout" ''
    gammaT = 0.45; # from a guess

    ## new guesses to tune theory to exp
    S = 0.357 # from p. 14 of "Debugging ACOF" physical notes
    dV = 3.08 # ibid, change sign to match "debugging ACOF 07/02/18" document p. 14
    
    z = np.tanh(rs*S/2/dV - np.arctan(z0))
    x = np.sqrt(1-z**2)*np.exp(-gammaT)
    return x,z    
##END theory_xz


def correlate_tomography(to_bin, tomographic, bin_min=None, bin_max=None,num_bins=30, verbose=False):
    '''
    DESC: averages tomographic outcomes for bins of 
    OUTPUT: (tomography bins, tomography averages, tomography binomial errors)
    '''
    ## bin the array to be binned
    # set of the bin range (if provided)
    if (bin_min is not None) and (bin_max is not None):
        bins = np.linspace(bin_min, bin_max, num_bins)
    else:
        bins = num_bins

    # histogram
    hist_values, bin_edges = np.histogram(to_bin, bins=bins)
    
    ## in each bin, get the bin's tomographic outcome and average sign.
    avg_tomo, avg_tomo_err = np.zeros((2,num_bins))
    for bin_index in range(len(hist_values)):
        left_bound  = bin_edges[bin_index]
        right_bound = bin_edges[bin_index+1]
        
        tomo_inBin = tomographic[(left_bound < to_bin) & (to_bin<right_bound)]
        if len(tomo_inBin): ## skip empty bins
            avg_tomo[bin_index] = np.mean(tomo_inBin)
            avg_tomo_err[bin_index] = binomial_error(tomo_inBin)
        else:
            avg_tomo[bin_index] = np.nan
            avg_tomo_err[bin_index] = np.nan
        if len(tomo_inBin) <= 3 and verbose:
            print(left_bound, right_bound)
            print(bin_index, tomo_inBin)
    
        binned = to_bin[(left_bound < to_bin) & (to_bin<right_bound)]
        n = len(binned)
    if verbose: print(bin_xs)
    ''' # more explicit version
    ## calculate average outcome for tomography in each bin
    tomo = np.zeros(num_bins)
    tomo_err = np.zeros(num_bins)
    sorted_toBin = np.sort(to_bin)
    data_index = 0
    #point = sorted_toBin[0]
    readout = tomo[0]  
    for bin_index, bin_thresh in enumerate(bin_xs):
        for data_index, datum in enumerate(to_bin):
            if bin_xs[bin_index-1] < datum < bin_thresh:
                readout = tomographic[data_index]
                tomo[bin_index] += np.sign(threshold - readout )
        ##END loop through points belonging to bin
        N = hist_values[bin_index]
        tomo[bin_index] /= N
        ## calculate 95% CI for binomial error
        p = 0.5*tomo[bin_index] + 0.5 # convert from expectation value to probability
        tomo_err[bin_index] = 1.96 *np.sqrt(p*(1-p)/N) /2
        tomo_err[bin_index] *= 2 # convert back to expectation values.
    ##END loop through bins
    '''
    
    ## output
    return bin_edges, avg_tomo, avg_tomo_err
##END correlate_tomography


def binomial_error(binary_outcomes):
    ## DESCR: calculate the binomial confidence interval 
    ##      see: wiki/Binomial_proportion_confidence_interval
    ## INPUT: binary should be list of +/-1

    z = 1.96  ## 95% confidence interval
    p = np.mean( 0.5*(binary_outcomes+1) )
    N = len(binary_outcomes)
    err = z*np.sqrt(p*(1-p)/N)
    return err*2 ## convert back to expectation value in [-1,1]

    ## alternate method (I think a bit slower, but equivalent)
    #n_succ = np.sum( binary_outcomes>0 )
    #n_fail = N-n_succ
    #err = z/N *np.sqrt(n_succ*n_fail/N) 
##END binomial_err 


def make_bloch(axes_object):
    ## in-place adjustment of axes
    fs = np.linspace(0,2*np.pi, 100)
    xs = np.cos(fs)
    ys = np.sin(fs)
    axes_object.plot(xs,ys,'k--')

    axes_object.set_xlim(-1,1)
    axes_object.set_ylim(-1,1)

    axes_object.set_xlabel(r"$\langle X\rangle$")
    axes_object.set_ylabel(r"$\langle Z\rangle$")

    axes_object.set_aspect('equal')

##END make_bloch
