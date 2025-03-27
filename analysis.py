#analysis.py
#Jonathan Borowsky
#2/21/25

#functions for comparing the results of different methods 
# of estimating energy lanscapes and kinetics 
# to the true landscapes and to each other

################################################################################################################

import numpy as np
import MSM_methods
import matplotlib.pyplot as plt


#----------------------------------------------------------------------------------------------------------------
#compare true and estimated populations for a single estimation method

#parameters
# system: the system object
# kT: the temperature of the system
# coordinates: the x-coordinates at which equilibrium populations have been estimated
# eq_pops_simulation: the estimated equilibrium populations
# metrics: a list of metrics to compute`
# ensemble_data: a list of tuples of (x-coordinates, populations) for each ensemble of a history augmented MSM

#returns
# returns: a dictionary of metrics computed`
def landscape_comparison(system, kT, coordinates, eq_pops_simulation, metrics = [], ensemble_data = []):

    #compute true populations 
    eq_pops_analytic, energies_analytic = system.normalized_pops_energies(kT, coordinates)

    #unnecessary extra precision; requires bin boundaries as an argument
    #bincenters, eq_pops_analytic = system.compute_true_populations(bins, kT, tolerance = 0.01)
    #plt.plot(bincenters, eq_pops_analytic)
    
    plt.plot(coordinates, eq_pops_analytic, linestyle="dashed")
    plt.plot(coordinates, eq_pops_simulation)

    #plot energy landscapes for each ensemble of a history augmented MSM if provided
    if ensemble_data:
        for xei, xpi in zip(ensemble_data[0], ensemble_data[1]):
            plt.plot(xei, xpi)
    
    plt.show()

    mean_binwidth = (coordinates[-1]-coordinates[0])/(len(coordinates)-1)

    metrics_out = {}

    #note that these metrics currently do not care how badly mis-localized any misplaced probability density is; 
    # there's probably a KDE approach that would do better with that
    if "rmsew" in metrics:
        #without weighting, extending the region over which the RMS error is computed makes it look artificially better because there are large areas where the true and estimated probability are both about 0
        rmse_weighted = np.sqrt(np.mean([epa*(eps-epa)**2 for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)])/mean_binwidth**3)
        print(f"weighted RMSE = {rmse_weighted}")
        metrics_out["rmsew"] = rmse_weighted

    if "maew" in metrics:
        #without weighting, extending the region over which the mean absolute error is computed makes it look artificially better because there are large areas where the true and estimated probability are both about 0
        mae_weighted = np.mean([epa*abs(eps-epa) for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)])/mean_binwidth**2
        print(f"weighted MAE = {mae_weighted}")            
        metrics_out["maew"] = mae_weighted

    return metrics_out
        
    #kl divergence has bad numerical properties when any estimated entries are 0, which some usually are
    #kl_divergence = sum([epa*np.log(epa/eps) for epa, eps in zip(eq_pops_analytic, eq_pops_simulation)])
    #print(f"kl divergence = {kl_divergence}")


#----------------------------------------------------------------------------------------------------------------
#helper function for printing mean first passage times neatly from matrices thereof

#parameters
# mfpts: a matrix of mean first passage times
# digits: the number of decimal places to print

#returns
# returns: None
def print_mfpts_2states(mfpts, digits = 0):
    inter_well_mpfts = [mfpts[0,1], mfpts[1,0]]

    meanfmt = f"{np.mean(inter_well_mpfts):.{digits}f}"
    stdfmt = f"{np.std(inter_well_mpfts):.{digits}f}"
    print(f"MFPT = {meanfmt}+-{stdfmt} steps")


#----------------------------------------------------------------------------------------------------------------
#construct a history augmented markov state model from a set of transitions

#parameters
# ha_transitions: a list of transitions, each of which is a tuple of (from_state, to_state)
# nbins: the number of bins to use for the analysis
# system: the system object
# save_period: the number of time steps between frames 
#   (assumed to be equal to the lag time since haMSMs do not require 
#   a lag time sufficient for their microstates to be markovian)
# show_TPM: whether to show the transition probability matrix

#returns
# ha_x_config: the x-coordinates of the bins for which populations were computed (same as bincenters??)
# ha_eqp_config: the equilibrium populations of the configurational bins derived by summing over all ensembles
# x_ensembles: a list of lists of x-coordinates for each ensemble for which state populations were computed
# p_ensembles: a list of lists of equilibrium populations for each ensemble for which state populations were computed
# mfpts: a matrix of mean first passage times between the macrostates
#   (the first index is the from_state and the second index is the to_state)

def hamsm_analysis(ha_transitions, nbins, system, save_period, lag_time=1, show_TPM=False):

    #for consiceness
    nm = system.n_macrostates

    #get bin boundaries
    binbounds, bincenters, step = system.analysis_bins(nbins)

    #assign the bins to macrostates
    macrostates_discrete = [system.macro_class(x) for x in bincenters]

    #-----------------------------------------------------------------------------------------------------------------
    #build MSM
    tpm, states_in_order = MSM_methods.transitions_2_msm(ha_transitions)
    if show_TPM:
        plt.matshow(tpm)
        plt.show()

    eqp_msm = MSM_methods.tpm_2_eqprobs(tpm)


    #-----------------------------------------------------------------------------------------------------------------
    #get populations in configuration space (along x) for each ensemble

    x_ensembles = [[] for element in range(nm)]
    p_ensembles = [[] for element in range(nm)]

    for i, so in enumerate(states_in_order):
        for j in range(nm):
            if so%nm == j:
                x_ensembles[j].append(bincenters[int(so//nm)])
                p_ensembles[j].append(eqp_msm[i][0])


    #-----------------------------------------------------------------------------------------------------------------
    #assemble halves of the energy landscape to get the overall energy

    all_eqp_config = []

    ha_x_config = []
    ha_eqp_config = []
    
    for i in range(0, len(bincenters)*2, 2):
        
        prob_all_ensembles = sum([eqp_msm[states_in_order.index(i+j)][0] if i+j in states_in_order else 0 for j in range(nm)])
        all_eqp_config.append(prob_all_ensembles)
        
        if sum([1 if i+j in states_in_order else 0 for j in range(nm)]) > 0:
            ha_x_config.append(bincenters[int(i/2)]) # I think it's fine to just return bincenters
            ha_eqp_config.append(prob_all_ensembles)


    #-----------------------------------------------------------------------------------------------------------------
    #calculate mfpts
    mfpts = MSM_methods.calc_ha_mfpts(states_in_order, eqp_msm, tpm, macrostates_discrete, nm, save_period*lag_time)


    return bincenters, all_eqp_config, ha_x_config, ha_eqp_config, x_ensembles, p_ensembles, mfpts


#----------------------------------------------------------------------------------------------------------------
#bootstrapping

#parameters
# n_bootstrap: number of times to run each method
# analysis_methods: a list of analysis methods to compare
#   each method should be a function that takes the parameters below and returns a tuple of (coordinates, populations, mfpts)
# system: the system object
# kT: the temperature of the system
# dt: the time step of the simulation
# propagator: the propagator to use for the simulation
# nsteps: the number of steps to run the simulation
# save_period: the number of time steps between saved frames
# n_parallel: the number of parallel simulations to run

#returns
# mean first passage times and populations for each method, complete with standard deviations

def bootstrap_method_comparison(n_bootstrap, analysis_methods, system, kT, dt, aggregate_simulation_limit, n_parallel, save_period, n_analysis_bins):
    
    #should we just rely on the user to check this?
    #is there some better way to do these checks?
    if n_bootstrap < 2:
        print("error: n_bootstrap must be at least 2")

    mfpts_all = []
    populations_all = []

    #loop over methods
    for m in analysis_methods:
        print(m)

        method_mfpts = []
        method_coords = []
        method_probabilities = []

        #replicates for each method
        for bi in range(n_bootstrap):
            print(f"round {bi}")

            aggregate_simulation, coords, probs, mfpts = m(system, kT, dt, aggregate_simulation_limit, n_parallel, save_period, n_analysis_bins)
            method_mfpts.append(mfpts)
            method_coords.append(coords)
            method_probabilities.append(probs)

        #TODO we ought to add code to detect systematic asymmetries between forward and reverse MFPTs
        mfpts_3d = np.vstack(method_mfpts)
        mfpts_all.append([np.mean(method_mfpts, axis=0), np.std(method_mfpts, axis=0)]) 
        
        #print(len(method_probabilities))
        #print([len(mp) for mp in method_probabilities])

        #calculate energy landscape with error bars, 
        # accounting for the fact that not all landscape estimators will yield estimates 
        # for the same states due to ergodic trimming of MSMs
        # all msm equilibrium probabilities should be nonzero;
        # even a very low probability is different from never having seen the state

        #get a list of all coordinates which appeared in any MSM
        method_coords_flat = [c for mci in method_coords for c in mci]
        method_coords_all = np.unique(method_coords_flat)

        mean_probs = []
        mean_probs_err = []
        for mc in method_coords_all:

            #collect all probabilities estimated for the each sampled state
            probs_c = []

            for i, method_c in enumerate(method_coords):
                if mc in method_c:
                    probs_c.append(method_probabilities[i][np.where(method_c == mc)[0][0]])
            
            mean_probs.append(np.mean(probs_c))

            #standard deviation (dispersion, not a confidence interval) of estimates; 
            # states for which probabilities were estimated only once are denoted with -1 as no standard deviation can be estimated
            if len(probs_c) > 1:
                mean_probs_err.append(np.std(probs_c))
            else:
                mean_probs_err.append(-1)

        populations_all.append([method_coords_all, mean_probs, mean_probs_err])

    return mfpts_all, populations_all 




def plot_bootstrapping_results(populations_all, system, kT, n_analysis_bins):
    
    binbounds, bincenters, step = system.analysis_bins(n_analysis_bins)

    eqp_analytic = [np.exp(-system.potential(x)/kT) for x in bincenters]
    eqp_sum = sum(eqp_analytic)
    eqp_analytic = [ea/eqp_sum for ea in eqp_analytic]

    plt.plot(bincenters, eqp_analytic, color="black")

    colorlist = ["red", "green", "blue"]

    for cx, method_data in enumerate(populations_all):
        for mci, mpi, mpei in zip(method_data[0], method_data[1], method_data[2]):
            
            if mpei == -1:
                plt.scatter(mci, mpi, color=colorlist[cx], marker=".")
            else:
                plt.errorbar(mci, mpi, mpei, color=colorlist[cx], marker="_")

